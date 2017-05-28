#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import data_helpers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#from tensorflow.contrib.keras.api.keras.models import Model
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.layers.embeddings import Embedding
from keras.layers import Embedding, Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate, LSTM, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.backend import expand_dims, reshape
from keras.layers.merge import concatenate
from keras.regularizers import l2

from gensim.models import KeyedVectors
#from keras.engine import topology
# set default encoding
# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("valid_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("quora_train_file", "./data/quora/train.csv", "Quora train data.")
tf.flags.DEFINE_string("test_data_file", "./data/quora/test.csv", "Test Data file")
tf.flags.DEFINE_string("embedding_file", "./data/quora/GoogleNews-vectors-negative300.bin", "Embedding Data file")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 300, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("max_nb_words", 200000, "number of words taking into count. 125642")
tf.flags.DEFINE_integer("max_sequence_length", 30, "number of words in sequence taking into count.")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 2048, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("output_every", 1, "Output model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_bool("re_weight", True, "whether to re-weight classes to fit the 17.5% share in test set")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================
#####################################################
## index word vectors
#####################################################
print('Indexing word vectors')
word2vec = KeyedVectors.load_word2vec_format(FLAGS.embedding_file, binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))
# 120600

#####################################################
## Prepare SpellCheck
#####################################################
#print('Preparing SpellCheck')
#WORDS = {}
#for i,word in enumerate(word2vec.index2word):
#    WORDS[str(word)] = i

####################################################
# Load data
####################################################
print("Loading data...")

import cPickle as pickle
train_data_path = 'data/quora/train.pkl'
test_data_path = 'data/quora/test.pkl'

if os.path.exists(train_data_path):
    x_text_q1, x_text_q2, y = pickle.load(open(train_data_path, "r"))
else:
    x_text_q1, x_text_q2, y = data_helpers.load_data_and_labels_from_quora(FLAGS.quora_train_file)
    pickle.dump((x_text_q1, x_text_q2, y), open(train_data_path, "w"))

print("train data shape ", len(x_text_q1) )
print("train label shape ", len(y))
print x_text_q1[0]

if os.path.exists(test_data_path):
    test_id, test_text_q1, test_text_q2 = pickle.load(open(test_data_path, "r"))
else:
    test_id, test_text_q1, test_text_q2 = data_helpers.load_data_from_quora(FLAGS.test_data_file)
    pickle.dump((test_id, test_text_q1, test_text_q2), open(test_data_path, "w"))

print("test data shape", len(test_text_q1))
print test_text_q1[0]
   
####################################################
# Build vocabulary
####################################################
print("Building vocabulary...")
tokenizer = Tokenizer(num_words=FLAGS.max_nb_words)
tokenizer.fit_on_texts(x_text_q1 + x_text_q2 + test_text_q1 + test_text_q2) 

sequences_1 = tokenizer.texts_to_sequences(x_text_q1)
sequences_2 = tokenizer.texts_to_sequences(x_text_q2)
test_sequences_1 = tokenizer.texts_to_sequences(test_text_q1)
test_sequences_2 = tokenizer.texts_to_sequences(test_text_q2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
# 120500


data_1 = pad_sequences(sequences_1, maxlen=FLAGS.max_sequence_length)
data_2 = pad_sequences(sequences_2, maxlen=FLAGS.max_sequence_length)

print('Shape of data tensor:', data_1.shape)
print('data type:', data_1.dtype)
print('Shape of label tensor:', len(y))

test_data_1 = pad_sequences(test_sequences_1, maxlen=FLAGS.max_sequence_length)
test_data_2 = pad_sequences(test_sequences_2, maxlen=FLAGS.max_sequence_length)



#####################################################
## prepare embeddings
#####################################################
print('Preparing embedding matrix')
nb_words = min(FLAGS.max_nb_words, len(word_index))+1
embedding_matrix = np.zeros((nb_words, FLAGS.embedding_dim))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
# full 61943
# mid  61882
# first 61789
# none translate 72131
# remove stop words 72131(120587)


#exit()

## sample train/validation data
#np.random.seed(1234)

y = np.array(y)
perm = np.random.permutation(len(y))
idx_train = perm[:int(len(y)*(1-FLAGS.valid_sample_percentage))]
idx_val = perm[int(len(y)*(1-FLAGS.valid_sample_percentage)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((y[idx_train], y[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((y[idx_val], y[idx_val]))

print('Shape of train data tensor:', data_1_train.shape)
print('Shape of train label tensor:', labels_train.shape)

print('Shape of valid data tensor:', data_1_val.shape)
print('Shape of valid label tensor:', labels_val.shape)   



########################################
## add class weight
########################################
if FLAGS.re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

weight_val = np.ones(len(labels_val))
if FLAGS.re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344
       
# Training
# =================================================
###################################################   
## TextCNN
###################################################
embedding_size=FLAGS.embedding_dim
filter_sizes=list(map(int, FLAGS.filter_sizes.split(",")))
num_filters=FLAGS.num_filters

embedding_layer = Embedding(input_dim=nb_words,
    output_dim=embedding_size,
    weights=[embedding_matrix],
    input_length=FLAGS.max_sequence_length,
    trainable=False)

merging_list = []
sequence_input = Input(shape=(FLAGS.max_sequence_length,), dtype='int32')
for filter_size in filter_sizes:
    embedded_chars_q = embedding_layer(sequence_input)
    conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu', \
                   )(embedded_chars_q)
    drop = Dropout(FLAGS.dropout_keep_prob)(conv)
    pooled = MaxPooling1D(FLAGS.max_sequence_length-filter_size+1)(drop)
    
#    conv = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(pooled)
#    drop = Dropout(FLAGS.dropout_keep_prob)(conv)
#    pooled = MaxPooling1D(pool_size=filter_size)(drop)
    
    flat = Flatten()(pooled)
    merging_list.append(flat)
    
merged_seq = concatenate(merging_list)

question_model = Model(sequence_input, merged_seq)
sequence_1_input = Input(shape=(FLAGS.max_sequence_length,), dtype='int32')
sequence_2_input = Input(shape=(FLAGS.max_sequence_length,), dtype='int32')

q1 = question_model(sequence_1_input)
q2 = question_model(sequence_2_input)

merged = concatenate([q1, q2])
merged = Dropout(FLAGS.dropout_keep_prob)(merged)
merged = BatchNormalization()(merged)

dense = Dense(128, activation='relu')(merged)
dense = Dropout(FLAGS.dropout_keep_prob)(dense)
dense = BatchNormalization()(dense)

#dense = Dense(128, activation='relu')(dense)
#dense = Dropout(FLAGS.dropout_keep_prob)(dense)

preds = Dense(1, activation='sigmoid')(dense)

# seq_len=200 loss 0.26;0.5211 accuracy 0.8077,7708
# seq_len=50  loss 0.23;0.43 accuracy 0.8277,789
# seq_len=30  loss 0.21;0.40 - val_acc: 0.80,0.8048 test:0.308
########################################
## TextRNN
########################################

#EMBEDDING_DIM = 300
#MAX_SEQUENCE_LENGTH = FLAGS.max_sequence_length
#rate_drop_dense = 0.5
#rate_drop_lstm = 0.5
#num_lstm = 200
#num_dense= 100
#embedding_layer = Embedding(nb_words,
#        EMBEDDING_DIM,
#        weights=[embedding_matrix],
#        input_length=MAX_SEQUENCE_LENGTH,
#        trainable=False)
#lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
#
#sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#embedded_sequences_1 = embedding_layer(sequence_1_input)
#x1 = lstm_layer(embedded_sequences_1)
#
#sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#embedded_sequences_2 = embedding_layer(sequence_2_input)
#y1 = lstm_layer(embedded_sequences_1)
#
#merged = concatenate([x1, y1])
#merged = Dropout(rate_drop_dense)(merged)
#merged = BatchNormalization()(merged)
#
#merged = Dense(num_dense, activation='relu')(merged)
#merged = Dropout(rate_drop_dense)(merged)
#merged = BatchNormalization()(merged)
#
#preds = Dense(1, activation='sigmoid')(merged)

# BiLSTM epoch 4: val_loss 0.6699 - val_acc: 0.6634
########################################
## train the model
########################################
model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)      
model.compile(loss='binary_crossentropy',
        optimizer='Adam',
        metrics=['accuracy'])

# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
os.mkdir(out_dir)
print("Writing to {}\n".format(out_dir))

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = os.path.join(out_dir, 'model.h5')
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val), \
        epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds = model.predict([test_data_1, test_data_2], batch_size=FLAGS.batch_size, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=FLAGS.batch_size, verbose=1)
preds /= 2

submission = pd.DataFrame({'test_id':test_id, 'is_duplicate':preds.ravel()})
bst_data_path = os.path.join(out_dir, 'sumbit.csv')
submission.to_csv(bst_data_path, index=False)

#%%