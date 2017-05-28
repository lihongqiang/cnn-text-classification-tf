import numpy as np
import re
from collections import Counter
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

# set default encoding
# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class SpellChecker():
    
    def __init__(self, WORDS):
        self.WORDS = WORDS
        
    def words(self, text): return re.findall(r'\w+', text.lower())
    
    def P(self, word): 
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.WORDS.get(word, 0)
    
    def correction(self, word): 
        "Most probable spelling correction for word."
        return max(self.candidates(word), key=self.P)
    
    def candidates(self, word): 
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
    
    def known(self, words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)
    
    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self, word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, WORDS=None, remove_stopwords=False, stem_words=False, spell_check=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    
    # Convert words to lower case and split them
    text = text.lower()
    
    
    if spell_check and WORDS :
        sc = SpellChecker(WORDS)
        text = [sc.correction(w) for w in text.split()]

    
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    
#    text = re.sub(r" m ", " am ", text)
#    text = re.sub(r"quikly", "quickly", text)
#    text = re.sub(r" usa ", " america ", text)
#    text = re.sub(r" uk ", " england ", text)
#    text = re.sub(r"imrovement", "improvement", text)
#    text = re.sub(r"intially", "initially", text)
#    text = re.sub(r" dms ", "direct messages ", text)  
#    text = re.sub(r"demonitization", "demonetization", text) 
#    text = re.sub(r"actived", "active", text)
#    text = re.sub(r"kms", " kilometers ", text)
#    text = re.sub(r" cs ", " computer science ", text) 
#    text = re.sub(r" upvotes ", " up votes ", text)
#    text = re.sub(r"\0rs ", " rs ", text) 
#    text = re.sub(r"calender", "calendar", text)
##    text = re.sub(r"ios", "operating system", text)
#    text = re.sub(r"programing", "programming", text)
#    text = re.sub(r"bestfriend", "best friend", text)
#    text = re.sub(r"III", "3", text) 
#    text = re.sub(r"the us", "america", text)
    
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text.split() if not w in stops]
        text = " ".join(text)
        
    # Optionally, shorten words to their stems
    if stem_words:
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text.split()]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_from_quora(data_file, WORDS=None):
    """
    Loads quora data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    examples = pd.read_csv(data_file, encoding='utf-8')
    # Fill NAN
    examples = examples.fillna('')
    # Split by words
    x_text_q1 = examples['question1'].apply(lambda x: str(x)).tolist()
    x_text_q1 = [text_to_wordlist(sent, WORDS) for sent in x_text_q1]
    x_text_q2 = examples['question2'].apply(lambda x: str(x)).tolist()
    x_text_q2 = [text_to_wordlist(sent, WORDS) for sent in x_text_q2]
    # Generate labels
    y = examples['is_duplicate'].tolist()
    y = [int(label) for label in y]
    return [x_text_q1, x_text_q2, y]

def load_data_from_quora(data_file, WORDS=None):
    """
    Loads quora data from files, splits the data into words.
    Returns split sentences.
    """
    # Load data from files
    examples = pd.read_csv(data_file, encoding='utf-8')
#    examples = examples.iloc[0:1000]
    # Fill NAN
    examples = examples.fillna('')
    # Split by words
    x_text_q1 = examples['question1'].apply(lambda x: str(x)).tolist()
    x_text_q1 = [text_to_wordlist(sent, WORDS) for sent in x_text_q1]
    x_text_q2 = examples['question2'].apply(lambda x: str(x)).tolist()
    x_text_q2 = [text_to_wordlist(sent, WORDS) for sent in x_text_q2]
    # id
    x_id = examples['test_id'].apply(lambda x: str(x)).tolist()
    return [x_id, x_text_q1, x_text_q2]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
#    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
