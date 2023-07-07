import nltk
import numpy as np
nltk.download('punkt') # punkt is a pre-traind tokenizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# This is tokenzing and stemming is the pre-processing of the sentence. 
#This is extablished using the pre-trained tokenizer and stemmer in NLTK

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words (tokenized_sentence, words):
    # Applying stemming
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # Initializing bag with 0 for each word
    bag = np.zeros(len(words),dtype=np.float32)
    for idx , w, in enumerate (words):
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag

# sentence = ['Hey', 'How', 'are', 'you']
# words = [ 'Hello', 'Good', 'day', 'Bye', 'See', 'you','Hey','are']

# test = bag_of_words(sentence,words)
# print(test)

# a = ["Universe","university","universal"]
# sw = [stem(w) for w in a]
# print((sw))#