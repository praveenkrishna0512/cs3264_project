# Jai Aslam, Anirban Sharma & Zhichao Carton Zeng, Nov 2022
# Kaggle challenge of automated essay grader

import numpy as np
import time
import sys
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from autocorrect import Speller
spell = Speller(lang='en')
tokenizer = RegexpTokenizer(r'\w+')

class essay(object): 
    def __init__(self, raw_text):
        self.raw_text = raw_text

    def correct_spellings(self):
        essay_tok = tokenizer.tokenize(self.raw_text)
        cnt = 0
        essay_new = []
        old_word = []
        new_word = []
        for wd in essay_tok:
            wd_new = spell(wd)
            if wd_new != wd:
                cnt += 1
                old_word.append(wd)
                new_word.append(wd_new)
            essay_new.append(str(wd_new)) 
        return (' '.join(essay_new)), cnt, old_word, new_word

    def rand_grading(self):
        # just an example/template of grading model, using random number generator

        cohesion, syntax, vocabulary, phraseology, grammar, conventions = 5*np.random.rand(), 5*np.random.rand(), 5*np.random.rand(), 5*np.random.rand(), 5*np.random.rand(), 5*np.random.rand()
        return cohesion, syntax, vocabulary, phraseology, grammar, conventions

