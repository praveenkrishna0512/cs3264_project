# Jai Aslam, Anirban Sharma & Zhichao Carton Zeng, Nov 2022
# Kaggle challenge of automated essay grader

import numpy as np
import time
import sys
import nltk
import transformers as ppb
import torch
import pickle

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
    
    def bert_encoding(self, text):
        num_input_ids = 1
        batch_size = 1
        num_batches = int(num_input_ids/batch_size)

        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        bert_tokenizer = tokenizer_class.from_pretrained(pretrained_weights, truncation = True, padding = True)
        model = model_class.from_pretrained(pretrained_weights)  
        for i in range(num_batches):
            tokenizer_output = bert_tokenizer(text, add_special_tokens=True, truncation = True, padding = True, return_attention_mask = True, return_tensors = "pt")
            input_ids = tokenizer_output['input_ids']
            attention_mask = tokenizer_output["attention_mask"]
            with torch.no_grad(): 
                last_hidden_states = model(input_ids, attention_mask=attention_mask)
            return last_hidden_states[0][:,0,:].numpy()

    def svm_grading(self):
        encoded_text = self.bert_encoding(self.raw_text)
        grade_categories = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
        grades = []
        for grade_category in grade_categories:
            with open("models/" + grade_category + 'model.pkl', 'rb') as f:
                 curr_clf = pickle.load(f)
            grades += [(curr_clf.predict(encoded_text)[0])/2]
        print(grades)
        return grades[0], grades[1], grades[2], grades[3], grades[4], grades[5]

            
    def rand_grading(self):
        # just an example/template of grading model, using random number generator

        cohesion, syntax, vocabulary, phraseology, grammar, conventions = 5*np.random.rand(), 5*np.random.rand(), 5*np.random.rand(), 5*np.random.rand(), 5*np.random.rand(), 5*np.random.rand()
        return cohesion, syntax, vocabulary, phraseology, grammar, conventions

