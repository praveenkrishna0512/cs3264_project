# %%
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Flatten,Reshape
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
import csv
import nltk
from tensorflow.keras.models import Model

# %%
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Input,Concatenate,Dropout,Dense,BatchNormalization,Conv1D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import scipy
from tensorflow.keras.initializers import he_normal,glorot_normal
from tensorflow.keras.regularizers import l1,l2
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from time import time
from tensorflow.keras.utils import plot_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,BatchNormalization,Dropout
from tensorflow.keras import optimizers
import random as rn
import string
from sklearn.metrics import f1_score
from tensorflow import keras
import numpy as np
import datetime
import os
import math
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.initializers import glorot_uniform,glorot_normal
from tensorflow.keras.layers import MaxPooling1D

# %%
#loading the training data
training_data=pd.read_csv("./data/train.csv")
# display(training_data.head(2))


# %% [markdown]
# # Preprocessing. 
# 
# Preprocessing includes removing spaces, special characters, contractions and stop words. 

# %%
# ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
def decontractions(phrase):
   
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase

#preprocessing: replacing special characters and space and make text lowercase

from nltk.corpus import stopwords
from tqdm import tqdm
import re
stopwords = stopwords.words('english')
def preprocess(text_col,stopword):
    preprocessed = []
    for sentence in tqdm(text_col.values):
        # Replace "carriage return" with "space".
        sentence=str(sentence)
        sent = sentence.replace('\\r', ' ')
        # Replace "quotes" with "space".
        sent = sent.replace('\\"', ' ')
        # Replace "line feed" with "space".
        sent = sent.replace('\\n', ' ')
         # Replace characters between words with "space".
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        #remove stop words
        #decontraction
        sent=decontractions(sent)
        if stopword:
            sent = ' '.join(e for e in sent.split() if e not in stopwords)
        else:
            sent = ' '.join(e for e in sent.split())
        # to lowercase
        preprocessed.append(sent.lower().strip())
    return preprocessed
training_data['full_text']=preprocess(training_data['full_text'],stopword=False)

# %%
training_data.head() 

# %%
#Train -test split, 20% of the data for validation set.
y=training_data[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']]
X=training_data.drop(['text_id','cohesion','syntax','vocabulary','phraseology','grammar','conventions'],axis=1)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20)
print(X_train.shape,y_train.shape)



# %%
#padding to make all the vectors of the same length

def pad_text(text,tokenizer,max_len):
    return pad_sequences(tokenizer.texts_to_sequences(text),maxlen=max_len,padding='post')


def text_padding(train,test,max_len):
    vocab=5000
    token=Tokenizer()
    token.fit_on_texts(train)
    padded_train_text=pad_text(train,token,max_len)
    padded_test_text=pad_text(test,token,max_len)
    return padded_train_text,padded_test_text,token
comm_len=200
train_com_pad,test_com_pad,token_com= text_padding(X_train['full_text'],X_test['full_text'],comm_len)

print(train_com_pad.shape,test_com_pad.shape)

# %%
def generate_embedding_matrix(token):
    embedding_path='./crawl-300d-2M.vec' #pre trained FastText English word vectors released by FB
    embedding_size=300
    vocab_size=5000
    embedding_index={}
    with open(embedding_path, 'r',encoding="utf8") as f:
         for line in f:
                values = line.rstrip().rsplit(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs
    num_words = len(token.word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_size))
    for word, i in token.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# %% [markdown]
# # Generate the embedding matrix or use the already stored one

# %%
# generating the embedding matrix containing (if not already generated)
embedding_comm = generate_embedding_matrix(token_com)
print(embedding_comm.shape)


# %%
#Storing the embedded matrix for trained
np.savetxt('./data/Embedded_matrix_trained_data.csv',embedding_comm,delimiter=',')

# %%
#Use the already stored embedded matrix for the trained data
embedding_comm = np.loadtxt('./data/Embedded_matrix_trained_data.csv', delimiter=',')
#print(embedding_comm1.shape)
#print(embedding_comm1)


# %%
#reshaping the data 
X_train=[train_com_pad,train_com_pad]
X_test=[test_com_pad,test_com_pad]
y_train=np.array(2.0*y_train,dtype=np.float64)
y_test=np.array(2.0*y_test, dtype=np.float64)

# mean columwise rmse
def mcrmse(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=0)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=0)

# %% [markdown]
# # LSTM + CNN Model

# %%
# We will use the Adam optimizer 
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
def LSTM_CNN1D(comm_len,token_com):
    drop_lstm = 0.25
    drop_dense = 0.25
    num_lstm=150
    input_1 = Input(shape=(comm_len,),name = 'input_comment_1')
    embedding_layer_1 = Embedding(len(token_com.word_index) + 1,300,weights=[embedding_comm],input_length=comm_len,trainable=False,dtype=tf.float32)(input_1)
    conv_1_1 = Conv1D(64,3,strides=1, padding='same',activation='relu')(embedding_layer_1)
    lstm_1_1 = LSTM(64,dropout=drop_lstm,return_sequences=True,dtype=tf.float32)(embedding_layer_1)
    concate_1 = keras.layers.Concatenate(axis=-1)([conv_1_1, lstm_1_1])
    flatten_1 = Flatten()(concate_1)

    # creating layers for parent comment data
    input_2 = Input(shape=(comm_len,),name = 'input_comment_2')
    embedding_layer_2 = Embedding(len(token_com.word_index) + 1,300,weights=[embedding_comm],input_length=comm_len,trainable=False,dtype=tf.float32)(input_2)
    conv_1_1 = Conv1D(128,3,strides=1, padding='same',activation='relu')(embedding_layer_2)
    lstm_1_2 =LSTM(128,dropout=drop_lstm,return_sequences=True,dtype=tf.float32)(embedding_layer_2)
    concate_2 = keras.layers.Concatenate(axis=-1)([conv_1_1, lstm_1_2])
    flatten_2 = Flatten()(concate_2)

    
    concatenated_layer = keras.layers.concatenate([flatten_1,flatten_2],axis=-1)

    # creating further layers
    x = Dense(128, activation = 'relu',kernel_initializer=glorot_uniform(seed=42))(concatenated_layer)
    x = BatchNormalization()(x)
    x = Dense(64, activation = 'relu',kernel_initializer=glorot_uniform(seed=42))(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation = 'relu',kernel_initializer=glorot_uniform(seed=42))(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation = 'relu',kernel_initializer=glorot_uniform(seed=42))(x)
    output = Dense(6,activation='linear')(x)
    model = Model(inputs = [input_1,input_2], outputs = [output])
    model.compile(optimizer=adam, loss = mcrmse, metrics = mcrmse)
    return model

# %%
# Loading the saved model
model=LSTM_CNN1D(comm_len,token_com)

# TODO: Enable to load weights
# model.load_weights("./data/LSTM_weights_final.h5")

# %%
#reduce_lr reduces the learning rate when the metric has stoppes improving for 2 epochs. 
#Using EarlyStopping to stop the calculation upon reaching enough accuracy

reduce_lr = ReduceLROnPlateau(monitor = 'val_mcrmse', factor = 0.25, patience = 2, verbose = 1)
earlystop = EarlyStopping(monitor = 'val_mcrmse',  mode="min",min_delta = 0.01, patience = 5,verbose = 1)
callbacks = [reduce_lr,earlystop]

# %%
hitory1=model.fit(x=X_train,y=y_train,epochs=30,batch_size=32,validation_data=(X_test, y_test),callbacks=callbacks)

# %%
#saving the model and weights for future 
from tensorflow.keras.models import Sequential, model_from_json
model_json = model.to_json()
with open("./data/LSTM_model_final.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("./data/LSTM_weights_final.h5")

# %%
# Loading the saved model
json_file=open('./data/LSTM_model_final.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./data/LSTM_weights_final.h5")

# %%
#predicted y
y_pred = model.predict(X_test)

# %% [markdown]
# # Errors

# %%
# calculating mean squared errors
categories=training_data.columns
from sklearn.metrics import mean_squared_error
for i in range(6):
    error = mean_squared_error(y_test[:,i],y_pred[:,i])
    print("mean squared error in",categories[i+2],"is",error)

# %%



