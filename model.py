# %%
import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Flatten,Reshape
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
import csv
import nltk
nltk.download('stopwords')
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
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau, CSVLogger
from time import asctime, time
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
from datetime import datetime
import os
import math
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.initializers import glorot_uniform,glorot_normal
from tensorflow.keras.layers import MaxPooling1D

# %%
# specify the directory path
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_tag = f"two---({current_time})"
log_directory = Path(f'./log/{experiment_tag}')
# data_directory = Path(f'./data/{current_time}')
# create the directory if it does not already exist
log_directory.mkdir(parents=True, exist_ok=True)
# Logging matters
logger = logging.getLogger()
# create a logger object
logger = logging.getLogger()
# set the logging level
logger.setLevel(logging.DEBUG)
# create a file handler to write the log messages to a file
log_filepath = f'{log_directory}/model-runtime.log'
file_handler = logging.FileHandler(log_filepath)
# create a stream handler to output the log messages to the console
stream_handler = logging.StreamHandler()
# set the formatter for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# %%
#loading the training data
training_data_filepath = "./data/train.csv"
logger.info(f"Loading training data from {training_data_filepath}")
training_data=pd.read_csv(training_data_filepath)


# %% [markdown]
# # Preprocessing. 
# 
# Preprocessing includes removing spaces, special characters, contractions and stop words. 

# %%
# ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
logger.info(f"DATA PROCESSING")
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

logger.info(f"Preprocessing now...")
training_data['full_text']=preprocess(training_data['full_text'],stopword=False)

# %%
training_data.head() 

# %%
#Train -test split, 20% of the data for validation set.
logger.info(f"Splitting training and test")
y=training_data[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']]
X=training_data.drop(['text_id','cohesion','syntax','vocabulary','phraseology','grammar','conventions'],axis=1)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20)
logger.info(f"Split training and test: {X_train.shape}, {y_train.shape}")



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

logger.info(f"Padding text")
comm_len=200
train_com_pad,test_com_pad,token_com= text_padding(X_train['full_text'],X_test['full_text'],comm_len)
logger.info(f"Padded text: {train_com_pad.shape}, {test_com_pad.shape}")

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
logger.info(f"Generating embedding matrix")
embedded_comm_filepath = './data/Embedded_matrix_trained_data.csv'
# if not os.path.exists(embedded_comm_filepath):
embedding_comm = generate_embedding_matrix(token_com)
logger.info(f"Generated embedding matrix: {embedding_comm.shape}\n")
#Storing the embedded matrix for trained
np.savetxt(embedded_comm_filepath, embedding_comm, delimiter=',')
#Use the already stored embedded matrix for the trained data
embedding_comm = np.loadtxt(embedded_comm_filepath, delimiter=',')


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
# Configure hyperparameters and training callbacks here
lr = 0.0001
epochs = 1
adam = tf.keras.optimizers.Adam(learning_rate=lr)
logger.info(f"Initialising Model with lr = {lr}")

#reduce_lr reduces the learning rate when the metric has stoppes improving for 2 epochs.
reduce_lr_factor = 0.25
reduce_lr_patience = 3
reduce_lr = ReduceLROnPlateau(monitor = 'val_mcrmse', factor = reduce_lr_factor, patience = reduce_lr_patience, verbose = 1)

#Using EarlyStopping to stop the calculation upon reaching enough accuracy
earlystop_min_delta = 0.0001
earlystop_patience = 5
earlystop = EarlyStopping(monitor = 'val_mcrmse',  mode="min", min_delta = earlystop_min_delta, patience = earlystop_patience, verbose = 1)

csv_logger = CSVLogger(f'{log_directory}/training.log')

callbacks = [reduce_lr, earlystop, csv_logger]

logger.info(f"ReduceLROnPlateau parameters: factor = {reduce_lr_factor}, patience = {reduce_lr_patience}")
logger.info(f"EarlyStopping parameters: min_delta = {earlystop_min_delta}, patience = {earlystop_patience}")

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
# Loading the model
model=LSTM_CNN1D(comm_len,token_com)

# TODO: Enable to load weights
# model.load_weights("./data/LSTM_weights_final.h5")

# %%
logger.info(f"Training Model with epoch = {str(epochs)}")
hitory1=model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)
logger.info(f"Training complete!\n")

# %%
#saving the model and weights for future 
from tensorflow.keras.models import Sequential, model_from_json
model_json = model.to_json()
model_save_json_filepath = f"{log_directory}/LSTM_model_final.json"
weights_save_h5_filepath = f"{log_directory}/LSTM_weights_final.h5"
logger.info(f"Saving model to {model_save_json_filepath} and weights to {weights_save_h5_filepath}")
with open(model_save_json_filepath,"w") as json_file:
    json_file.write(model_json)
model.save_weights(weights_save_h5_filepath)

# %%
# Loading the saved model
json_file=open(model_save_json_filepath,'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights_save_h5_filepath)

# %%
#predicted y
logger.info(f"Testing model")
y_pred = model.predict(X_test)

# %% [markdown]
# # Errors

# %%
# calculating mean squared errors
categories=training_data.columns
from sklearn.metrics import mean_squared_error
mse_error_filepath = f"{log_directory}/mse.log"
with open(mse_error_filepath,"w") as mse_log_file:
    error_str = ""
    for i in range(6):
        error = mean_squared_error(y_test[:,i],y_pred[:,i])
        category_error_str = "mean squared error in " + str(categories[i+2]) + " is " + str(error) + "\n"
        error_str += category_error_str
    mse_log_file.write(error_str)
logger.info(f"Saved MSE for categories in {mse_error_filepath}")
logger.info(f"Testing complete")



