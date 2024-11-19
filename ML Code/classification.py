import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import regularizers

from tensorflow.keras import layers
from tensorflow.keras import losses

from collections import Counter

import pandas as pd
import numpy as np

import gensim #library used for Word2Vec 
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pydot


def remove_number(text):
    number_pattern = re.compile('[-+]?([0-9]*\.[0-9]+|[0-9]+)')
    #Match all digits in the string and replace them by empty string
    return number_pattern.sub(r'',text)

def remove_email(text):
    email_pattern = re.compile('[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+')
    #Match email in the string and replace them by empty string
    return email_pattern.sub(r'',text)

def remove_punctuation(text):
    punctuation_pattern=re.compile('[,@\'?\.$%?!#;&_:\"]')
    return punctuation_pattern.sub(r'',text)

def remove_signs(text):
    signs_pattern = re.compile('[-/+*|\[\](){}]')
    return signs_pattern.sub(r'',text)



train_data=pd.read_excel(r'/Users/abhaypatil/Downloads/project_train_dataset.xlsx')

train_data.dropna(axis = 0, how = 'any' ,inplace=True)
train_data['Num_words_text'] = train_data['text'].apply(lambda x:len(str(x).split()))
mask = train_data['Num_words_text'] >2
train_data=train_data[mask]
max_train_sentence_length = train_data['Num_words_text'].max()

train_data['text'] = train_data['text'].apply(remove_number)
train_data['text'] = train_data['text'].apply(remove_email)
train_data['text'] = train_data['text'].apply(remove_punctuation)
train_data['text'] = train_data['text'].apply(remove_signs)

print(train_data)

test_data=pd.read_excel(r'/Users/abhaypatil/Downloads/Project_test_dataset.xlsx')
test_data.dropna(axis = 0, how = 'any' ,inplace=True)
test_data['Num_words_text'] = test_data['text'].apply(lambda x:len(str(x).split()))
mask = test_data['Num_words_text'] >2
test_data=test_data[mask]
max_test_sentence_length = test_data['Num_words_text'].max()

test_data['text'] = test_data['text'].apply(remove_number)
test_data['text'] = test_data['text'].apply(remove_email)
test_data['text'] = test_data['text'].apply(remove_punctuation)
test_data['text'] = test_data['text'].apply(remove_signs)

print(test_data)

num_words = 20000


tokenizer = Tokenizer(num_words = num_words,oov_token="unk")
tokenizer.fit_on_texts(train_data['text'].tolist())

X_train, X_valid, y_train, y_valid = train_test_split(train_data['text'].tolist(),\
                                                      train_data['class'].tolist(),\
                                                      test_size=0.25,\
                                                      stratify = train_data['class'].tolist(),\
                                                      random_state=0)


print('Train data len:'+str(len(X_train)))
print('Class distribution'+str(Counter(y_train)))
print('Valid data len:'+str(len(X_valid)))
print('Class distribution'+ str(Counter(y_valid)))


x_train = np.array( tokenizer.texts_to_sequences(X_train) )
x_valid = np.array( tokenizer.texts_to_sequences(X_valid) )
x_test  = np.array( tokenizer.texts_to_sequences(test_data['text'].tolist()) )



x_train = pad_sequences(x_train, padding='post', maxlen=50)
x_valid = pad_sequences(x_valid, padding='post', maxlen=50)
x_test = pad_sequences(x_test, padding='post', maxlen=50)

print(x_train[0])

le = LabelEncoder()

train_labels = le.fit_transform(y_train)
train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))
#print(train_labels)
valid_labels = le.transform(y_valid)
valid_labels = np.asarray( tf.keras.utils.to_categorical(valid_labels))


list(le.classes_)


train_ds = tf.data.Dataset.from_tensor_slices((x_train,train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid,valid_labels))

print(y_train[:10])
train_labels = le.fit_transform(y_train)
print('Text to number')
print(train_labels[:10])
train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))
print('Number to category')
print(train_labels[:10])

max_features =20000
embedding_dim =100
sequence_length = 50

model = Word2Vec.load('custom_model.bin')
#extract KeyedVectors from the model
word_vectors=model.wv
#get indexes of the words in dictionary
word_index = tokenizer.word_index
EMBEDDING_DIM=100
vocabulary_size=min(len(word_index)+1,num_words) #set the vocabulary length according to the word_index
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM)) #set the embedding matrix of the given size and set all values to 0
for word, i in word_index.items():
    if i>=num_words:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

del(word_vectors)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding( vocabulary_size,EMBEDDING_DIM, \
                                    weights=[embedding_matrix],embeddings_regularizer = regularizers.l2(0.01)))                                    

model.add(tf.keras.layers.Conv1D(128,3, activation='relu',\
                                 kernel_regularizer = regularizers.l2(0.01),\
                                 bias_regularizer = regularizers.l2(0.01)))                               


model.add(tf.keras.layers.GlobalMaxPooling1D())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(2, activation='sigmoid',\
                                kernel_regularizer=regularizers.l2(0.01),\
                                bias_regularizer=regularizers.l2(0.01),))
                               


model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='Nadam', metrics=["CategoricalAccuracy"])



epochs = 10
# Fit the model using the train and test datasets.
#history = model.fit(x_train, train_labels,validation_data= (x_test,test_labels),epochs=epochs )
history = model.fit(train_ds.shuffle(2000).batch(30),
                    validation_data=valid_ds.batch(30),
                    epochs= epochs ,
                    verbose=1)

model.save('/Users/abhaypatil/Desktop/Project/saved model/testing7') 
json_string = tokenizer.to_json()

import json
with open('/Users/abhaypatil/Desktop/Project/saved model/tokenizer.json', 'w') as outfile:
    json.dump(json_string, outfile)

from keras import backend as K 

# Do some code, e.g. train and save model

K.clear_session()