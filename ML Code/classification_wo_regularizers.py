#dependencies
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from numpy import array
import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import LabelBinarizer
import nltk
from sklearn.preprocessing import LabelEncoder

def create_train_test_sets(data,split):
    np.random.seed(0)
    mask = np.random.rand(len(data)) < split
    train_data = data[mask]
    test_data = data[~mask]
    return train_data,test_data

def clean_and_get_tokens(doc):
    tokens = doc.split()
    table = str.maketrans('','',punctuation)  #removes punctuations using 
    tokens = [w.translate(table) for w in tokens] #dictionary of punctuations
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word)>2]
    return tokens

data=pd.read_csv(r'/Users/abhaypatil/Desktop/Project/dataset.csv')

print(data)
words = set()
vocab = {}


token = data['text'][0].split()
table = str.maketrans('','',punctuation)
tokens = [w.translate(table) for w in token] 
#print(tokens)
tokens = [word for word in tokens if word.isalpha()]
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
tokens = [word for word in tokens if len(word)>2]
#print(tokens)

documents = data['text']
for doc in documents:
    tokens = clean_and_get_tokens(doc)
    for token in tokens:
        if token in vocab:
            vocab[token] += 1
        else:
            vocab[token] = 1

for word in vocab:
    if vocab[word] > 5:
        words.add(word)

print(data)

train_data,test_data = create_train_test_sets(data,0.8)

train_documents = []
for doc in train_data['text']:
    tokens = doc.split()
    final_tokens = []
    #final_string = ''
    for token in tokens:
        if token in words:
            final_tokens.append(token)
    final_string = ' '.join(final_tokens)
    train_documents.append(final_string)

test_documents = []
for doc in test_data['text']:
    tokens = doc.split()
    final_tokens = []
    #final_string = ''
    for token in tokens:
        if token in words:
            final_tokens.append(token)
    final_string = ' '.join(final_tokens)
    test_documents.append(final_string)

    tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_documents)
encoded_docs = tokenizer.texts_to_sequences(train_documents)

max_length = max(([len(s.split()) for s in train_documents]))
le = LabelEncoder()
train_labels = le.fit_transform(train_data['class'])
ytrain=tf.keras.utils.to_categorical(train_labels, num_classes=2)
Xtrain = pad_sequences(encoded_docs, maxlen=50, padding='post')


encoded_docs = tokenizer.texts_to_sequences(test_documents)
test_labels =le.fit_transform(test_data['class'])
Xtest = pad_sequences(encoded_docs, maxlen=50, padding='post')
ytest = keras.utils.to_categorical(test_labels, num_classes=2)

vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length = 50))
model.add(Conv1D(filters=16, kernel_size=16, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=7, verbose=2, validation_data = (Xtest,ytest))

model.save('/Users/abhaypatil/Desktop/Project/saved model/testing5') 


from keras import backend as K 

# Do some code, e.g. train and save model

K.clear_session()
