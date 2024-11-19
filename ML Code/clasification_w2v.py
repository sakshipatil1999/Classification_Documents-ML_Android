#import of libraries
import logging 
import pandas as pd
import numpy as np
import re
from numpy import random

import nltk #natural language tool kit : Used for removal of stopwords. 
from nltk.corpus import stopwords

#import scikitlearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

#import tensorflow
import tensorflow as tf
from tensorflow.keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#import gensim
import gensim #library used for Word2Vec 
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from collections import Counter

#read training dataset using read_csv function from pandas library
df=pd.read_csv(r'/Users/abhaypatil/Desktop/Project/dataset.csv')

#remove rows which have null values in the class attribute
df=df[pd.notnull(df['class'])]

#RE for text processing
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
number_pattern = re.compile('[-+]?([0-9]*\.[0-9]+|[0-9]+)')
punctuation_pattern=re.compile('[,@\'?\.$%?!#;&_:\"]')
signs_pattern = re.compile('[-/+*|\[\](){}]')
STOPWORDS = set(stopwords.words('english'))


#function for cleaning of text
def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """

    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text=number_pattern.sub('', text)
    text=punctuation_pattern.sub(' ',text)
    text=signs_pattern.sub(' ',text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

#apply cleaning function on the dataset
df['text']=df['text'].apply(clean_text)

train_data = df

#Set the total no of words to keep
num_words = 20000
# vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) 
"""
Tokenizer(num_words,filters,lower, split, char_level, oov_token,document_count, **kwargs)
num_words=the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.
oov_token=if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls
   """
tokenizer = Tokenizer(num_words = num_words,oov_token="unk")

"""
fit_on_texts:
Updates internal vocabulary based on a list of texts.Required before using texts_to_sequences
input can be list of strings,list of list of strings

"""
tokenizer.fit_on_texts(train_data['text'].tolist())


"""
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
Split arrays or matrices into random train and test subsets

stratify:This stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify.
For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.

"""
X_train, X_valid, y_train, y_valid = train_test_split(train_data['text'].tolist(),\
                                                      train_data['class'].tolist(),\
                                                      test_size=0.25,\
                                                      stratify = train_data['class'].tolist(),\
                                                      random_state=0)


print('Train data len:'+str(len(X_train)))
print('Class distribution'+str(Counter(y_train)))
print('Valid data len:'+str(len(X_valid)))
print('Class distribution'+ str(Counter(y_valid)))


"""
texts_to_sequences:
Transforms each text in texts to a sequence of integers.
Only top num_words-1 most frequent words will be taken into account. Only words known by the tokenizer will be taken into account.
input:list of texts output:list of sequence
eg : "This is a cat"
      [1 2 3 4 ]

      [[1 2 3 4] [2 3 45 0]]
"""
x_train = np.array( tokenizer.texts_to_sequences(X_train) )
x_valid = np.array( tokenizer.texts_to_sequences(X_valid) )


"""
pad_sequences:
tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=None, dtype='int32', padding='pre/post',
    truncating='pre', value=0.0
)
This function transforms a list (of length num_samples) of sequences (lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps)
output:Numpy array with shape (len(sequences), maxlen)

"""
x_train = pad_sequences(x_train, padding='post', maxlen=50)
x_valid = pad_sequences(x_valid, padding='post', maxlen=50)


print(x_train[0])

#Label Encoder:Encode target labels with value between 0 and n_classes-1.
le = LabelEncoder()

#fit_transform:Fit label encoder and return encoded labels.
train_labels = le.fit_transform(y_train)
train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))
#print(train_labels)

#transform:Transform labels to normalized encoding.
valid_labels = le.transform(y_valid)
valid_labels = np.asarray( tf.keras.utils.to_categorical(valid_labels))
#to_categorical:Converts a class vector (integers) to binary class matrix.
list(le.classes_)

"""
The given tensors are sliced along their first dimension. This operation preserves the structure of the input tensors, removing the first dimension of each tensor and using it as the dataset dimension. All input tensors must have the same size in their first dimensions.
for eg:
    [[1,2],[2,3]] is converted to 1D tensor elements like [1,2] [2,3]
"""
train_ds = tf.data.Dataset.from_tensor_slices((x_train,train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid,valid_labels))

print(y_train[:10])
print('Text to number')
print(train_labels[:10])
print('Number to category')
print(train_labels[:10])


#load word2vec model for creating embedding matrix
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


from keras.layers import Embedding
embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
            
                            trainable=True)

from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
sequence_length = 50
filter_sizes = [3,4,5]
num_filters = 100
drop = 0.5


#Input() is used to instantiate a Keras tensor.
inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)

maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
flatten = Flatten()(merged_tensor)
reshape = Reshape((3*num_filters,))(flatten)
dropout = Dropout(drop)(flatten)
output = Dense(units=2, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)

# this creates a model that includes
model = Model(inputs, output)

adam=Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])
callbacks = [EarlyStopping(monitor='val_loss')]

epochs =3
history = model.fit(train_ds.shuffle(2000).batch(30),
                    validation_data=valid_ds.batch(30),
                    epochs= epochs ,
                    verbose=1)

model.save('/Users/abhaypatil/Desktop/Project/saved model/testing8') 



from keras import backend as K 

# Do some code, e.g. train and save model

K.clear_session()
