import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import json
from nltk.corpus import stopwords
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import stats


df=pd.read_csv(r'/Users/abhaypatil/Desktop/Project/test dataset.csv')
df=df[pd.notnull(df['class'])]
df = df.sample(frac=1).reset_index(drop=True)
print(df.head(10))
print(df['text'].apply(lambda x: len(x.split(' '))).sum())

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
number_pattern = re.compile('[-+]?([0-9]*\.[0-9]+|[0-9]+)')
punctuation_pattern=re.compile('[,@\'?\.$%?!#;&_:\"]')
signs_pattern = re.compile('[-/+*|\[\](){}]')
STOPWORDS = set(stopwords.words('english'))

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

df['text']=df['text'].apply(clean_text)
test_data = df

num_words = 20000
df=pd.read_csv(r'/Users/abhaypatil/Desktop/Project/dataset.csv')

df['text']=df['text'].apply(clean_text)

train_data = df

tokenizer = Tokenizer(num_words = num_words,oov_token="unk")
tokenizer.fit_on_texts(train_data['text'].tolist())

x_test  = np.array( tokenizer.texts_to_sequences(test_data['text'].tolist()) )
x_test = pad_sequences(x_test, padding='post', maxlen=50)

new_model = tf.keras.models.load_model('/Users/abhaypatil/Desktop/Project/saved model/testing8')
new_model.summary()
predictions = new_model.predict(x_test)

predict_results=predictions.argmax(axis=1)

test_data['prediction'] = predict_results
test_data['prediction']= np.where((test_data.prediction == '0'),'prescription',test_data.prediction)
test_data['prediction']= np.where((test_data.prediction == '1'),'report','prescription')

test_data.to_excel("result.xlsx")

labels = ['prescription','report']

print(confusion_matrix(test_data['class'].tolist(),test_data['prediction'].tolist(),labels=labels))
print(classification_report(test_data['class'].tolist(),test_data['prediction'].tolist(),labels=labels))

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

y_test=test_data['class']
y_test= np.where((y_test== 'prescription'),0,y_test)
y_test= np.where((y_test == 'report'),1,0)

y_pred_keras =predictions.argmax(axis=1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

"""
  When your ground truth output is 0,1 and your prediction is 0,1, you get an angle-shape elbow.
   If your prediction or ground truth are confidence values or probabilities (say in the range [0,1]), 
   then you will get curved elbow.
"""



from keras import backend as K 

# Do some code, e.g. train and save model

K.clear_session()