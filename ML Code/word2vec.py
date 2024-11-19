import re
import pandas as pd
from time import time 
from collections import defaultdict
import spacy
import logging  #setting up the logging to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",datefmt= '%H:%M:%S',level=logging.INFO)

#Loading dataset
df=pd.read_csv(r'/Users/abhaypatil/Desktop/Project/dataset.csv')
#Cleaning:  lemmatizing and removing the stopwords and non-alphabetic characters

#disabling named entity recognition for speed
nlp=spacy.load('en_core_web_sm', disable=['ner' , 'parser'])
def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt=[token.lemma_ for token in doc if not token.is_stop]
     # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt)>2:
        return ' '.join(txt)

#Removes non-alphabetic characters
brief_cleaning=(re.sub("[^A-Za-z']+",' ',str(row)).lower() for row in df['text'])

#spaCy.pipe() to speed up the cleaning process
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]

#Put the results in a DataFrame to remove missing values and duplicates
df_clean= pd.DataFrame({'clean' : txt})
print(df_clean)

#to cut down memory consumption of phrases
from gensim.models.phrases import Phrases,Phraser

sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count =30, progress_per =1000)

bigram = Phraser(phrases)
sentences = bigram[sent]

word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)

sorted(word_freq , key = word_freq.get , reverse=True)[:10]

#Training the model

from gensim.models import Word2Vec

w2v_model = Word2Vec() #object creation


w2v_model.build_vocab(sentences, ) #building our vocab

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)#training w2v model

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.init_sims(replace=True) 

w2v_model.save('custom_model.bin')