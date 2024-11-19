

import re

import string
import tensorflow as tf

from os.path import dirname, join
import pandas as pd
import numpy as np

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
number_pattern = re.compile('[-+]?([0-9]*\.[0-9]+|[0-9]+)')
punctuation_pattern=re.compile('[,@\'?\.$%?!#;&_:\"]')
signs_pattern = re.compile('[-/+*|\[\](){}]')

def main(text):
    #text='ADITYA BIRLA  etd AI MEMORIAL HOSPITA DEPARTMENT OF LABORATORY MEDICINE  Aditya Birla Health Sersices Limited                                                                   MRNNO i “ABiaiosbay VuirRe, OF-001 “me Mi Ms ANKITA S KaLsHETT ‘Age 19Yeus 9 Monihs | Days Sex FEMALE DR + VUAY Doctors PULMONARY MEDICINE 5 Speclalty AMPLE 1908260688 Se xe meson FRIDAY t6cw2019 1835 Reported Ou FRIDAY {6/082019 18:01 vet On FRIDAY 160872019 to Report Status Final HAEMATOLOGY Semple Type Whale Blod EDTA : r 7 - i} ; meter eal : Pen i vent a  [aterval H ‘HAEMOGRAM emo Pte. si rol ease ta wa. pews | act ACAD pers ae % 30460 | RBC Comat iii - ee - 4 reOULTER ramen asa esi 3848 Mey HEAL ULATEDraRAMETERD wo “ 0101.0 cu 1s (CALCULATED baRaptETERS, a PS TORO MCHC 328 dl. Ts36s AaTATEr Howser) ; aes : 0 « craery : a ; 7 _ ‘WBC Count er RHA doing, Conrenranon ae ; . _- Platelet Count ais xara, Is004t09 ; ‘uvcren rere _ a Difecential WBC Gown . . oy ‘Neutropbils 0 ae * 875.0 srw : : : Absolute Neutrophil Count S41 ssp, 2oz0 | Aestttcnacor ay . » Hartt : mane : Absolute Lyenphocyte Couet we MOH, Wao ee iexolcors ; 3 x hee wo Monestietacon Abaclte Meany Conat a 1044. ane Auris 2 . eee *atinoptl Fetes | ‘Absolute Eoxinapbtl Coust oz x1DAL tae : Act! a _ . : 020 Ba oes “US “et beste ap 7 Cy Hava, ‘02-4 Coan “Abate Basophl Coomt fetenacencaeh ne — _ . Seen irene Coat Td (ines. cevererGeneRateD RErcRt “REE tee Sgmunowu sen x ADITVA BIRLA HEALTH SERVICES LIMITED ECAP e     Le) ®️ re AUC ‘Ava teria howpadal Marg, POL Gnchwas. Pi my Aenean a4 —— ee ee eM HLI/T7SS. Par Comarparcy Cal 693-20-207 (7777 nein'
    #text='Global Multispeciality Hospital  Dr. Lokesh Zanje 49> MBBS MD a Reg. No - M-02312  MD Medicine     Patient Details : Ganesh Chavan, Male / 43 yr, B+ Chief Complaint - Vomiting, Loss of appetite  Diagnosis - Food poisoning Treatment/Advice - Rest and drink fluids  Re  - Tabrol 2mg Stat (Immediately) for 1 Days  - Tab Demisone 0.5mg BD (Twice A Day) After Meal for 5 Days , (Route/Form - Oral)  We  Date : 18-May-2021 Signature Page No. 7     Working : Mon,Tue,Wed,Thu,Fri Contact : +918855805920 Time : 10am - 02pm, 04pm - 07pm  Mail td : zanjetokesh22@gmail.com  Address : Savarkar bridge Nashik'
    text=clean_text(text)
    num_words = 20000
    
    filename = join(dirname(__file__), "dataset.csv")
    df=pd.read_csv(filename)

    df['text']=df['text'].apply(clean_text)

    train_data = df

    tokenizer = Tokenizer(num_words = num_words,oov_token="unk")
    tokenizer.fit_on_texts(train_data['text'].tolist())
    list=[text]
    x_test  = np.array( tokenizer.texts_to_sequences(list))
    #print(x_test)
    x_test = pad_sequences(x_test, padding='post', maxlen=50)
    
    # Load the TFLite model and allocate tensors.
    filename = join(dirname(__file__), "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=filename)
    interpreter.allocate_tensors()

# Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(x_test, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output = output_data.argmax(axis=1)

    if(output==0):
        return 'Prescription'
    else :
        return 'Report'




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

    return text
