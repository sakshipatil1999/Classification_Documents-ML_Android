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

from sklearn.metrics import classification_report

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy import stats


#text='reg name mrs vidya patil female years qo alaspan tab rar ust wat aisi gh er nutex tab mg ws raw test ot cat aaa aay keto b lotion raw ea ia ch se utd araquaret arte scheduled vpdicines sunspied days months wie ise ba sst hts areal erferardt araoralt afar zfarst yet saat stat wd acie ad seg ar shotat'
#text='Global Multispeciality Hospital  Dr. Lokesh Zanje 49> MBBS MD a Reg. No - M-02312  MD Medicine     Patient Details : Ganesh Chavan, Male / 43 yr, B+ Chief Complaint - Vomiting, Loss of appetite  Diagnosis - Food poisoning Treatment/Advice - Rest and drink fluids  Re  - Tabrol 2mg Stat (Immediately) for 1 Days  - Tab Demisone 0.5mg BD (Twice A Day) After Meal for 5 Days , (Route/Form - Oral)  We  Date : 18-May-2021 Signature Page No. 7     Working : Mon,Tue,Wed,Thu,Fri Contact : +918855805920 Time : 10am - 02pm, 04pm - 07pm  Mail td : zanjetokesh22@gmail.com  Address : Savarkar bridge Nashik'
#text='Deccan Multispeciality Hospital  Dr. Tushar Ghorpade V4 M.B.B.S., MD Reg. No - M-02587  MD medicine  Patient Details : Yogita karape, Female / 36 yr, B+ Chief Complaint - Leg Cramps  Re  - Tab Stugeron Forte BD (Twice A Day) After Meal for 5 Days , (Route/Form - Oral)  oe  Date : 08-May-2021 Signature Page No. 1  Working : Mon, Tue,Wed, Thu, Fri Contact : +9173787831 68  Time : 10am - 02pm, 04pm - 07pm  Mail Id : omkaranilmane1998@gmail.com  Address : Deccan chawk Pune'
#text='| wd Feats ~ Tora ara, i. bis Geter : aft 4, 3, arfarar aT, Wee eA shee oral, are, ORAL t758 fiewenia, y-33. ahr. Rea acauy BW: wrcaets ©️ 88: 91.008 930 emgrpootioce oe wtante Narfi@ § S Kalshetty fates 20-02-2021 11:10 am Age/Sex: 43y /M Mobile: 8600004552!  Office ID; HH2954  Symptoms: Anxiety, General weakness Notes: Ont9.2.21 15/105,Sp02 96,p 99 Vitals: Weight: 70.7 kg, SPO2: 98 %, BP: 141/95 mmHg        1. Tablet Revotrit 5: Ss O-1/2-1/2X 5 fee 2. Tablet Benmix. Gold: DA 0-0-1X20 fae | wat saat 3. Tablet Bonark- D3: cope) 0-1-0X 20 faa gat sacra  Follow up: 06 Mar 2021, Saturday        Dr. Hemant Huligotkar AIG11758  ——      ——_ PLEASE D0 Nor pe aga MEDICIN'
#text='aT. 3, URI Siar,  A                  hie art, ararerr, ees. At-1- 1758 Feerenia, ye-93. HM. 822 se ayy BE: tonrarey, @ aH : 99.00 8 9.30 ©️ Ui goo a 90.00 ©️ <ffar az NaMé: S S Kalshetty fartierte: 25-02-2021 Age/Sex: 43y /M Mobile: 8600004!  Office ID: HH2954  Symptoms: Hyperlipedemia Findings: Low vit D (Mild)  Notes: Cholesterol 253 Low D3  Vitals: Weight: 70.7 kg, Pulse: 72 /min, SPO2: 98 %, BP: 131/90 mmHg       aire:  1. Tablet TG Tor - F: Bos! 0-0-+X1ufeT: : wat sao  2. Tablet Benmix. Gold: Bo 1-0-0X1RfeaT” , rea eartav  3. Sachet Calcirol D3 sachets:  amcasnigeat 2 Dr. Hemant Hullgolkar Oh A1-1-11788'
#text='Org fete ~ erordeen sera, Bi Beier perttooar at 4. 3, ream ferremiy, O2LEH LEE. wiht arch, Serrere, eit, At -1- 11758 facenia, g-39. HT. 8e22¢3cayy We: reqeatts NaI®️ S S Kaishetty fete: 11-03-2021 11:08 Age/Sex: 44y /M Mobile: 8600004552 Office ID: HH2954         Symptoms: Leg cramps. Muscle pain Vitals: Weight: 71.5 kg, Pulse: 73 /min, SPO2: 98 %, BP: 125/87 mmtg  atta: Tablet Evion LC: Bo 0-1-1X15 fare  Dr. Hemant Hullgoikar Ube  Al-411758 Bet Street       ‘fia docon.co.in'
#text='Name Ma Ms on ANKITA S KALSHUTTI ‘Age (9 Years 9 Months 1 Days. Sex FEMALE. I  Bester ANAND. VUAY Doctors PULMONARY MEDICINE i SampteNo 1508160658 Specialty !  ‘Ward & Bed No Reported On FRIDAY. {6/08/2919 18:01 Report Status Final  Collected On FRIDAY I6/ow2019 15:35 Revelved On FRIDAY 160820189 16:10                       HAE! ~ Semole Type Whole Blood BOTA MATOLOGY : H Poem “TO :  meter I i Rent | Vat faeces HE : : Interval : HAEMOGRAM Hemeptobia - Ge (PROTOMSDUE HEARARS M42 wat 120.50 Bcr - ‘eaLcan ate anasto a8 Fa soaso RBC CK a i — - -. court mess asa 1a. 348 Have wo (CALCULATED FaRAsARTEM a a30010 McH we - oe {EaLcuuatenpatawerens B mo39 CH ease envoy maa wa. sisses we 1 ee bse * Nota  BESS i. os won ~ aide’ Plaset Count _ as xiv3ut——“isoaing™️ —— _Diereatial WBC-Comst a a ‘Neutropiaits oe * worse 7 ‘resaucroscor) } . ‘Adssalute Neutrophil Comat st sis, 2o70 . fiesamnssascorn | Br # mphoe r Fearn)  mong ‘Absolute Lymphoryte Count 1 ia lado  Reza taexstear i 103 * 1 evened Aorao fies te Monocyte Couat a8 Aut v2.19 AEIMEaon! " * postnophlls a * 1060  Feteieatcorn : ‘Alwolute Ezsinophil Count a2 x10) ome . : fteamcnoncory po “Bavopbile 4 ix noe ena acne) en i : f Abactute Basophtl Count oa “at DO Low bore ” - oe eee eee  Sinsis coveremieversteo Reeth “PWSB OTe ete Maen email,           sspacou 2-10 fe ADITYA BIRLA HEALTH GERVICES LIMITED a CAP adv Bra evel Bary, BEY Chctined, Pave 42K AE CREDITEO. - arom oh GAARN Por urgency GAR 991-2 MOYUFFTECaeweseo mare'
text='ADITYA BIRLA  etd AI MEMORIAL HOSPITA DEPARTMENT OF LABORATORY MEDICINE  Aditya Birla Health Sersices Limited                                                                   MRNNO i “ABiaiosbay VuirRe, OF-001 “me Mi Ms ANKITA S KaLsHETT ‘Age 19Yeus 9 Monihs | Days Sex FEMALE DR + VUAY Doctors PULMONARY MEDICINE 5 Speclalty AMPLE 1908260688 Se xe meson FRIDAY t6cw2019 1835 Reported Ou FRIDAY {6/082019 18:01 vet On FRIDAY 160872019 to Report Status Final HAEMATOLOGY Semple Type Whale Blod EDTA : r 7 - i} ; meter eal : Pen i vent a  [aterval H ‘HAEMOGRAM emo Pte. si rol ease ta wa. pews | act ACAD pers ae % 30460 | RBC Comat iii - ee - 4 reOULTER ramen asa esi 3848 Mey HEAL ULATEDraRAMETERD wo “ 0101.0 cu 1s (CALCULATED baRaptETERS, a PS TORO MCHC 328 dl. Ts36s AaTATEr Howser) ; aes : 0 « craery : a ; 7 _ ‘WBC Count er RHA doing, Conrenranon ae ; . _- Platelet Count ais xara, Is004t09 ; ‘uvcren rere _ a Difecential WBC Gown . . oy ‘Neutropbils 0 ae * 875.0 srw : : : Absolute Neutrophil Count S41 ssp, 2oz0 | Aestttcnacor ay . » Hartt : mane : Absolute Lyenphocyte Couet we MOH, Wao ee iexolcors ; 3 x hee wo Monestietacon Abaclte Meany Conat a 1044. ane Auris 2 . eee *atinoptl Fetes | ‘Absolute Eoxinapbtl Coust oz x1DAL tae : Act! a _ . : 020 Ba oes “US “et beste ap 7 Cy Hava, ‘02-4 Coan “Abate Basophl Coomt fetenacencaeh ne — _ . Seen irene Coat Td (ines. cevererGeneRateD RErcRt “REE tee Sgmunowu sen x ADITVA BIRLA HEALTH SERVICES LIMITED ECAP e     Le) ®️ re AUC ‘Ava teria howpadal Marg, POL Gnchwas. Pi my Aenean a4 —— ee ee eM HLI/T7SS. Par Comarparcy Cal 693-20-207 (7777 nein'
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

text=clean_text(text)
print(text)
num_words = 20000
df=pd.read_csv(r'/Users/abhaypatil/Desktop/Project/dataset.csv')

df['text']=df['text'].apply(clean_text)

train_data = df

tokenizer = Tokenizer(num_words = num_words,oov_token="unk")
tokenizer.fit_on_texts(train_data['text'].tolist())
list=[text]
x_test  = np.array( tokenizer.texts_to_sequences(list))
#print(x_test)
x_test = pad_sequences(x_test, padding='post', maxlen=50)

new_model = tf.keras.models.load_model('/Users/abhaypatil/Desktop/Project/saved model/testing')

predictions = new_model.predict(x_test)

predict_results=predictions.argmax(axis=1)



print(predict_results)
result=np.bincount(predict_results).argmax()
print(result)
if(result == 0):
    print('PRESCRIPTION')
else:
    print('REPORT')


"""y_score1 = new_model.predict_proba(x_test)[:,1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(x_test, y_score1)
print('roc_auc_score for CNN: ', roc_auc_score(x_test, y_score1))
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - CNN')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
*/"""
from keras import backend as K 

# Do some code, e.g. train and save model

K.clear_session()