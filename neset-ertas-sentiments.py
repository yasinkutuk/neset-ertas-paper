#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:51:25 2021

@author: @yasinkutuk
"""
# -*- coding: utf-8 -*-

#####################################################################
#                       _         _            _           _        #
#    _   _   __ _  ___ (_) _ __  | | __ _   _ | |_  _   _ | | __    #
#   | | | | / _  |/ __|| || '_ \ | |/ /| | | || __|| | | || |/ /    #
#   | |_| || (_| |\__ \| || | | ||   < | |_| || |_ | |_| ||   <     #
#    \__, | \__,_||___/|_||_| |_||_|\_\ \__,_| \__| \__,_||_|\_\    #
#    |___/                                                          #
#    ____                            _  _                           #
#   / __ \   __ _  _ __ ___    __ _ (_)| |    ___  ___   _ __ ___   #
#  / / _  | / _  || '_   _ \  / _  || || |   / __|/ _ \ | '_   _ \  #
# | | (_| || (_| || | | | | || (_| || || | _| (__| (_) || | | | | | #
#  \ \__,_| \__, ||_| |_| |_| \__,_||_||_|(_)\___|\___/ |_| |_| |_| #
#   \____/  |___/                                                   #
#####################################################################
#@author: Yasin KÜTÜK          ######################################
#@web   : yasinkutuk.com       ######################################
#@email : yasinkutuk@gmail.com ######################################
#####################################################################



#Modules
import platform, matplotlib, torch, csv, statistics, zeyrek, nltk, simplemma
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import  pipeline, BertTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm as tqdm


#Path Specifier (Kodlarım Hem linux hemde Windows'da çalışsın)
if platform.system()=='Linux':
    path= r'/media/DRIVE/Dropbox/_My_Research/NesetErtas-Paper/03.Data/'
    prevpath= r'/media/DRIVE/Dropbox/_My_Research/NesetErtas-Paper/04.Meth/'
    print("Abi, Linux bu!")
else:
    path= r'//media//DRIVE//Dropbox//_My_Research//NesetErtas-Paper//03.Data//'
    prevpath= r'//media//DRIVE//Dropbox//_My_Research//NesetErtas-Paper//04.Meth//'
    print("Hocam Windows'dasın")
    


#Import of Turkuler
nesetertas = pd.read_excel(path+'NesetErtas.xlsx', sheet_name='NE')


#Settingup Device of Cuda
device = 'cuda'

#Huggingface - savasy/bert-base-turkish-sentiment-cased
model1 = AutoModelForSequenceClassification.from_pretrained('savasy/bert-base-turkish-sentiment-cased')
tokenizer1 = BertTokenizer.from_pretrained('savasy/bert-base-turkish-sentiment-cased')
sentiment1 = pipeline('sentiment-analysis', tokenizer=tokenizer1, model=model1)

#Huggingface - kuzgunlar/electra-turkish-sentiment-analysis
model2 = AutoModelForSequenceClassification.from_pretrained('kuzgunlar/electra-turkish-sentiment-analysis')
tokenizer2 = BertTokenizer.from_pretrained('kuzgunlar/electra-turkish-sentiment-analysis')
sentiment2 = pipeline("sentiment-analysis", tokenizer=tokenizer2, model=model2)

#Huggingface - yigitbekir/turkish-bert-uncased-sentiment
model3 = AutoModelForSequenceClassification.from_pretrained('yigitbekir/turkish-bert-uncased-sentiment')
tokenizer3 = BertTokenizer.from_pretrained('yigitbekir/turkish-bert-uncased-sentiment')
sentiment3 = pipeline("sentiment-analysis", tokenizer=tokenizer3, model=model3)






#Lemmatization and tokenized words
langdata = simplemma.load_data('tr')
# ' '.join(simplemma.text_lemmatizer(sozlerr[0], langdata))


# Tokenization
tokenizer = nltk.RegexpTokenizer(r"\w+")
        
        
        
turkusayisi = len(nesetertas['turku'])
   

colnames = ['turku', 'sira', 'senti1mean', 'senti1var', 'senti2mean', 'senti2var', 'senti3mean', 'senti3var', \
            'keys1','vals1', 'keyvals1', 'keys2', 'vals2', 'keyvals2', 'keys3', 'vals3', 'keyvals3', 'lemmas']
               
with open(path+ 'nesetertas-sentiment.csv', "a") as csv_file:
    writer = csv.writer(csv_file,delimiter =";", dialect='excel')
    writer.writerow(colnames)
    for t in range(turkusayisi):
        text = nesetertas['soz'][t]
        sozler = text.split('\n')
        sozler = [x for x in sozler if x]
        tokened = [' '.join(tokenizer.tokenize(x)) for x in sozler if ' '.join(tokenizer.tokenize(x))]
        tokened = [' '.join(simplemma.text_lemmatizer(x, langdata)) for x in tokened if ' '.join(simplemma.text_lemmatizer(x, langdata)) ]
        tokened = ' '.join(tokened)
        senti1 = sentiment1(sozler)
        senti2 = sentiment2(sozler)
        senti3 = sentiment3(sozler)
        
        keys1 = []
        values1 = []
        keyvalues1 = []
        for s in range(len(senti1)):
             keys1.append(senti1[s]['label'])
             values1.append(senti1[s]['score']) 
             if(senti1[s]['label'] == 'negative'):
                 keyvalues1.append(senti1[s]['score'] * (-1))
             else:
                keyvalues1.append(senti1[s]['score'])

        keys2 = []
        values2 = []
        keyvalues2 = []
        for s in range(len(senti2)):
             keys2.append(senti2[s]['label'])
             values2.append(senti2[s]['score']) 
             if(senti2[s]['label'] == 'Negative'):
                 keyvalues2.append(senti2[s]['score'] * (-1))
             else:
                keyvalues2.append(senti2[s]['score'])

        keys3 = []
        values3 = []
        keyvalues3 = []
        for s in range(len(senti3)):
             keys3.append(senti3[s]['label'])
             values3.append(senti3[s]['score']) 
             if(senti3[s]['label'] == 'LABEL_1'):
                 keyvalues3.append(senti3[s]['score'] * (-1))
             else:
                keyvalues3.append(senti3[s]['score'])

        
        writer.writerow([nesetertas['turkuadi'][t], nesetertas['sira'][t],  \
                         statistics.mean(keyvalues1), statistics.variance(keyvalues1), \
                         statistics.mean(keyvalues2), statistics.variance(keyvalues2), \
                         statistics.mean(keyvalues3), statistics.variance(keyvalues3), \
                         '|'.join(keys1),'|'.join(str(v) for v in values1), '|'.join(str(kv) for kv in keyvalues1),\
                         '|'.join(keys2),'|'.join(str(v) for v in values2), '|'.join(str(kv) for kv in keyvalues2),\
                         '|'.join(keys3),'|'.join(str(v) for v in values3), '|'.join(str(kv) for kv in keyvalues3),\
                         tokened])
            





# text = nesetertas['soz'][1]
# sozler = text.split('\n')
# sozler = [x for x in sozler if x]








# model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
# tokenizer = BertTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
# sa = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)

# p = sa("bu telefon modelleri çok kaliteli , her parçası çok özel bence")
# print(p)
# # [{'label': 'LABEL_1', 'score': 0.9871089}]
# print(p[0]['label'] == 'LABEL_1')
# # True

# p = sa("Film çok kötü ve çok sahteydi")
# print(p)
# # [{'label': 'LABEL_0', 'score': 0.9975505}]
# print(p[0]['label'] == 'LABEL_1')
# # False



# ## Çöp kısmı
# # Lemmatization çalışıyor
# import zeyrek
# analyzer = zeyrek.MorphAnalyzer()
# k = analyzer.lemmatize('zalimlik bu yaptığın')

# k = analyzer.lemmatize(sozler[0])



# x = tokenizer.encode(sozler[0])
# with torch.no_grad():
#     x, _ = bert(torch.stack([torch.tensor(x)]).to(device))
    
    
#     k = list(x[0][0].cpu().numpy())




# def feature_extraction(text):
#     x = tokenizer.encode(filter(text))
#     with torch.no_grad():
#         x, _ = bert(torch.stack([torch.tensor(x)]).to(device))
#         return list(x[0][0].cpu().numpy())


# from polyglot.downloader import downloader
# print(downloader.supported_languages_table("sentiment2", 3))
# from polyglot.text import Text
# text = Text(sozler[0])
# text = Text('Bu çok iyi değil.')
# print("{:<16}{}".format("Word", "Polarity")+"\n"+"-"*30)
# for w in text.words:
#     print("{:<16}{:>2}".format(w, w.polarity))





# from textblob import TextBlob
# blob_obj = TextBlob(sozler[0])
# blob_obj.tokens
# from textblob.taggers import NLTKTagger
# nltk_tagger = NLTKTagger()
# blob_obj = TextBlob(k[0][1][0], pos_tagger=nltk_tagger)
# blob_obj.pos_tags
# blob_obj.sentiment

