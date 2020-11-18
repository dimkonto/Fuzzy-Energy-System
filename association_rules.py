# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:43:14 2020

@author: dimkonto
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib import pyplot as pp

"""
#LOAD 1 HOT ENCODED DATASET AND SPECIFY COLUMNS
#ASSOCIATION EXPERIMENT CODE
path = r'D:\Datasets\hpc\fuzzy1hot_v2.csv'
dataset = pd.read_csv(path,sep=',',header=0,low_memory=False,infer_datetime_format=True,index_col=0)
print(dataset.shape)

dataset.drop(dataset.columns[0:30],axis=1,inplace=True)
print(dataset.shape)
print(dataset.head())
dataset.to_csv(r'D:\Datasets\hpc\fuzzy_features.csv')
#df=pd.DataFrame(dataset.values)
#print(df.shape)

"""

"""
freq_items = apriori(dataset, min_support=0.00001, use_colnames=True, max_len=3, verbose=1, low_memory=True)
#print(freq_items)    

rules = association_rules(freq_items, metric="confidence", min_threshold=0.00001)
print(rules.head())
#rules[(rules['consequents']=={'APPLIANCE_Low'}) & (rules['confidence']>0.992)].to_csv(r'D:\Datasets\hpc\association_applow.csv')
rules[rules['consequents']=={'APPLIANCE_Medium'}].to_csv(r'D:\Datasets\hpc\association_appmedium.csv')
rules[rules['consequents']=={'APPLIANCE_High'}].to_csv(r'D:\Datasets\hpc\association_apphigh.csv')
"""

#KEEPING LABELS NEEDED FOR ONE HOT ENCODING
path = r'D:\Datasets\hpc\fuzzy1hot_v3.csv'
dataset = pd.read_csv(path,sep=',',header=0,low_memory=False,infer_datetime_format=True)
keep_list=['T1_label','RH1_label','T2_label','RH2_label','T3_label','RH3_label','T4_label','RH4_label','T5_label','RH5_label','T6_label','RH6_label','T7_label','RH7_label','T8_label','RH8_label','T9_label','RH9_label','PRESSURE_label','WIND_label','VISIBILITY_label','APPLIANCE_label']
dataset[keep_list].to_csv(r'D:\Datasets\hpc\fuzzy_labels.csv')

#print(dataset.shape)