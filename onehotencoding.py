# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:38:11 2020

@author: dimkonto
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib import pyplot as pp
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_text

from sklearn.tree import export_graphviz
from io import StringIO  
from IPython.display import Image
import pydot  
import pydotplus
import docx


path = r'D:\Datasets\hpc\fuzzy_labels.csv'
dataset = pd.read_csv(path,sep=',',header=0,low_memory=False,infer_datetime_format=True)

# LOOP FOR TEMPERATURE NEW LABELED COLUMNS
for k in range(9):
    number=k+1
    dataset['temp_'+str(number)+'-verycold']=""
    dataset['temp_'+str(number)+'-cold']=""
    dataset['temp_'+str(number)+'-cool']=""
    dataset['temp_'+str(number)+'-warm']=""
    dataset['temp_'+str(number)+'-hot']=""

# LOOP FOR HUMIDITY NEW LABELED COLUMNS
for k in range(9):
    number=k+1
    dataset['humidity_'+str(number)+'-dry']=""
    dataset['humidity_'+str(number)+'-comfortable']=""
    dataset['humidity_'+str(number)+'-humid']=""
#LABELED WINDSPEED,VISIBILITY & PRESSURE
dataset['windspeed-Low']=""
dataset['windspeed-Medium']=""
dataset['windspeed-High']=""
    

dataset['visibility-Low']=""
dataset['visibility-Medium']=""
dataset['visibility-High']=""
    

dataset['pressure-Low']=""
dataset['pressure-Medium']=""
dataset['pressure-High']=""

#LABELS FOR TEMPERATURE 
templabels=['verycold','cold','cool','warm','hot']
#LABELS FOR HUMIDITY
humlabels=['dry','comfortable','humid']
#LABELS FOR WINDSPEED
windlabels=['Low','Medium','High']
#LABELS FOR VISIBILITY
vislabels=['Low','Medium','High']
#LABELS FOR PRESSURE
preslabels=['Low','Medium','High']

for j in range(dataset['T1_label'].shape[0]):
    for m in range(9):
        for k in range(5):
            if dataset['T'+str(m+1)+'_label'].values[j]==templabels[k]:
                dataset['temp_'+str(m+1)+'-'+templabels[k]].values[j]=1
            else:
                dataset['temp_'+str(m+1)+'-'+templabels[k]].values[j]=0
                
        for b in range(3):
            if dataset['RH'+str(m+1)+'_label'].values[j]==humlabels[b]:
                dataset['humidity_'+str(m+1)+'-'+humlabels[b]].values[j]=1
            else:
                dataset['humidity_'+str(m+1)+'-'+humlabels[b]].values[j]=0
    for c in range(3):
            if dataset['WIND'+'_label'].values[j]==windlabels[c]:
                dataset['windspeed'+'-'+windlabels[c]].values[j]=1
            else:
                dataset['windspeed'+'-'+windlabels[c]].values[j]=0
            
            if dataset['VISIBILITY'+'_label'].values[j]==vislabels[c]:
                dataset['visibility'+'-'+vislabels[c]].values[j]=1
            else:
                dataset['visibility'+'-'+vislabels[c]].values[j]=0
                
            if dataset['PRESSURE'+'_label'].values[j]==preslabels[c]:
                dataset['pressure'+'-'+preslabels[c]].values[j]=1
            else:
                dataset['pressure'+'-'+preslabels[c]].values[j]=0
                
                
    


dataset.to_csv(r'D:\Datasets\hpc\onehot_final.csv')