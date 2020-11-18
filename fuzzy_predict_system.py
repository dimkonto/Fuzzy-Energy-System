# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:47:31 2020

@author: dimkonto
"""

import pandas as pd
import numpy as np
import math
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from matplotlib import pyplot as pp
import datetime
#load dataset
path = r'D:\Datasets\hpc\energydata_complete.csv'
dataset = pd.read_csv(path,sep=',',header=0,low_memory=False,infer_datetime_format=True)

#print shape and a few records
#print(dataset.shape)
#print(dataset.head())
for i in range(9):
    i=i+1
    #print(dataset['T'+str(i)].min())
    #print(dataset['T'+str(i)].max())

#print(dataset['RH_1'].min())
#print(dataset['RH_1'].max())


#print(dataset['Windspeed'].min())
#print(dataset['Windspeed'].max())

#print(dataset['Visibility'].min())
#print(dataset['Visibility'].max())

#print(dataset['Press_mm_hg'].min())
#print(dataset['Press_mm_hg'].max())


#print(dataset['Appliances'].min())
#print(dataset['Appliances'].max())



#print(dataset['T1'].values[0])
#Box Plots
#dataset.plot(kind='box',subplots=True,layout=(6,6),sharex=False, figsize=(20,15))
#pp.savefig(r'D:\Datasets\hpc\charts\boxplt_fuzz.png',dpi=300,bbox_inches="tight")
#pp.show()

#Temperature of room membership
temp_1 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_1')
temp_1['verycold'] = fuzz.zmf(temp_1.universe, -7, 3)
temp_1['cold'] = fuzz.trimf(temp_1.universe,[0,5,10])
temp_1['cool'] = fuzz.trimf(temp_1.universe,[8,12,15])
temp_1['warm'] = fuzz.trimf(temp_1.universe,[12,17,20])
temp_1['hot'] = fuzz.smf(temp_1.universe, 18, 30)

#temp_1.view()
#pp.savefig(r'D:\Datasets\hpc\charts\temp_fuzz.png',dpi=300,bbox_inches="tight")

temp_2 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_2')
temp_2['verycold'] = fuzz.zmf(temp_2.universe, -7, 3)
temp_2['cold'] = fuzz.trimf(temp_2.universe,[0,5,10])
temp_2['cool'] = fuzz.trimf(temp_2.universe,[8,12,15])
temp_2['warm'] = fuzz.trimf(temp_2.universe,[12,17,20])
temp_2['hot'] = fuzz.smf(temp_2.universe, 18, 30)

temp_3 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_3')
temp_3['verycold'] = fuzz.zmf(temp_3.universe, -7, 3)
temp_3['cold'] = fuzz.trimf(temp_3.universe,[0,5,10])
temp_3['cool'] = fuzz.trimf(temp_3.universe,[8,12,15])
temp_3['warm'] = fuzz.trimf(temp_3.universe,[12,17,20])
temp_3['hot'] = fuzz.smf(temp_3.universe, 18, 30)

temp_4 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_4')
temp_4['verycold'] = fuzz.zmf(temp_4.universe, -7, 3)
temp_4['cold'] = fuzz.trimf(temp_4.universe,[0,5,10])
temp_4['cool'] = fuzz.trimf(temp_4.universe,[8,12,15])
temp_4['warm'] = fuzz.trimf(temp_4.universe,[12,17,20])
temp_4['hot'] = fuzz.smf(temp_4.universe, 18, 30)

temp_5 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_5')
temp_5['verycold'] = fuzz.zmf(temp_5.universe, -7, 3)
temp_5['cold'] = fuzz.trimf(temp_5.universe,[0,5,10])
temp_5['cool'] = fuzz.trimf(temp_5.universe,[8,12,15])
temp_5['warm'] = fuzz.trimf(temp_5.universe,[12,17,20])
temp_5['hot'] = fuzz.smf(temp_5.universe, 18, 30)

temp_6 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_6')
temp_6['verycold'] = fuzz.zmf(temp_6.universe, -7, 3)
temp_6['cold'] = fuzz.trimf(temp_6.universe,[0,5,10])
temp_6['cool'] = fuzz.trimf(temp_6.universe,[8,12,15])
temp_6['warm'] = fuzz.trimf(temp_6.universe,[12,17,20])
temp_6['hot'] = fuzz.smf(temp_6.universe, 18, 30)

temp_7 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_7')
temp_7['verycold'] = fuzz.zmf(temp_7.universe, -7, 3)
temp_7['cold'] = fuzz.trimf(temp_7.universe,[0,5,10])
temp_7['cool'] = fuzz.trimf(temp_7.universe,[8,12,15])
temp_7['warm'] = fuzz.trimf(temp_7.universe,[12,17,20])
temp_7['hot'] = fuzz.smf(temp_7.universe, 18, 30)

temp_8 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_8')
temp_8['verycold'] = fuzz.zmf(temp_8.universe, -7, 3)
temp_8['cold'] = fuzz.trimf(temp_8.universe,[0,5,10])
temp_8['cool'] = fuzz.trimf(temp_8.universe,[8,12,15])
temp_8['warm'] = fuzz.trimf(temp_8.universe,[12,17,20])
temp_8['hot'] = fuzz.smf(temp_8.universe, 18, 30)

temp_9 = ctrl.Antecedent(np.arange(-7, 31,0.01), 'temp_9')
temp_9['verycold'] = fuzz.zmf(temp_9.universe, -7, 3)
temp_9['cold'] = fuzz.trimf(temp_9.universe,[0,5,10])
temp_9['cool'] = fuzz.trimf(temp_9.universe,[8,12,15])
temp_9['warm'] = fuzz.trimf(temp_9.universe,[12,17,20])
temp_9['hot'] = fuzz.smf(temp_9.universe, 18, 30)

#temp_1.view()

#Humidity Percentage membership
humidity_1 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_1')
humidity_1['dry'] = fuzz.trapmf(humidity_1.universe,[0,0,25,40])
humidity_1['comfortable'] = fuzz.trapmf(humidity_1.universe,[25,40,60,75])
humidity_1['humid'] = fuzz.trapmf(humidity_1.universe,[60,75,100,100])

#humidity_1.view()
#pp.savefig(r'D:\Datasets\hpc\charts\humidity_fuzz.png',dpi=300,bbox_inches="tight")

humidity_2 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_2')
humidity_2['dry'] = fuzz.trapmf(humidity_2.universe,[0,0,25,40])
humidity_2['comfortable'] = fuzz.trapmf(humidity_2.universe,[25,40,60,75])
humidity_2['humid'] = fuzz.trapmf(humidity_2.universe,[60,75,100,100])

humidity_3 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_3')
humidity_3['dry'] = fuzz.trapmf(humidity_3.universe,[0,0,25,40])
humidity_3['comfortable'] = fuzz.trapmf(humidity_3.universe,[25,40,60,75])
humidity_3['humid'] = fuzz.trapmf(humidity_3.universe,[60,75,100,100])

humidity_4 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_4')
humidity_4['dry'] = fuzz.trapmf(humidity_4.universe,[0,0,25,40])
humidity_4['comfortable'] = fuzz.trapmf(humidity_4.universe,[25,40,60,75])
humidity_4['humid'] = fuzz.trapmf(humidity_4.universe,[60,75,100,100])

humidity_5 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_5')
humidity_5['dry'] = fuzz.trapmf(humidity_5.universe,[0,0,25,40])
humidity_5['comfortable'] = fuzz.trapmf(humidity_5.universe,[25,40,60,75])
humidity_5['humid'] = fuzz.trapmf(humidity_5.universe,[60,75,100,100])

humidity_6 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_6')
humidity_6['dry'] = fuzz.trapmf(humidity_6.universe,[0,0,25,40])
humidity_6['comfortable'] = fuzz.trapmf(humidity_6.universe,[25,40,60,75])
humidity_6['humid'] = fuzz.trapmf(humidity_6.universe,[60,75,100,100])

humidity_7 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_7')
humidity_7['dry'] = fuzz.trapmf(humidity_7.universe,[0,0,25,40])
humidity_7['comfortable'] = fuzz.trapmf(humidity_7.universe,[25,40,60,75])
humidity_7['humid'] = fuzz.trapmf(humidity_7.universe,[60,75,100,100])

humidity_8 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_8')
humidity_8['dry'] = fuzz.trapmf(humidity_8.universe,[0,0,25,40])
humidity_8['comfortable'] = fuzz.trapmf(humidity_8.universe,[25,40,60,75])
humidity_8['humid'] = fuzz.trapmf(humidity_8.universe,[60,75,100,100])

humidity_9 = ctrl.Antecedent(np.arange(0,101,0.01), 'humidity_9')
humidity_9['dry'] = fuzz.trapmf(humidity_9.universe,[0,0,25,40])
humidity_9['comfortable'] = fuzz.trapmf(humidity_9.universe,[25,40,60,75])
humidity_9['humid'] = fuzz.trapmf(humidity_9.universe,[60,75,100,100])

#humidity.view()

#Wind Speed Membership
windspeed = ctrl.Antecedent(np.arange(0,21,0.01), 'windspeed')

windspeed['Low'] = fuzz.trimf(windspeed.universe,[0,0,4])
windspeed['Medium'] = fuzz.trimf(windspeed.universe,[3,5,7])
windspeed['High'] = fuzz.trapmf(windspeed.universe,[6,10,20,20])

#windspeed.view()
#pp.savefig(r'D:\Datasets\hpc\charts\windspeed_fuzz.png',dpi=300,bbox_inches="tight")

#Visibility Membership
visibility = ctrl.Antecedent(np.arange(0,76,0.01), 'visibility')

visibility['Low'] = fuzz.trimf(visibility.universe,[0,0,40])
visibility['Medium'] = fuzz.trimf(visibility.universe,[30,45,60])
visibility['High'] = fuzz.trapmf(visibility.universe,[50,65,75,75])

#visibility.view()
#pp.savefig(r'D:\Datasets\hpc\charts\visibility_fuzz.png',dpi=300,bbox_inches="tight")


#Pressure Membership
pressure = ctrl.Antecedent(np.arange(700,801,0.01), 'pressure')

pressure['Low'] = fuzz.trimf(pressure.universe,[700,700,740])
pressure['Medium'] = fuzz.trimf(pressure.universe,[720,750,780])
pressure['High'] = fuzz.trapmf(pressure.universe,[760,790,800,800])

#pressure.view()
#pp.savefig(r'D:\Datasets\hpc\charts\pressure_fuzz.png',dpi=300,bbox_inches="tight")

#Membership for Output - Appliance Consumption - Consequent
consumption = ctrl.Consequent(np.arange(0,1201,1), 'consumption')

consumption['Low'] = fuzz.trimf(consumption.universe,[0,0,300])
consumption['Medium'] = fuzz.trimf(consumption.universe,[100,300,500])
consumption['High'] = fuzz.trimf(consumption.universe,[300,500,700])

consumption['VeryHigh'] = fuzz.trapmf(consumption.universe,[500,800,1200,1200])

consumption.view()
pp.savefig(r'D:\Datasets\hpc\charts\consumption_fuzz.png',dpi=300,bbox_inches="tight")



"""
#FUZZIFY INPUT - Find degrees of membership and get the max
#INITIALIZE DEGREE MATRICES
#tempdeg
deg1=np.empty([9,5])
#humiditydeg
deg2=np.empty([10,3])
deg3=np.empty(3)
deg4=np.empty(3)
deg5=np.empty(3)
deg6=np.empty(3)
deg7=np.empty(5)
deg8=np.empty(5)
deg9=np.empty(5)

degout=np.empty(4)

#SETLABELS FOR FUZZY VARIABLES

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
#LABELS FOR APPLIANCE CONSUMPTION OUTPUT
outlabels=['Low','Medium','High','VeryHigh']


#CREATE EXTRA DATASET COLUMNS FOR EACH LABELED VARIABLE and FUZZIFY

# LOOP FOR TEMPERATURE NEW LABELED COLUMNS
for k in range(9):
    number=k+1
    dataset['T'+str(number)+'_label']=""
    dataset['T'+str(number)+'_verycold']=""
    dataset['T'+str(number)+'_cold']=""
    dataset['T'+str(number)+'_cool']=""
    dataset['T'+str(number)+'_warm']=""
    dataset['T'+str(number)+'_hot']=""

# LOOP FOR HUMIDITY NEW LABELED COLUMNS
for k in range(9):
    number=k+1
    dataset['RH'+str(number)+'_label']=""
    dataset['RH'+str(number)+'_dry']=""
    dataset['RH'+str(number)+'_comfortable']=""
    dataset['RH'+str(number)+'_humid']=""
#LABELED WINDSPEED,VISIBILITY & PRESSURE
dataset['WIND_label']=""
dataset['WIND_Low']=""
dataset['WIND_Medium']=""
dataset['WIND_High']=""
    
dataset['VISIBILITY_label']=""
dataset['VISIBILITY_Low']=""
dataset['VISIBILITY_Medium']=""
dataset['VISIBILITY_High']=""
    
dataset['PRESSURE_label']=""
dataset['PRESSURE_Low']=""
dataset['PRESSURE_Medium']=""
dataset['PRESSURE_High']=""

dataset['APPLIANCE_label']=""
dataset['APPLIANCE_Low']=""
dataset['APPLIANCE_Medium']=""
dataset['APPLIANCE_High']=""
dataset['APPLIANCE_VeryHigh']=""      
    

#dataset['T1_label'].values[0]='elephant'

#FUZZIFICATION OF TEMPERATURE
#dataset['T1'].shape[0] is the loop
for j in range(dataset['T1'].shape[0]):
    #print(dataset['T1'].values[dataset['T1'].shape[0]-j-1])
    #FOR EACH ROW POPULATE DEGREE ARRAY
    for m in range(9):
        #print(dataset['T'+str(m+1)].values[j])
        for i in range(5):
            #deg1[i]=fuzz.interp_membership(temp_1.universe,temp[templabels[i]].mf,dataset['T1'].values[j])
            deg1[m][i]=fuzz.interp_membership(temp_1.universe,temp[templabels[i]].mf,dataset['T'+str(m+1)].values[j])
            if deg1[m][i]>0:
                dataset['T'+str(m+1)+'_'+templabels[i]].values[j]=1
            else:
                dataset['T'+str(m+1)+'_'+templabels[i]].values[j]=0
           
        #print(deg[i])
    #maxlabelindex=np.where(deg1==deg1.max())[0]   
    #print(maxlabelindex[0]) 
    #print(templabels[maxlabelindex[0]])
    #dataset['T1_label'].values[j]=templabels[maxlabelindex[0]]
    #NEWCODE
    
    #deg1=np.empty([9,5])
    #print(deg1)
    #print(np.amax(deg1,axis=1))
    #print(np.argmax(deg1,axis=1))
    
    #FIND THE LABEL BASED ON DEGREE ARRAY AND PUT IT IN EACH NEW COLUMN
    maxlabelindex=np.argmax(deg1,axis=1)
    for n in range(9):
        dataset['T'+str(n+1)+'_label'].values[j]=templabels[maxlabelindex[n]]
        #print(dataset['T'+str(n+1)+'_label'].values[j])
       
#print(dataset.head())

#FUZZIFY HUMIDITY
for j in range(dataset['T1'].shape[0]):
    #FOR EACH ROW POPULATE DEGREE ARRAY
    for m in range(9):
        #print(dataset['T'+str(m+1)].values[j])
        for i in range(3):
            deg2[m][i]=fuzz.interp_membership(humidity_1.universe,humidity[humlabels[i]].mf,dataset['RH_'+str(m+1)].values[j])
            if deg2[m][i]>0:
                dataset['RH'+str(m+1)+'_'+humlabels[i]].values[j]=1
            else:
                dataset['RH'+str(m+1)+'_'+humlabels[i]].values[j]=0

    maxlabelindex=np.argmax(deg2,axis=1)
    for n in range(9):
        dataset['RH'+str(n+1)+'_label'].values[j]=humlabels[maxlabelindex[n]]
        #print(dataset['T'+str(n+1)+'_label'].values[j])        

#FUZZIFY WIND VISIBILITY PRESSURE and APPLIANCE OUTPUT
for j in range(dataset['T1'].shape[0]):
    #FOR EACH ROW POPULATE DEGREE ARRAY

    for i in range(3):
        deg3[i]=fuzz.interp_membership(windspeed.universe,windspeed[windlabels[i]].mf,dataset['Windspeed'].values[j])
        if deg3[i]>0:
            dataset['WIND_'+windlabels[i]].values[j]=1
        else:
            dataset['WIND_'+windlabels[i]].values[j]=0
        deg4[i]=fuzz.interp_membership(visibility.universe,visibility[vislabels[i]].mf,dataset['Visibility'].values[j])
        if deg4[i]>0:
            dataset['VISIBILITY_'+vislabels[i]].values[j]=1
        else:
            dataset['VISIBILITY_'+vislabels[i]].values[j]=0
        deg5[i]=fuzz.interp_membership(pressure.universe,pressure[preslabels[i]].mf,dataset['Press_mm_hg'].values[j])
        if deg5[i]>0:
            dataset['PRESSURE_'+preslabels[i]].values[j]=1
        else:
            dataset['PRESSURE_'+preslabels[i]].values[j]=0
    for i in range(4):
        degout[i]=fuzz.interp_membership(consumption.universe,consumption[outlabels[i]].mf,dataset['Appliances'].values[j])
        if degout[i]>0:
            dataset['APPLIANCE_'+outlabels[i]].values[j]=1
        else:
            dataset['APPLIANCE_'+outlabels[i]].values[j]=0
        
        
    maxlabelindex=np.argmax(deg3)
    dataset['WIND_label'].values[j]=windlabels[maxlabelindex]
    
    maxlabelindex=np.argmax(deg4)
    dataset['VISIBILITY_label'].values[j]=vislabels[maxlabelindex]
    
    maxlabelindex=np.argmax(deg5)
    dataset['PRESSURE_label'].values[j]=preslabels[maxlabelindex]
    
    maxlabelindex=np.argmax(degout)
    dataset['APPLIANCE_label'].values[j]=outlabels[maxlabelindex]
        #print(dataset['T'+str(n+1)+'_label'].values[j])        



#save as new csv
#dataset.to_csv(r'D:\Datasets\hpc\fuzzified.csv')
#1HOT ENCODED        
#dataset.to_csv(r'D:\Datasets\hpc\fuzzy1hot.csv')
dataset.to_csv(r'D:\Datasets\hpc\fuzzy1hot_v3.csv')


"""


"""
#EXAMPLE RULES
exec("rule1 =ctrl.Rule(temp['hot'], consumption['Low'])")
rule2 = ctrl.Rule(pressure['Medium'] | temp['hot'], consumption['Medium'])
rule3 = ctrl.Rule(temp['hot'] | windspeed['Medium'], consumption['High'])
rule1.view()
rule2.view()
rule3.view()
"""



#READ AND PARSE RULES INTO THE SYSTEM
rulefile=open("D:/Datasets/hpc/paths.txt",'r')
rulelines=rulefile.readlines()

#PARSE RULES INTO CONSEQUENTS AND ANTECEDENTS/// CREATING RULEBASE 
count=0
consequent=""
antecedent=""
rule_consequent=""
rulebase=[]
ctrlargs=""
ctrlsys=""
for line in rulelines:
    ruleunparsed=line.strip()
    #print(ruleunparsed)
    rulecomponents=ruleunparsed.split("&")
    #print(rulecomponents)
    for r in rulecomponents:
        if r!="":
            #print(r)
            component_split=r.split(":")
            #print(component_split)
            if component_split[1]!="":
                antecedent_parts=component_split[0].split("-")
                #print(antecedent_parts)
                if component_split[1]=="  False":
                    antecedent=antecedent +"~"+ antecedent_parts[0] + "['"+antecedent_parts[1]+"']" + "&"
                else:
                    antecedent=antecedent + antecedent_parts[0] + "['"+antecedent_parts[1]+"']" + "&"
               #rule
            else:
                consequent="consumption['" + component_split[0] + "']"
                
    antecedent=antecedent[:-1]
    #print(antecedent)
    #print(consequent)
    rule="rule"+str(count)+"="+"ctrl.Rule("+antecedent+","+consequent+")"
    ctrlargs=ctrlargs+"rule"+str(count)+","
    #print(rule)
    rulebase.append(rule)
    count=count+1
    consequent=""
    antecedent=""
    #break


ctrlargs=ctrlargs[:-1]
#print(ctrlargs)
ctrlsys="consumption_ctrl = ctrl.ControlSystem(["+ctrlargs+"])"
print(ctrlsys)

#RULE EXECUTION
for ru in rulebase:
    #print(ru)
    exec(ru)
    #exec("rule0.view()")

exec(ctrlsys)
exec("consumption_simulation = ctrl.ControlSystemSimulation(consumption_ctrl)")

#PASS INPUTS PER ROW OF THE ORIGINAL DATASET AND TEST THE SYSTEM BY COMPUTING
#INPUTS NEED TO BE DERIVED FROM THE BEST FEATURES/FILTERED
#print(dataset.head())

for j in range(dataset['T1'].shape[0]):
    #Bind temperature

    #print(dataset['Appliances'].values[j])
    if any("temp_1" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_1']="+str(dataset['T1'].values[j]))
    else:
        print("temp_1 Not in rulebase")
    
    if any("temp_2" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_2']="+str(dataset['T2'].values[j]))
    else:
        print("temp_2 Not in rulebase")
    
    if any("temp_3" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_3']="+str(dataset['T3'].values[j]))
    else:
        print("temp_3 Not in rulebase")
        
    if any("temp_4" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_4']="+str(dataset['T4'].values[j]))
    else:
        print("temp_4 Not in rulebase")
        
    if any("temp_5" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_5']="+str(dataset['T5'].values[j]))
    else:
        print("temp_5 Not in rulebase")
        
    if any("temp_6" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_6']="+str(dataset['T6'].values[j]))
    else:
        print("temp_6 Not in rulebase")
    if any("temp_7" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_7']="+str(dataset['T7'].values[j]))
    else:
        print("temp_7 Not in rulebase")
    if any("temp_8" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_8']="+str(dataset['T8'].values[j]))
    else:
        print("temp_8 Not in rulebase")
    if any("temp_9" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['temp_9']="+str(dataset['T9'].values[j]))
    else:
        print("temp_9 Not in rulebase")
    
    #Bind Humidity/////////////////////////////////////////////////////
    if any("humidity_1" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_1']="+str(dataset['RH_1'].values[j]))
    else:
        print("humidity_1 Not in rulebase")
    
    if any("humidity_2" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_2']="+str(dataset['RH_2'].values[j]))
    else:
        print("humidity_2 Not in rulebase")
    
    if any("humidity_3" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_3']="+str(dataset['RH_3'].values[j]))
    else:
        print("humidity_3 Not in rulebase")
        
    if any("humidity_4" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_4']="+str(dataset['RH_4'].values[j]))
    else:
        print("humidity_4 Not in rulebase")
        
    if any("humidity_5" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_5']="+str(dataset['RH_5'].values[j]))
    else:
        print("humidity_5 Not in rulebase")
        
    if any("humidity_6" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_6']="+str(dataset['RH_6'].values[j]))
    else:
        print("humidity_6 Not in rulebase")
    if any("humidity_7" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_7']="+str(dataset['RH_7'].values[j]))
    else:
        print("humidity_7 Not in rulebase")
    if any("humidity_8" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_8']="+str(dataset['RH_8'].values[j]))
    else:
        print("humidity_8 Not in rulebase")
    if any("humidity_9" in inputterm for inputterm in rulebase):
        exec("consumption_simulation.input['humidity_9']="+str(dataset['RH_9'].values[j]))
    else:
        print("humidity_9 Not in rulebase")
    
    
    #Bind windspeed, visibility, pressure
    if any("windspeed" in inputterm for inputterm in rulebase):    
        exec("consumption_simulation.input['windspeed']="+str(dataset['Windspeed'].values[j]))
    else:
        print("windspeed Not in rulebase")
    if any("visibility" in inputterm for inputterm in rulebase):    
        exec("consumption_simulation.input['visibility']="+str(dataset['Visibility'].values[j]))
    else:
        print("visibility Not in rulebase")
    if any("pressure" in inputterm for inputterm in rulebase):    
        exec("consumption_simulation.input['pressure']="+str(dataset['Press_mm_hg'].values[j]))
    else:
        print("pressure Not in rulebase")
    
    #break

    #COMPUTATION
    begin_time=datetime.datetime.now()
    exec("consumption_simulation.compute()")
    print(datetime.datetime.now()-begin_time)
    
    exec("print(consumption_simulation.output['consumption'])")
    exec("consumption.view(sim=consumption_simulation)")
    pp.savefig(r'D:\Datasets\hpc\charts\consumption_simulation.png',dpi=300,bbox_inches="tight")
    break
    #exec("print(consumption_simulation.output['consumption'])")

        
    
    
    