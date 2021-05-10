# -*- coding: utf-8 -*-
"""
Created on Sun May  9 00:16:04 2021

@author: narayanann2
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import simpledialog

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
import seaborn as sns

root=tk.Tk()

class SVM():
    def __init__(self, csv):
        data=pd.read_csv(csv)

        #drop unnecessary data
        data=data.drop(['State', 'Phase', 'Time Period Label', 'Confidence Interval', 'Quartile Range', 'LowCI', 'HighCI', 'Time Period'],1)
        
        #what group are we analyzing
        group=simpledialog.askstring("Input","What group are we analyzing?")
        
        #subset matrix by group to analyze
        data= data[data["Group"]==group]  
        labels=data['Indicator']
        data=data.drop(['Indicator', 'Group'],1)           
        
        self.group=group
        self.labels=labels.reset_index(drop=True)
        self.data=data
        self.accuracy=0
    
    def ohe(self):
        #onehot encoding the subgroups
        features= {}
        newdata=self.data

        for item in set(newdata['Subgroup'].values):
            features[item] = []
   
        for value in newdata['Subgroup']:
            for key in features:
                if value == key:
                    features[key].append(1)
                else:
                    features[key].append(0)
        for key in features:
           newdata[key]=features[key]
         
        self.data=newdata.drop(['Subgroup'],1)
        self.columns=list(self.data.columns)
        return self.data, self.columns
    
    
    def scale(self):
        #scaling our values
        scaled_data=self.data.values
        scaler=MinMaxScaler()
        self.data=scaler.fit_transform(scaled_data)
        self.data=pd.DataFrame(self.data)
        return self.data
    
    def predict(self, kern='linear'):
        #implementing SVM model
        svm_model=SVC(kernel=kern)
        skf=StratifiedKFold(n_splits=4)
        
        #initializing arrays
        self.feature_impt= np.zeros((len(set(self.labels)), len(self.data.columns)))
        rand_labels=[]
        rand_predictions=[]
        
        
        #going thru training/testing data and predicting
        for train_index, test_index in skf.split(self.data,self.labels):
            data_test, labels_test= self.data.iloc[test_index], self.labels.iloc[test_index]
            data_train, labels_train= self.data.iloc[train_index], self.labels.iloc[train_index]
            print('train: %s, test: %s' % (train_index, test_index))
            svm_model.fit(data_train, labels_train)
            self.feature_impt+=svm_model.coef_
            predictions=svm_model.predict(data_test)
            
            rand_predictions.extend(predictions)
            rand_labels.extend(labels_test)
                
            self.accuracy+=accuracy_score(predictions,labels_test)
            
        #looking at the confusion matrix and accuracy
        cm=confusion_matrix(rand_labels, rand_predictions, normalize='true')
        print(cm)
            
        print(self.accuracy/4)
        return self.accuracy, self.feature_impt
    
    def visualize(self):
        #calculating average feature importances
        average_feature_importances = []
        num= self.feature_impt.shape[1]
        for i in range(1,num):
            average_value = 0
            for j in self.feature_impt:
                average_value += abs(j[i])
            average_feature_importances.append(average_value/4)
       
        plt.figure(figsize= (20,8))
        plt.bar(self.columns, average_feature_importances, width=0.5)                     
        plt.title(self.group)



svming=SVM('C:/Users/narayanann2/Documents/Python Scripts\onlyonetimeperiod.csv')
svming.ohe()
svming.scale()
svming.predict()
svming.visualize()
    