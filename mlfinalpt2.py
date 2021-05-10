# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:20:25 2021

@author: narayanann2
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import simpledialog
from tkinter import filedialog
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
import seaborn as sns 


root=tk.Tk()

#reading data in
data=pd.read_csv('onlyonetimeperiod.csv')
data=pd.DataFrame(data)

#drop unnecessary data
data=data.drop(['State', 'Phase', 'Time Period Label', 'Confidence Interval', 'Quartile Range', 'LowCI', 'HighCI', 'Time Period'],1)

#what group are we analyzing
group=simpledialog.askstring("Input","What group are we analyzing?")

#subset matrix by group to analyze
newdata= data[data["Group"]==group]     
newdata=newdata.drop(['Group'],1)
dataint=newdata.set_index(['Indicator', 'Subgroup']).unstack()
dataint=dataint.reset_index()

labels=dataint['Indicator']
dataint=dataint.drop('Indicator', axis=1)

xlabels=dataint.columns.levels[1]



# sns.heatmap(feature_impt, annot=True, yticklabels=labels, xticklabels=xlabels)
# name=group+'.png'
# plt.savefig(name)

