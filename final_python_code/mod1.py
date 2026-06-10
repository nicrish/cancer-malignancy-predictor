#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 18:49:50 2026

@author: nicolerishwain
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set option to display all columns by default
pd.set_option('display.max_columns', None)

filename1 =  "/Users/nicolerishwain/src/CS432/ML/Prostate_Cancer.csv"
filename2 = "/Users/nicolerishwain/src/CS432/ML/breast-cancer.csv"

#read dataset as dataframe

df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename2)



#print the dataframe
print(df1)
print(df1.dtypes)
print((df1.isnull().sum()))


plt.show() 
# Create box plot
sns.boxplot(x="radius", data=df1, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="texture", data=df1, whis=np.inf)


plt.show() 
# Create box plot
sns.boxplot(x="perimeter", data=df1, whis=np.inf)

plt.show()
# Create box plot
sns.boxplot(x="area", data=df1, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="smoothness", data=df1, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="compactness", data=df1, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="symmetry", data=df1, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="fractal_dimension", data=df1, whis=np.inf)

plt.show() 

diagnosis_plot=sns.countplot(data=df1, x="diagnosis_result", hue = "diagnosis_result", dodge=False, 
                 palette="Set2")
for p in diagnosis_plot.patches:
    height = p.get_height()                     
    x_pos = p.get_x() + 0.4      
    diagnosis_plot.annotate(
        f'{int(height)}',                       
        (x_pos, height),                        
        ha='center', va='bottom',               
        fontsize=8
     )
    
    
df1['diagnosis_result']= df1['diagnosis_result'].astype('category')
df1.drop('id', axis=1, inplace = True)

print(df1)

df1.to_csv('prostate_cancer_clean.csv')

#print the dataframe
print(df2)
print(df2.dtypes)
print((df2.isnull().sum()))




plt.show() 
# Create box plot
sns.boxplot(x="radius_mean", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="texture_mean", data=df2, whis=np.inf)


plt.show() 
# Create box plot
sns.boxplot(x="perimeter_mean", data=df2, whis=np.inf)

plt.show()
# Create box plot
sns.boxplot(x="area_mean", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="smoothness_mean", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="compactness_mean", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="concavity_mean", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="concave points_mean", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="symmetry_mean", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="fractal_dimension_mean", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="radius_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="texture_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="perimeter_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="area_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="smoothness_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="compactness_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="concavity_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="concave points_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="symmetry_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="fractal_dimension_se", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="radius_worst", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="texture_worst", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="perimeter_worst", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="area_worst", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="smoothness_worst", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="compactness_worst", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="concavity_worst", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="concave points_worst", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="symmetry_worst", data=df2, whis=np.inf)

plt.show() 
# Create box plot
sns.boxplot(x="fractal_dimension_worst", data=df2, whis=np.inf)









plt.show() 

diagnosis_plot_2=sns.countplot(data=df2, x="diagnosis", hue = "diagnosis", dodge=False, 
                 palette="Set2")
for p in diagnosis_plot_2.patches:
    height = p.get_height()                     
    x_pos = p.get_x() + 0.4      
    diagnosis_plot_2.annotate(
        f'{int(height)}',                       
        (x_pos, height),                        
        ha='center', va='bottom',               
        fontsize=8
     )
    
    
df2['diagnosis']= df2['diagnosis'].astype('category')
df2.drop('id', axis=1, inplace = True)

print(df2)

df2.to_csv('breast_cancer_clean.csv')







