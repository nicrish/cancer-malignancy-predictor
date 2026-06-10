#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 17:39:42 2026

@author: nicolerishwain
"""



## Import Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns  ## This will be for the prettier confusion matrix vis
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.model_selection import train_test_split
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
import matplotlib.pyplot as plt
#https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html
##
## LINK TO THE DATASET
## https://drive.google.com/file/d/1FEZ5tsTRBZsakBqXPB0c4TZCR17CNDsf/view?usp=sharing
##
## -- Read in and print the dataset. 
## If you are working with a large dataset, just print the first 10 - 15 rows.
## !! ATTENTION - Remember - this is my path. You will need to update this to your path
filepath="/Users/nicolerishwain/src/CS432/ML/prostate_cancer_clean.csv"
MyDataSet=pd.read_csv(filepath)
pd.set_option('display.max_columns', None)
print(MyDataSet)
# MyDataSet = MyDataSet.drop(["Unnamed: 0", "diagnosis_result", "perimeter", 
#                             "area", "smoothness", "compactness", 
#                             "fractal_dimension"], axis=1)
MyDataSet = MyDataSet.drop(["Unnamed: 0"], axis=1)
print(MyDataSet)
MyDataSet2 = MyDataSet.drop(["diagnosis_result"], axis=1)
corr = MyDataSet2.corr()
print(corr)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(MyDataSet["compactness"], MyDataSet["symmetry"], MyDataSet["fractal_dimension"])

ax.set_xlabel("compactness")
ax.set_ylabel("symmetry")
ax.set_zlabel("fractal_dimension")

plt.show()

# -------------------------------
# MULTIVARIABLE X and Y
# -------------------------------
X = MyDataSet[["compactness", "fractal_dimension"]]
Y = MyDataSet[["symmetry"]]

print(X.shape)  # (10, 4)
print(Y.shape)  # (10, 1)

# -------------------------------
# Train / Test split
# -------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)

# -------------------------------
# Train Linear Regression Model
# -------------------------------
MyLR = LinearRegression()
MyLR.fit(x_train, y_train)

# -------------------------------
# Predictions (MOVE THIS UP)
# -------------------------------
y_pred = MyLR.predict(x_test)


# -------------------------------
# Model parameters
# -------------------------------
print("Coefficients:")
for feature, coef in zip(X.columns, MyLR.coef_[0]):
    print(f"{feature}: {coef}")

print("Intercept:")
print(MyLR.intercept_)



# -------------------------------
# Model accuracy (R^2)
# -------------------------------
print("R^2 score:")
print(MyLR.score(x_train, y_train))


x1_range = np.linspace(MyDataSet["compactness"].min(), MyDataSet["compactness"].max(), 30)
x2_range = np.linspace(MyDataSet["fractal_dimension"].min(), MyDataSet["fractal_dimension"].max(), 30)

x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

y_pred = MyLR.predict(
    np.c_[x1_grid.ravel(), x2_grid.ravel()]
).reshape(x1_grid.shape)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(222, projection='3d')
# Regression plane
ax.plot_surface(
    x1_grid, x2_grid, y_pred,
    color="red", alpha=0.4
)
ax.scatter(MyDataSet["compactness"], MyDataSet["symmetry"], MyDataSet["fractal_dimension"])

ax.set_xlabel("compactness", labelpad=5)
ax.set_ylabel("symmetry", labelpad=5)
ax.set_zlabel("fractal_dimension", labelpad=5)
plt.title("Linear Regression", pad = 5)


fig.subplots_adjust(
    left=0.01,
    right=0.99,
    bottom=0.05,
    top=0.90
)
plt.show()


## We want to Train the model and then we want to Test the model
## To do this, we need to split up the data into:
    ## Training Data and Training Label
    ## Testing Data and Testing Label
## There are many ways to do this. 

TrainingData, TestingData = train_test_split(MyDataSet, test_size=.3)
print(TrainingData)
print(TestingData)

## Next, remove and save the labels from the Training Data
TrainingLabels = TrainingData["diagnosis_result"]
## Drop the label from the TrainingData now that we have saved it
TrainingData=TrainingData.drop(["diagnosis_result"], axis=1)
## print everything to make sure it looks right
print("The Training Labels are:")
print(TrainingLabels)
print("The Training Data is:")
print(TrainingData)

##Now - repeat this for the Testing Data so that you 
## end up with Testing Data and Testing Labels
TestingLabels = TestingData["diagnosis_result"]
## Drop the label from the TrainingData now that we have saved it
TestingData=TestingData.drop(["diagnosis_result"], axis=1)
## print everything to make sure it looks right
print("The Testing Labels are:")
print(TestingLabels)
print("The Testing Data is:")
print(TestingData)

##-------------------------
## Make sure you understand what we have.
## We now have Training Data and the Training Labels 
## We also have the Testing Data and the Testing Labels
##-------------------------------------------------------

##########################################
## Perform Logistic Regression
##
## In Sklearn, there are many parameter options
## for Logistic Regression. 
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
## We will use the defaults here.
###########################################################################

## Instantiate Logistic Regression (create your own copy)
## Notice that we are choosing to use the "default" Sklearn 
## Logistic Regression Library values because we are instantiating
## with nothing in the parentheses. 
MyLR = LogisticRegression()

##Perform Logistic Regression - using your copy - on the Training Data and Training Labels
My_LR_Model=MyLR.fit(TrainingData, TrainingLabels)

## Now that we created the model, we can do several things.
## 1) We can use the model to predict the Testing Data labels
##    and then we can compare the predictions to the actual labels.
##    We can use a Confusion Matrix to compare. 
## 2) We can get the parameters of the model we created so that
##    we can write down the actual model. 
##    For example, we can get the coefficients (coef_) and
##    the intercept (intercept_) so that we can 
##    build the y = w1x1 + w2x2 + ...wnxn + b
##
##    Here, the w1, w2, ..., wn are the coefficients
##    and the "b" is the intercept.
##    See the example in the code lower down .....

## -----------------------------------
## Use the model to predict the Test data
## -------------------------------------------------
MyModelPredictions=My_LR_Model.predict(TestingData)
print("predictions")
print(MyModelPredictions)
y_probs = My_LR_Model.predict_proba(TestingData)
print(y_probs)

## Print the actual labels
print(TestingLabels)

## Create a standard Confusion Matrix to compare the actual and predicted labels
MyCM=confusion_matrix(TestingLabels, MyModelPredictions)
print(MyCM)

## Use Seaborn to create a pretty confusion matrix visualization
sns.heatmap(MyCM, annot=True, cmap='Blues')
from sklearn.metrics import roc_curve, auc

# Probability of class "M"
y_prob_M = y_probs[:, 1]

# Convert labels to binary (M=1, B=0)
TestingLabelsBinary = (TestingLabels == "M").astype(int)

fpr, tpr, _ = roc_curve(TestingLabelsBinary, y_prob_M)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()
##-------------------------------------------
## Print some properties of the model
##---------------------------------------------
## We can print an accuracy score for the model
print(My_LR_Model.score(TrainingData, TrainingLabels))
## Recall that Logistic Regression calculates a value between
## 0 and 1 (a probability) for each prediction using the Sigmoid. 
## A threshold (cut-off) of .5 is used for the final
## prediction of 0 or 1. 
## However, you have to open to seeing the original probabilities
## as they were before they were thresholded to 0 or 1
print(My_LR_Model.predict_proba(TestingData))

##------------
## Printing the model
##-------------------------
## Recall the formula:
    ## y - w1x1 + w2x1 + .. + wnxn + b
## We can print the coefficients (the weights) of the model
print("coef")
print(My_LR_Model.coef_)
## and we can print the intercept (b) for the model
print("b")
print(My_LR_Model.intercept_)



## For a coef_ output of:
    ## [[0.02394083 0.54278591 0.13992734]]
## and for an intercept output of [-82.43127784]
## Our actual model is (rounded):
    ## y = .024x1 + .543x2 + .140x3  - 82.44
## We can plug in our variable values here, calculate y
## and then apply the Sigmoid to get any final classification
## for any new data.

## For example, suppose we have a new student with 
## a GPA (x1) of 3.82, a Q_GRE (x2) of 166, and Months_Int (x3) if 5
## 
## Then we have:
    ## y = .024x1 + .543x2 + .140x3  - 82.44
    ## y = .024(3.82) + .543(166) + .140(5)  - 82.44
    ## y = 8.5
    
    ## Then, applying the Sigmoid:
        ## S(8.5) = 1/ (1 + e^(-8.5))  = .9998
        ## Which predicts as "1"