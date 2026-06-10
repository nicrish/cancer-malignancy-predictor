#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 11:15:40 2026

@author: nicolerishwain
"""

############################################################
##
## Naive Bayes Classification Code Example
## Gates, 2024
##
## This code uses the Prostate Cancer dataset for both
## Gaussian and Bernoulli Naive Bayes, plus a Decision Tree.
##
## Dataset: Prostate_Cancer.csv
## Label column: diagnosis_result (M = Malignant, B = Benign)
##
## !! Please note that when using code written by others
## it is a best practice to use the code as a 
## reference and resource - NOT as a copy/paste/hope.
##
############################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

## -----------------------------------------------
## Read in the dataset
## !! UPDATE THIS PATH to where your file is saved
## -----------------------------------------------
pd.set_option('display.max_columns', None)
dataset = "/Users/nicolerishwain/src/CS432/ML/Prostate_Cancer.csv"

df = pd.read_csv(dataset)

print("Prostate Cancer Dataset:")
print(df)
print(df.columns)



## -----------------------------------------------
## For GAUSSIAN NB - use the raw continuous features
## -----------------------------------------------

Dataset_Gaussian = df.copy()

## -----------------------------------------------
## For BERNOULLI NB - binarize all features
## Each feature is split at its median:
##   >= median --> 1
##   <  median --> 0
## We do this AFTER the train/test split to avoid
## data leakage (we use the TRAINING median only)
## -----------------------------------------------

Dataset_Bernoulli = df.copy()

## -----------------------------------------------
## Train/Test Split - Gaussian
## -----------------------------------------------

Training_G, Testing_G = train_test_split(
    Dataset_Gaussian,
    test_size=0.3,
    stratify=Dataset_Gaussian["diagnosis_result"],
    random_state=42
)

Training_G_Label = Training_G["diagnosis_result"]
Training_G = Training_G.drop(["diagnosis_result"], axis=1)
Testing_G_Label = Testing_G["diagnosis_result"]
Testing_G = Testing_G.drop(["diagnosis_result"], axis=1)

print("Testing G:", Testing_G)
print("Testing G Labels:", Testing_G_Label)

## -----------------------------------------------
## Train/Test Split - Bernoulli
## -----------------------------------------------

Training_B, Testing_B = train_test_split(
    Dataset_Bernoulli,
    test_size=0.3,
    stratify=Dataset_Bernoulli["diagnosis_result"],
    random_state=42
)

Training_B_Label = Training_B["diagnosis_result"]
Training_B = Training_B.drop(["diagnosis_result"], axis=1)
Testing_B_Label = Testing_B["diagnosis_result"]
Testing_B = Testing_B.drop(["diagnosis_result"], axis=1)

## Binarize using the TRAINING median to prevent data leakage
train_medians = Training_B.median()
Training_B = (Training_B >= train_medians).astype(int)
Testing_B  = (Testing_B  >= train_medians).astype(int)

print("Testing B (binarized):", Testing_B)
print("Testing B Labels:", Testing_B_Label)

## -----------------------------------------------
## Set up the 2x1 subplot for confusion matrices
## -----------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

####################################################
## Run Naive Bayes
####################################################

## -----------------------------------------------
## Gaussian Naive Bayes
## -----------------------------------------------

MyGNB = GaussianNB()

My_GNB_Model = MyGNB.fit(Training_G, Training_G_Label)
print(My_GNB_Model)

Predictions_G = My_GNB_Model.predict(Testing_G)
print("Gaussian Predictions:", Predictions_G)

print("The Gaussian NB Model Prediction Probabilities are:")
print(My_GNB_Model.predict_proba(Testing_G).round(3))

CM_G = confusion_matrix(
    Testing_G_Label,
    Predictions_G,
    labels=My_GNB_Model.classes_
)
print("Gaussian Confusion Matrix:")
print(CM_G)

disp = ConfusionMatrixDisplay(confusion_matrix=CM_G, display_labels=My_GNB_Model.classes_)
ax[0].set_title("Gaussian Naive Bayes")
disp.plot(ax=ax[0])

## -----------------------------------------------
## Bernoulli Naive Bayes
## -----------------------------------------------

MyBNB = BernoulliNB()

My_BNB_Model = MyBNB.fit(Training_B, Training_B_Label)
print(My_BNB_Model)
print(My_BNB_Model.classes_)

Predictions_B = My_BNB_Model.predict(Testing_B)
print("Bernoulli Predictions:", Predictions_B)

print("The Bernoulli NB Model Prediction Probabilities are:")
print(My_BNB_Model.predict_proba(Testing_B).round(3))

CM_B = confusion_matrix(
    Testing_B_Label,
    Predictions_B,
    labels=My_BNB_Model.classes_
)
print("Bernoulli Confusion Matrix:")
print(CM_B)

disp = ConfusionMatrixDisplay(confusion_matrix=CM_B, display_labels=My_BNB_Model.classes_)
ax[1].set_title("Bernoulli Naive Bayes")
disp.plot(ax=ax[1])

plt.tight_layout()
plt.show()

## Train/Test Split - Decision Tree
Training_DT, Testing_DT = train_test_split(
    df,
    test_size=0.3,
    stratify=df["diagnosis_result"],
    random_state=42
)

Training_DT_Label = Training_DT["diagnosis_result"]
Training_DT = Training_DT.drop(["diagnosis_result"], axis=1)
Testing_DT_Label = Testing_DT["diagnosis_result"]
Testing_DT = Testing_DT.drop(["diagnosis_result"], axis=1)

# print("Decision Tree All features:")
# print("Training Data:")
# print(Training_DT)
# print("Training Labels:")
# print(Training_DT_Label)
# print("Testing Data:")
# print(Testing_DT)
# print("Testing Labels:")
# print(Testing_DT_Label)


## Fit on TRAINING data only
MyDT_Classifier = DecisionTreeClassifier(max_depth=3)
MyDT_Classifier = MyDT_Classifier.fit(Training_DT, Training_DT_Label)

FeatureNames = Training_DT.columns.values
ClassNames = MyDT_Classifier.classes_

## Plot the Tree
plt.figure(figsize=(40, 20))
tree.plot_tree(MyDT_Classifier, feature_names=FeatureNames, class_names=ClassNames, filled=True)
plt.savefig("mod6mod7_decisiontree.jpg", dpi=300, bbox_inches='tight')
plt.close()

## Predict on TESTING data
Prediction = MyDT_Classifier.predict(Testing_DT)
print("Decision Tree Predictions:", Prediction)

My_Conf_Mat = confusion_matrix(Testing_DT_Label, Prediction)
print("Decision Tree Confusion Matrix:")
print(My_Conf_Mat)

## Seaborn Heatmap
sns.heatmap(My_Conf_Mat, annot=True, cmap='Blues',
            xticklabels=ClassNames, yticklabels=ClassNames, cbar=False)
plt.title("Confusion Matrix For Decision Tree Classification", fontsize=12)
plt.xlabel("Predicted", fontsize=15)
plt.ylabel("Actual", fontsize=15)
plt.savefig("seaborn_plot.jpg")
plt.close()

## Sklearn Confusion Matrix Display
CM_disp = ConfusionMatrixDisplay(confusion_matrix=My_Conf_Mat, display_labels=ClassNames)
CM_disp.plot()
plt.savefig("CM_mod6mod7.jpg")
plt.close()

## -----------------------------------------------
## Decision Tree 1 - Shape Features
## radius, perimeter, area, fractal_dimension
## -----------------------------------------------

shape_features = ["radius", "perimeter", "area", "fractal_dimension", "diagnosis_result"]
df_shape = df[shape_features].copy()

Training_DT1, Testing_DT1 = train_test_split(
    df_shape,
    test_size=0.3,
    stratify=df_shape["diagnosis_result"],
    random_state=42
)

Training_DT1_Label = Training_DT1["diagnosis_result"]
Training_DT1 = Training_DT1.drop(["diagnosis_result"], axis=1)
Testing_DT1_Label = Testing_DT1["diagnosis_result"]
Testing_DT1 = Testing_DT1.drop(["diagnosis_result"], axis=1)

# print("Decision Tree 1 - Shape Features:")
# print("Training Data:")
# print(Training_DT1)
# print("Training Labels:")
# print(Training_DT1_Label)
# print("Testing Data:")
# print(Testing_DT1)
# print("Testing Labels:")
# print(Testing_DT1_Label)

MyDT1 = DecisionTreeClassifier(max_depth=3)
MyDT1.fit(Training_DT1, Training_DT1_Label)

plt.figure(figsize=(40, 20))
tree.plot_tree(MyDT1, feature_names=Training_DT1.columns.values, class_names=MyDT1.classes_, filled=True)
plt.title("Decision Tree 1 - Shape Features")
plt.savefig("MyTree_Shape.jpg", dpi=300, bbox_inches='tight')
plt.close()

Prediction_DT1 = MyDT1.predict(Testing_DT1)
CM_DT1 = confusion_matrix(Testing_DT1_Label, Prediction_DT1)
print("Decision Tree 1 (Shape) Confusion Matrix:")
print(CM_DT1)

sns.heatmap(CM_DT1, annot=True, cmap='Blues',
            xticklabels=MyDT1.classes_, yticklabels=MyDT1.classes_, cbar=False)
plt.title("Decision Tree 1 - Shape Features", fontsize=12)
plt.xlabel("Predicted", fontsize=15)
plt.ylabel("Actual", fontsize=15)
plt.savefig("CM_Shape.jpg")
plt.close()

## -----------------------------------------------
## Decision Tree 2 - Texture/Smoothness Features
## texture, smoothness, compactness, symmetry
## -----------------------------------------------

texture_features = ["texture", "smoothness", "compactness", "symmetry", "diagnosis_result"]
df_texture = df[texture_features].copy()

Training_DT2, Testing_DT2 = train_test_split(
    df_texture,
    test_size=0.3,
    stratify=df_texture["diagnosis_result"],
    random_state=42
)

Training_DT2_Label = Training_DT2["diagnosis_result"]
Training_DT2 = Training_DT2.drop(["diagnosis_result"], axis=1)
Testing_DT2_Label = Testing_DT2["diagnosis_result"]
Testing_DT2 = Testing_DT2.drop(["diagnosis_result"], axis=1)

# print("Decision Tree 2 - Texture/Smoothness Features:")
# print("Training Data:")
# print(Training_DT2)
# print("Training Labels:")
# print(Training_DT2_Label)
# print("Testing Data:")
# print(Testing_DT2)
# print("Testing Labels:")
# print(Testing_DT2_Label)


MyDT2 = DecisionTreeClassifier(max_depth=3)
MyDT2.fit(Training_DT2, Training_DT2_Label)

plt.figure(figsize=(40, 20))
tree.plot_tree(MyDT2, feature_names=Training_DT2.columns.values, class_names=MyDT2.classes_, filled=True)
plt.title("Decision Tree 2 - Texture/Smoothness Features")
plt.savefig("MyTree_Texture.jpg", dpi=300, bbox_inches='tight')
plt.close()

Prediction_DT2 = MyDT2.predict(Testing_DT2)
CM_DT2 = confusion_matrix(Testing_DT2_Label, Prediction_DT2)
print("Decision Tree 2 (Texture/Smoothness) Confusion Matrix:")
print(CM_DT2)

sns.heatmap(CM_DT2, annot=True, cmap='Blues',
            xticklabels=MyDT2.classes_, yticklabels=MyDT2.classes_, cbar=False)
plt.title("Decision Tree 2 - Texture/Smoothness Features", fontsize=12)
plt.xlabel("Predicted", fontsize=15)
plt.ylabel("Actual", fontsize=15)
plt.savefig("CM_Texture.jpg")
plt.close()


## -----------------------------------------------
## Gaussian NB - Predicted Probabilities
## -----------------------------------------------

## Get probabilities
GNB_Probs = My_GNB_Model.predict_proba(Testing_G)

## Create a dataframe matching probabilities to actual labels
GNB_Prob_DF = pd.DataFrame({
    "Actual"         : Testing_G_Label.values,
    "Predicted"      : Predictions_G,
    "Prob_Benign"    : GNB_Probs[:, 0].round(3),
    "Prob_Malignant" : GNB_Probs[:, 1].round(3),
    "Correct"        : Testing_G_Label.values == Predictions_G
})
print("Gaussian NB - Predicted Probabilities vs Actual:")
print(GNB_Prob_DF)

## Plot the probabilities
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

## Benign probability
ax[0].bar(range(len(GNB_Prob_DF)), GNB_Prob_DF["Prob_Benign"],
          color=GNB_Prob_DF["Correct"].map({True: "steelblue", False: "red"}))
ax[0].axhline(y=0.5, color="black", linestyle="--", label="50% threshold")
ax[0].set_title("Gaussian NB - Probability of Benign")
ax[0].set_xlabel("Test Sample")
ax[0].set_ylabel("Probability")
ax[0].legend()

## Malignant probability
ax[1].bar(range(len(GNB_Prob_DF)), GNB_Prob_DF["Prob_Malignant"],
          color=GNB_Prob_DF["Correct"].map({True: "steelblue", False: "red"}))
ax[1].axhline(y=0.5, color="black", linestyle="--", label="50% threshold")
ax[1].set_title("Gaussian NB - Probability of Malignant")
ax[1].set_xlabel("Test Sample")
ax[1].set_ylabel("Probability")
ax[1].legend()

plt.suptitle("Gaussian NB Predicted Probabilities\n(Blue = Correct, Red = Misclassified)", fontsize=13)
plt.tight_layout()
plt.savefig("GNB_Probabilities.jpg", dpi=300, bbox_inches="tight")
plt.show()

## -----------------------------------------------
## Bernoulli NB - Predicted Probabilities
## -----------------------------------------------

## Get probabilities
BNB_Probs = My_BNB_Model.predict_proba(Testing_B)

## Create a dataframe matching probabilities to actual labels
BNB_Prob_DF = pd.DataFrame({
    "Actual"         : Testing_B_Label.values,
    "Predicted"      : Predictions_B,
    "Prob_Benign"    : BNB_Probs[:, 0].round(3),
    "Prob_Malignant" : BNB_Probs[:, 1].round(3),
    "Correct"        : Testing_B_Label.values == Predictions_B
})
print("Bernoulli NB - Predicted Probabilities vs Actual:")
print(BNB_Prob_DF)

## Plot the probabilities
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

## Benign probability
ax[0].bar(range(len(BNB_Prob_DF)), BNB_Prob_DF["Prob_Benign"],
          color=BNB_Prob_DF["Correct"].map({True: "steelblue", False: "red"}))
ax[0].axhline(y=0.5, color="black", linestyle="--", label="50% threshold")
ax[0].set_title("Bernoulli NB - Probability of Benign")
ax[0].set_xlabel("Test Sample")
ax[0].set_ylabel("Probability")
ax[0].legend()

## Malignant probability
ax[1].bar(range(len(BNB_Prob_DF)), BNB_Prob_DF["Prob_Malignant"],
          color=BNB_Prob_DF["Correct"].map({True: "steelblue", False: "red"}))
ax[1].axhline(y=0.5, color="black", linestyle="--", label="50% threshold")
ax[1].set_title("Bernoulli NB - Probability of Malignant")
ax[1].set_xlabel("Test Sample")
ax[1].set_ylabel("Probability")
ax[1].legend()

plt.suptitle("Bernoulli NB Predicted Probabilities\n(Blue = Correct, Red = Misclassified)", fontsize=13)
plt.tight_layout()
plt.savefig("BNB_Probabilities.jpg", dpi=300, bbox_inches="tight")
plt.show()

## -----------------------------------------------
## Gaussian NB - Predicted Probabilities (Improved)
## -----------------------------------------------

GNB_Probs = My_GNB_Model.predict_proba(Testing_G)

GNB_Prob_DF = pd.DataFrame({
    "Actual"         : Testing_G_Label.values,
    "Predicted"      : Predictions_G,
    "Prob_Benign"    : GNB_Probs[:, 0].round(3),
    "Prob_Malignant" : GNB_Probs[:, 1].round(3),
    "Correct"        : Testing_G_Label.values == Predictions_G
})
print("Gaussian NB - Predicted Probabilities vs Actual:")
print(GNB_Prob_DF)

## Separate correct and incorrect predictions
correct   = GNB_Prob_DF[GNB_Prob_DF["Correct"] == True]
incorrect = GNB_Prob_DF[GNB_Prob_DF["Correct"] == False]

plt.figure(figsize=(12, 6))

## Plot correct predictions as dots
plt.scatter(correct.index, correct["Prob_Malignant"],
            color="steelblue", label="Correct", zorder=3, s=80)

## Plot incorrect predictions as large red X marks
plt.scatter(incorrect.index, incorrect["Prob_Malignant"],
            color="red", label="Misclassified", marker="X", zorder=4, s=200)

## Add actual label as text above each point
for i, row in GNB_Prob_DF.iterrows():
    plt.text(i, row["Prob_Malignant"] + 0.02, row["Actual"],
             ha="center", fontsize=8, color="black")

plt.axhline(y=0.5, color="black", linestyle="--", label="50% Decision Boundary")
plt.title("Gaussian NB - Probability of Malignant\n(X = Misclassified, label = Actual diagnosis)", fontsize=13)
plt.xlabel("Test Sample Index")
plt.ylabel("Probability of Malignant")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig("GNB_Probabilities.jpg", dpi=300, bbox_inches="tight")
plt.show()

## -----------------------------------------------
## Bernoulli NB - Predicted Probabilities (Improved)
## -----------------------------------------------

BNB_Probs = My_BNB_Model.predict_proba(Testing_B)

BNB_Prob_DF = pd.DataFrame({
    "Actual"         : Testing_B_Label.values,
    "Predicted"      : Predictions_B,
    "Prob_Benign"    : BNB_Probs[:, 0].round(3),
    "Prob_Malignant" : BNB_Probs[:, 1].round(3),
    "Correct"        : Testing_B_Label.values == Predictions_B
})
print("Bernoulli NB - Predicted Probabilities vs Actual:")
print(BNB_Prob_DF)

correct   = BNB_Prob_DF[BNB_Prob_DF["Correct"] == True]
incorrect = BNB_Prob_DF[BNB_Prob_DF["Correct"] == False]

plt.figure(figsize=(12, 6))

plt.scatter(correct.index, correct["Prob_Malignant"],
            color="steelblue", label="Correct", zorder=3, s=80)

plt.scatter(incorrect.index, incorrect["Prob_Malignant"],
            color="red", label="Misclassified", marker="X", zorder=4, s=200)

for i, row in BNB_Prob_DF.iterrows():
    plt.text(i, row["Prob_Malignant"] + 0.02, row["Actual"],
             ha="center", fontsize=8, color="black")

plt.axhline(y=0.5, color="black", linestyle="--", label="50% Decision Boundary")
plt.title("Bernoulli NB - Probability of Malignant\n(X = Misclassified, label = Actual diagnosis)", fontsize=13)
plt.xlabel("Test Sample Index")
plt.ylabel("Probability of Malignant")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig("BNB_Probabilities.jpg", dpi=300, bbox_inches="tight")
plt.show()
