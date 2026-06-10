#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:05:13 2026

@author: nicolerishwain
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:56:31 2026

@author: nicolerishwain
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn import tree
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from IPython.display import Image

from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
dataset = "/Users/nicolerishwain/src/CS432/ML/Prostate_Cancer.csv"

df = pd.read_csv(dataset)

print("Prostate Cancer Dataset:")
print(df)
print(df.columns)



Dataset1=pd.read_csv(dataset)
print(Dataset1)




le = LabelEncoder()
label = le.fit_transform(Dataset1['diagnosis_result'])
print(label)




Training, Testing = train_test_split(Dataset1, test_size=0.30, random_state=42)
print(Training)
print(Testing)

TrainingLabel=Training["diagnosis_result"]
TestingLabel=Testing["diagnosis_result"]
print(TrainingLabel)
print(TestingLabel)

Training=Training.drop(["diagnosis_result"], axis=1)
Testing=Testing.drop(["diagnosis_result"], axis=1)
print(Training)
print(Testing)


kernels = ['linear', 'poly', 'rbf']
cost_values = [0.1, 1, 10]

font = {
        'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 16
}
plt.rc('font', **font)

for k in kernels:
    for Cval in cost_values:
        
        print(f"\nTraining SVM with {k} kernel and C={Cval}...")
        
        # Create model
        model = SVC(C=Cval, kernel=k, degree=2)
        
        # Train
        model.fit(Training, TrainingLabel)
        classes = model.classes_
        
        # Predict
        predictions = model.predict(Testing)
        
        # Confusion Matrix
        CM = confusion_matrix(TestingLabel, predictions)
        
        # Display
        disp = ConfusionMatrixDisplay(confusion_matrix=CM,
                                      display_labels=classes)
        
        disp.plot(cmap='Blues')
        disp.ax_.set_title(
            f"Confusion Matrix - Dataset 1\nKernel: {k}, C={Cval}"
        )
        
        # Save file with kernel and C in name
        plt.savefig(f"ConfusionMatrix_Dataset1_{k}_C{Cval}.pdf",
                    format="pdf",
                    bbox_inches="tight")
        
        plt.show()


MySVM= SVC(C=100, kernel='poly', degree=4)
MySVM_model=MySVM.fit(Training, TrainingLabel)
MyPrediction=MySVM_model.predict(Testing)
classes = MySVM_model.classes_
MyCM=confusion_matrix(TestingLabel, MyPrediction)
print(MyCM)

disp = ConfusionMatrixDisplay(confusion_matrix=MyCM,
                              display_labels=classes)

disp.plot(cmap='Blues')


# =========================================================
# RANDOM FOREST
# =========================================================

print("\n================ RANDOM FOREST ================\n")

My_RF = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

My_RF.fit(Training, TrainingLabel)

RF_Predictions = My_RF.predict(Testing)

RF_CM = confusion_matrix(TestingLabel, RF_Predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=RF_CM,
    display_labels=My_RF.classes_
)

disp.plot(cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.show()


# ==============================
# Feature Importance
# ==============================
feature_imp = pd.Series(
    My_RF.feature_importances_,
    index=Training.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_imp)


# =========================================================
# VISUALIZE 3 TREES FROM RANDOM FOREST
# =========================================================

print("\nVisualizing 3 Trees from Random Forest")

RF_3 = RandomForestClassifier(
    n_estimators=3,
    max_depth=5,
    random_state=42
)

RF_3.fit(Training, TrainingLabel)

for i in range(3):
    plt.figure(figsize=(12,8))

    plot_tree(
        RF_3.estimators_[i],
        feature_names=Training.columns,
        class_names=[str(c) for c in RF_3.classes_],
        filled=True
    )

    plt.title(f"Random Forest Tree {i+1}")
    plt.show()


# =========================================================
# ADABOOST
# =========================================================

print("\n================ ADABOOST ================\n")

My_Ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

My_AdaBoost = My_Ada.fit(Training, TrainingLabel)

Ada_Predictions = My_AdaBoost.predict(Testing)

Ada_CM = confusion_matrix(TestingLabel, Ada_Predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=Ada_CM,
    display_labels=My_Ada.classes_
)

disp.plot(cmap='Blues')
plt.title("AdaBoost Confusion Matrix")
plt.show()


# =========================================================
# STACKING MODEL
# =========================================================

print("\n================ STACKING ================\n")

estimators = [

    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),

    ('svm', SVC(C=1.0, kernel='poly', degree=3, probability=True)),

    ('nb', MultinomialNB())

]

Stack_Model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

Stack_Model.fit(Training, TrainingLabel)

Stack_Predictions = Stack_Model.predict(Testing)

Stack_CM = confusion_matrix(TestingLabel, Stack_Predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=Stack_CM,
    display_labels=Stack_Model.classes_
)

disp.plot(cmap='Blues')
plt.title("Stacking Confusion Matrix")
plt.show()


# =========================================================
# Accuracy Comparison
# =========================================================

RF_accuracy = My_RF.score(Testing, TestingLabel)
Ada_accuracy = My_AdaBoost.score(Testing, TestingLabel)
Stack_accuracy = Stack_Model.score(Testing, TestingLabel)

print("\n================ MODEL ACCURACY ================\n")
print("Random Forest Accuracy:", RF_accuracy)
print("AdaBoost Accuracy:", Ada_accuracy)
print("Stacking Accuracy:", Stack_accuracy)


'''



## Split the dataset into training and testing data
Training, Testing = train_test_split(Data, test_size = 0.30)
print(Training)
print(Testing)

## Remove the Labels from Training and testing and keep them
## Get the labe and save it
TrainingLabels=Training["LABEL"]
## Remove the label
Training=Training.drop(["LABEL"], axis=1)
## Repeat for the Testing Data
TestingLabels=Testing["LABEL"]
Testing=Testing.drop(["LABEL"], axis=1)
print(TrainingLabels)
print(Training)

## Instantiate Random Forest
My_RF= RandomForestClassifier(n_estimators = 10, max_depth=5)  
## Train the model
My_RF.fit(Training, TrainingLabels)
## Use the model to make predictions on the Testing dataset
RF_Predictions=My_RF.predict(Testing)
## Use a confusion matrix to compare the predictions
## to the actual labels
My_CM=confusion_matrix(TestingLabels, RF_Predictions)
## Use the nice display to view the CM
PrettyCM=ConfusionMatrixDisplay(confusion_matrix=My_CM, display_labels=My_RF.classes_)
PrettyCM.plot()
PrettyCM.ax_.set_title("random forest")

plt.show()

## Print feature importance
print(My_RF.feature_importances_)
print(Training.columns.values)
feature_imp = pd.Series(My_RF.feature_importances_, index =Training.columns.values).sort_values(ascending = False)
print(feature_imp)
 
# ###
# ## See a few of the trees
# ###
# for i in range(3):
#     next_tree = My_RF.estimators_[i]
#     plt.figure(figsize=(12, 8))
#     plot_tree(next_tree, feature_names=Training.columns.values, 
#           class_names=My_RF.classes_, filled=True)
#     plt.show()
    
    





estimators = [
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('svm', SVC(C=1.0, kernel='poly', degree=3)), 
    ('nb', MultinomialNB())
]

My_Stacker = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)


My_Stacker=My_Stacker.fit(Training, TrainingLabels)
My_Stacker.score(Testing, TestingLabels)
Stacking_Predictions=My_Stacker.predict(Testing)

print(TestingLabels)
print(Stacking_Predictions)
## Confusion Matrix
## Use a confusion matrix to compare the predictions
## to the actual labels
MyCM=confusion_matrix(TestingLabels, Stacking_Predictions)
## Use the nice display to view the CM
PrettyCM=ConfusionMatrixDisplay(confusion_matrix=MyCM, display_labels=My_Stacker.classes_)
PrettyCM.plot()
PrettyCM.ax_.set_title("Stacking")

plt.show()





from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt



## Instantiate the model
My_Ada=AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
## Fit/Train the model on the training data and labels
My_AdaBoost=My_Ada.fit(Training, TrainingLabels)
## Use the model to predict the test data
My_Ada_Prediction=My_AdaBoost.predict(Testing)
print(My_Ada_Prediction)
## Confusion Matrix
## Use a confusion matrix to compare the predictions
## to the actual labels
MyAdaCM=confusion_matrix(TestingLabels, My_Ada_Prediction)
## Use the nice display to view the CM
PrettyCM=ConfusionMatrixDisplay(confusion_matrix=MyAdaCM, display_labels=My_Ada.classes_)
PrettyCM.plot()
PrettyCM.ax_.set_title("AdaBoost")
plt.show()
'''