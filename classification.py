# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:08:09 2021

@author: Parisa
"""

# Implementing SVM and Random Forest Classification


# Support Vector Machine (SVM)

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('completetrainingfull.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:,0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and the rpinting the results
from sklearn.metrics import classification_report,confusion_matrix
print("------------------------SVM------------------------")
print("confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("results:")
print(classification_report(y_test, y_pred))

# Randon Forest Classification

# Fitting Randon Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
R_classifier = RandomForestClassifier(n_estimators=10, criterion= 'entropy', random_state=0)
R_classifier.fit(X_train, y_train)

# Predicting the Test set results
R_y_pred = R_classifier.predict(X_test)

# Making the Confusion Matrix and the printing the results
from sklearn.metrics import classification_report,confusion_matrix
print("--------------Randon Forest Classification--------------")
print("confusion matrix:")
print(confusion_matrix(y_test, R_y_pred))
print("results:")
print(classification_report(y_test, R_y_pred))
