# XGBoost model
# XBBoost is a powerful and efficient open-source library for gradient boosting. 
# It is a machine learning algorithm that is based on decision trees. 
# It is used for supervised learning problems, where we have a training dataset (X, y) and we want to predict the target variable y for a new set of data. 
# It is a scalable and accurate implementation of gradient boosting machines and has proven to be one of the most effective machine learning algorithms. 
# It is widely used in machine learning competitions and is known for its speed and performance. 
# It is an ensemble learning method that combines the predictions of several base models to improve the overall performance.

import matplotlib.pyplot as plt # Allows us to plot charts
import pandas as pd # Allows us to import datasets and create the matrix of features and dependent variable
import numpy as np
import os

# Get the directory of the current script. 
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory.
os.chdir(script_dir) 

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Check the unique values in the target variable
unique_classes = np.unique(y)
print(f"Unique classes in the target variable: {unique_classes}")

# Encode the target variable if necessary
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Check the unique values after encoding
unique_classes = np.unique(y)
print(f"Unique classes after encoding: {unique_classes}")

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))