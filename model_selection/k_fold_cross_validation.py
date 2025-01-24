# k-Fold Cross Validation
# k-Fold Cross Validation is a common type of cross validation that is widely used in machine learning.
# k-Fold Cross Validation is a technique used to evaluate the performance of a machine learning model. 
# It is used to estimate how accurately a predictive model will perform in practice.
# The dataset is divided into k subsets. 
# The model is trained on k-1 of the subsets and tested on the remaining subset. 
# This process is repeated k times, with each of the k subsets used exactly once as the test data. 
# The k results from the folds can then be averaged to produce a single estimation of model performance.
# The advantage of k-Fold Cross Validation is that it provides a more accurate estimate of the model's performance than other methods.
# The disadvantage of k-Fold Cross Validation is that it is computationally expensive.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Get the directory of the current script.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory.
os.chdir(script_dir)

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print(accuracy_score)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) # cv is the number of folds. 
#estimator is the model we want to evaluate. it can be any classifier or regressor.
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
# The accuracy is the average of the accuracies of all the folds.
# The standard deviation is the standard deviation of the accuracies of all the folds.
# The standard deviation is a measure of how spread out the accuracies are.
# The smaller the standard deviation, the better the model.