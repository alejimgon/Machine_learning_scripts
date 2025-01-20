# Naive Bayes classifier
# Naive Bayes is a probabilistic algorithm based on the Bayes theorem. 
# It is a generative classifier that models the joint probability of the features and the class label. 
# Naive Bayes assumes that the features are conditionally independent given the class label. This is a strong assumption that is rarely true in practice.
# Naive Bayes is a simple algorithm that is easy to implement and works well with small datasets.

# Importing the libraries
import numpy as np # Allows us to work with arrays
import matplotlib.pyplot as plt # Allows us to plot charts
import pandas as pd # Allows us to import datasets and create the matrix of features and dependent variable
import os

# Get the directory of the current script.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory.
os.chdir(script_dir)

# Importing the dataset
dataset = pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.csv')
X = dataset.iloc[:, :-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and :-1 all the columns except the last one.
y = dataset.iloc[:, -1].values # : means all the rows and -1 the last column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) # We split the dataset into 75% training set and 25% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

# Feature Scaling
# We need to apply feature scaling to the independent variables because we want to avoid the domination of one independent variable over the other. We don't want the model to be biased towards one independent variable.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # Create an object of the StandardScaler class.
X_train = sc.fit_transform(X_train) # Fit and transform the training set. We need to fit the training set to the StandardScaler object and then transform it.
X_test = sc.transform(X_test) # We don't need to fit the test set to the StandardScaler object because it's already fitted to the training set.

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB() # Create an object of the GaussianNB class.
classifier.fit(X_train, y_train) # Fit the classifier to the training set.

# Predicting the Test set results and Making the Confusion Matrix
# The confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.
# The confusion matrix will show us the number of correct and incorrect predictions.
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test) # Predict the test set results.
cm = confusion_matrix(y_test, y_pred) # Create a confusion matrix to evaluate the model's performance.
print(cm)
accuracy_score = accuracy_score(y_test, y_pred) # Calculate the accuracy of the model. The accuracy in the test set is the number of correct predictions divided by the total number of predictions in the test set.
print(accuracy_score)
