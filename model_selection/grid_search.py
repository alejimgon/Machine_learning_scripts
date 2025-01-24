# Grid Search for hyperparameter tuning
# Grid search is a method to perform hyperparameter tuning in order to determine the optimal values for a given model.
# It is an exhaustive search that is performed on a specified parameter grid.

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
# The parameters dictionary contains the hyperparameters we want to tune.
# We need to specify the hyperparameters we want to tune and the values we want to try.
# C, kernel, and gamma are the hyperparameters of the SVC model.
grid_search = GridSearchCV(estimator = classifier, # The model we want to evaluate.
                           param_grid = parameters, # The grid of parameters we want to search.
                           scoring = 'accuracy', # The metric we want to optimize.
                           cv = 10, # The number of folds.
                           n_jobs = -1) # The number of CPUs to use to do the computation. -1 means all CPUs.
grid_search.fit(X_train, y_train) # Fit the grid search to the training set.
best_accuracy = grid_search.best_score_ # Get the best accuracy.
best_parameters = grid_search.best_params_ # Get the best parameters.
print("Best Accuracy: {:.2f} %".format(best_accuracy*100)) 
print("Best Parameters:", best_parameters)