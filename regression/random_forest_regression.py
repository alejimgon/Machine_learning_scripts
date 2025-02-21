# Random Forest Regression
# The random forest regression model is an ensemble learning method that combines multiple decision trees to create a more accurate model. 
# The model is based on the random forest algorithm, which uses a technique called bagging to create multiple decision trees and combine their predictions. 
# The random forest algorithm is used for both classification and regression tasks.
# The Random Forest regression model is not the best model to use on sigle feature datasets. 
# It is more suitable for datasets with multiple features (high-dimensional dataset).

# Importing the libraries
import numpy as np # Allows us to work with arrays
import matplotlib.pyplot as plt # Allows us to plot charts
import pandas as pd # Allows us to import datasets and create the matrix of features and dependent variable

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Importing the dataset
# To use a different dataset, change the name of the csv file and the indexes of the columns. 
# You will need to handle missing data or encode categorical data. 
# We don't need to apply feature scaling because the Random Forest Regression model is not sensitive to the scale of the features. 
# The model is based on a tree structure that splits the dataset into different regions based on the independent variables. 
# The model makes predictions based on the average of the dependent variable in each region. 
# We don't need to apply feature scaling because the model will make the same predictions regardless of the scale of the features.
dataset = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv')
X = dataset.iloc[:, :-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and :-1 all the columns except the last one.
y = dataset.iloc[:, -1].values # : means all the rows and -1 the last column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # We split the dataset into 80% training set and 20% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

# Training the Random Forest Regression model on the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0) # Create an object of the RandomForestRegressor class and specify the number of trees in the forest. The n_estimators parameter specifies the number of trees in the forest. The random_state parameter is used to ensure that we get the same results every time we run the code.
regressor.fit(X_train, y_train) # Fit the model to the dataset. This method trains the model on the dataset. It takes the matrix of features and the dependent variable as arguments. The model will learn the correlations between the matrix of features and the dependent variable.

# Predicting the Test set results
y_pred = regressor.predict(X_test) # Predict the test set results. This method takes the matrix of features of the test set as an argument and returns the predicted dependent variable values.
np.set_printoptions(precision=2) # Set the number of decimal places to 2.
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # Concatenate the predicted values and the actual values. The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array. The 1 argument specifies the axis along which the arrays will be joined.

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print("Random Forest Regression Evaluation")
print("R2 Score: ", r2_score(y_test, y_pred)) # The r2_score function is used to calculate the coefficient of determination. It takes the actual values and the predicted values as arguments and returns the coefficient of determination. The coefficient of determination is a measure of how well the model is performing. It ranges from 0 to 1, where 1 indicates a perfect model.