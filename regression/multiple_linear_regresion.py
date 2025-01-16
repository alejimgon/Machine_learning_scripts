# Multiple Linear Regression
# This script needs to be modified if the dataset has missing data or categorical data.

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

# Encoding categorical data (creating dummy variables). Use if applicable.
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') # The OneHotEncoder class is used to encode categorical data. The ColumnTransformer class is used to apply the OneHotEncoder to the dataset. The transformers argument specifies the encoder to use and the columns to encode. The remainder argument specifies what to do with the columns that are not transformed.
#X = np.array(ct.fit_transform(X)) # The fit_transform method is used to transform the dataset. It takes the matrix of features as an argument and returns the transformed matrix of features.

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # We split the dataset into 80% training set and 20% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

# We don't need to apply feature scaling to multiple linear regression because the coefficients of the independent variables will compensate for the different scales.

# Training the Multiple Linear Regression model on the Training set
# We don't need to worry about the dummy variable trap because the class (LinearRegression) takes care of it.
# We don't need to worry about the backward elimination because the class (LinearRegression) takes care of it.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Create an object of the LinearRegression class.
regressor.fit(X_train, y_train) # Fit the model to the training set. This method trains the model on the training set. It takes the matrix of features and the dependent variable as arguments. The model will learn the correlations between the matrix of features and the dependent variable.

# Predicting the Test set results
y_pred = regressor.predict(X_test) # Predict the test set results. This method takes the matrix of features of the test set as an argument and returns the predicted dependent variable values.
np.set_printoptions(precision=2) # Set the number of decimal places to 2.
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # Concatenate the predicted values and the actual values. The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array. The 1 argument specifies the axis along which the arrays will be joined.

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print("Multiple Linear Regression Evaluation")
print("R2 Score: ", r2_score(y_test, y_pred)) # The r2_score function is used to calculate the coefficient of determination. It takes the actual values and the predicted values as arguments and returns the coefficient of determination. The coefficient of determination is a measure of how well the model is performing. It ranges from 0 to 1, where 1 indicates a perfect model.

# Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = California)
#print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]])) # The predict method expects a 2D array as the input. We need to pass the values of the independent variables as a 2D array. The result will be the predicted profit of a startup with the specified values of the independent variables.

# Getting the final linear regression equation with the values of the coefficients
#print(regressor.coef_) # Get the value of the coefficients.
#print(regressor.intercept_) # Get the value of the intercept.
