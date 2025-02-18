# Polynomial Regression
# This script needs to be modified if the dataset has missing data or categorical data.

# Importing the libraries
import numpy as np # Allows us to work with arrays
import matplotlib.pyplot as plt # Allows us to plot charts
import pandas as pd # Allows us to import datasets and create the matrix of features and dependent variable

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Importing the dataset
dataset = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv')
X = dataset.iloc[:, :-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and :-1 all the columns except the last one.
y = dataset.iloc[:, -1].values # : means all the rows and -1 the last column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # We split the dataset into 80% training set and 20% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

# Training the Polynomial Regression model on the Training set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4) # Create an object of the PolynomialFeatures class and specify the degree of the polynomial. The degree parameter specifies the degree of the polynomial. The higher the degree, the more complex the model will be.
X_poly = poly_reg.fit_transform(X_train) # Fit and transform the matrix of features. The fit_transform method is used to transform the matrix of features. It takes the matrix of features as an argument and returns the transformed matrix of features.
regressor = LinearRegression() # Create an object of the LinearRegression class. This class will be used to fit the polynomial regression model.
regressor.fit(X_poly, y_train) # Fit the model to the dataset. This method trains the model on the dataset. It takes the matrix of features and the dependent variable as arguments. The model will learn the correlations between the matrix of features and the dependent variable.

# Predicting the Test set results
y_pred = regressor.predict(poly_reg.transform(X_test)) # Predict the test set results. This method takes the matrix of features of the test set as an argument and returns the predicted dependent variable values.
np.set_printoptions(precision=2) # Set the number of decimal places to 2.
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # Concatenate the predicted values and the actual values. The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array. The 1 argument specifies the axis along which the arrays will be joined.

# Evaluating the Polynomial Regression model
from sklearn.metrics import r2_score
print("Polynomial Regression Evaluation")
print("R2 Score: ", r2_score(y_test, y_pred)) # The r2_score function is used to calculate the coefficient of determination. It takes the actual values and the predicted values as arguments and returns the coefficient of determination. The coefficient of determination is a measure of how well the model is performing. It ranges from 0 to 1, where 1 indicates a perfect model.
