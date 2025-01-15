# Polynomial Regression

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
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and 1:-1 all the columns except the last one. We don't need the first column because the second column already contains the level.
y = dataset.iloc[:, -1].values # : means all the rows and -1 the last column

# We don't need to split the dataset into a training set and a test set because we want to make the most accurate predictions possible. We need all the data we can get to train the model.
# We will compare the Linear Regression model with the Polynomial Regression model. 
# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # Create an object of the LinearRegression class.
lin_reg.fit(X, y) # Fit the model to the dataset. This method trains the model on the dataset. It takes the matrix of features and the dependent variable as arguments. The model will learn the correlations between the matrix of features and the dependent variable.

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # Create an object of the PolynomialFeatures class and specify the degree of the polynomial. The degree parameter specifies the degree of the polynomial. The higher the degree, the more complex the model will be.
X_poly = poly_reg.fit_transform(X) # Fit and transform the matrix of features. The fit_transform method is used to transform the matrix of features. It takes the matrix of features as an argument and returns the transformed matrix of features.
lin_reg_2 = LinearRegression() # Create an object of the LinearRegression class. This class will be used to fit the polynomial regression model.
lin_reg_2.fit(X_poly, y) # Fit the model to the dataset. This method trains the model on the dataset. It takes the matrix of features and the dependent variable as arguments. The model will learn the correlations between the matrix of features and the dependent variable.

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red') # Create a scatter plot of the dataset. The scatter method takes the x and y coordinates of the points to be plotted as arguments. The color parameter specifies the color of the points.
plt.plot(X, lin_reg.predict(X), color = 'blue') # Create a line plot of the predictions. The predict method is used to generate predictions for the input data. It takes the input features and returns the predicted values.
plt.title('Truth or Bluff (Linear Regression)') # Set the title of the plot.
plt.xlabel('Position Level') # Set the x-axis label.
plt.ylabel('Salary') # Set the y-axis label.
plt.show() # Display the plot.

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red') # Create a scatter plot of the dataset. The scatter method takes the x and y coordinates of the points to be plotted as arguments. The color parameter specifies the color of the points.
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue') # Create a line plot of the predictions. The predict method is used to generate predictions for the input data. It takes the input features and returns the predicted values.
plt.title('Truth or Bluff (Polynomial Regression)') # Set the title of the plot.
plt.xlabel('Position Level') # Set the x-axis label.
plt.ylabel('Salary') # Set the y-axis label.
plt.show() # Display the plot.

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1) # Create a range of values from the minimum to the maximum of X with a step of 0.1. The arange function is used to create a range of values. It takes the start, stop, and step as arguments and returns an array of evenly spaced values.
X_grid = X_grid.reshape((len(X_grid), 1)) # The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array.
X_poly_grid = poly_reg.fit_transform(X_grid) # Fit and transform the matrix of features. The fit_transform method is used to transform the matrix of features. It takes the matrix of features as an argument and returns the transformed matrix of features.
plt.scatter(X, y, color = 'red') # Create a scatter plot of the dataset. The scatter method takes the x and y coordinates of the points to be plotted as arguments. The color parameter specifies the color of the points.
plt.plot(X_grid, lin_reg_2.predict(X_poly_grid), color = 'blue') # Create a line plot of the predictions. The predict method is used to generate predictions for the input data. It takes the input features and returns the predicted values.
plt.title('Truth or Bluff (Polynomial Regression)') # Set the title of the plot.
plt.xlabel('Position Level') # Set the x-axis label.
plt.ylabel('Salary') # Set the y-axis label.
plt.show() # Display the plot.

# Predicting a new result with Linear Regression
print("Linear Regression Prediction")
print(lin_reg.predict([[6.5]])) # The predict method expects a 2D array as the input. We need to pass the level as a 2D array. The result will be the predicted salary of an employee with a level of 6.5.

# Predicting a new result with Polynomial Regression
print("Polynomial Regression Prediction")
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))) # The predict method expects a 2D array as the input. We need to pass the level as a 2D array. The result will be the predicted salary of an employee with a level of 6.5. The fit_transform method is used to transform the matrix of features. It takes the matrix of features as an argument and returns the transformed matrix of features.

# Evaluating the Linear Regression model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
y_pred = lin_reg.predict(X) # Predict the dependent variable values. This method takes the matrix of features as an argument and returns the predicted dependent variable values.
print("Linear Regression Evaluation")
print("Mean Squared Error: ", mean_squared_error(y, y_pred)) # The mean_squared_error function is used to calculate the mean squared error. It takes the actual values and the predicted values as arguments and returns the mean squared error. The mean squared error is a measure of how well the model is performing. It ranges from 0 to infinity, where 0 indicates a perfect model.
print("R2 Score: ", r2_score(y, y_pred)) # The r2_score function is used to calculate the coefficient of determination. It takes the actual values and the predicted values as arguments and returns the coefficient of determination. The coefficient of determination is a measure of how well the model is performing. It ranges from 0 to 1, where 1 indicates a perfect model.

# Evaluating the Polynomial Regression model
y_pred = lin_reg_2.predict(X_poly) # Predict the dependent variable values. This method takes the matrix of features as an argument and returns the predicted dependent variable values.
print("Polynomial Regression Evaluation")
print("Mean Squared Error: ", mean_squared_error(y, y_pred)) # The mean_squared_error function is used to calculate the mean squared error. It takes the actual values and the predicted values as arguments and returns the mean squared error. The mean squared error is a measure of how well the model is performing. It ranges from 0 to infinity, where 0 indicates a perfect model.
print("R2 Score: ", r2_score(y, y_pred)) # The r2_score function is used to calculate the coefficient of determination. It takes the actual values and the predicted values as arguments and returns the coefficient of determination. The coefficient of determination is a measure of how well the model is performing. It ranges from 0 to 1, where 1 indicates a perfect model.
