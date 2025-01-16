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
import os

# Get the directory of the current script.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory.
os.chdir(script_dir)

# Importing the dataset
# Importing the dataset
# To use a different dataset, change the name of the csv file and the indexes of the columns. 
# You will need to handle missing data, encode categorical data. 
# You can also split the dataset into a training set and a test set if you want to evaluate the model performance on new observations.
# We don't need to apply feature scaling because the Random Forest Regression model is not sensitive to the scale of the features. The model is based on a tree structure that splits the dataset into different regions based on the independent variables. The model makes predictions based on the average of the dependent variable in each region. We don't need to apply feature scaling because the model will make the same predictions regardless of the scale of the features.
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and 1:-1 all the columns except the last one. We don't need the first column because the second column already contains the level.
y = dataset.iloc[:, -1].values # : means all the rows and -1 the last column

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0) # Create an object of the RandomForestRegressor class and specify the number of trees in the forest. The n_estimators parameter specifies the number of trees in the forest. The random_state parameter is used to ensure that we get the same results every time we run the code.
regressor.fit(X, y) # Fit the model to the dataset. This method trains the model on the dataset. It takes the matrix of features and the dependent variable as arguments. The model will learn the correlations between the matrix of features and the dependent variable.

# Predicting a new result
print("Random Forest Regression Prediction")
print(regressor.predict([[6.5]])) # The predict method expects a 2D array as the input. We need to pass the level as a 2D array. The result will be the predicted salary of an employee with a level of 6.5.

# Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # Create a range of values from the minimum to the maximum of X with a step of 0.01. The arange function is used to create a range of values. It takes the start, stop, and step as arguments and returns an array of evenly spaced values.
X_grid = X_grid.reshape((len(X_grid), 1)) # The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array.
plt.scatter(X, y, color = 'red') # Create a scatter plot of the dataset. The scatter method takes the x and y coordinates of the points to be plotted as arguments. The color parameter specifies the color of the points.
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') # Create a line plot of the predictions. The predict method is used to generate predictions for the input data. It takes the input features and returns the predicted values.
plt.title('Truth or Bluff (Decision Tree Regression)') # Set the title of the plot.
plt.xlabel('Position Level') # Set the x-axis label.
plt.ylabel('Salary') # Set the y-axis label.
plt.show() # Display the plot.