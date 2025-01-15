# Support Vector Regression

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
y = y.reshape(len(y),1) # We need to reshape the dependent variable because the StandardScaler class expects a 2D array as the input. The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array. 
#print(y)

# We don't need to split the dataset into a training set and a test set because we want to make the most accurate predictions possible. We need all the data we can get to train the model.

# Feature Scaling
# Support Vector Regression does not have a built-in feature scaling method. We need to apply feature scaling to the dataset before training the model. 
# When the dependent variable takes high values with respect to the other features, we need to apply feature scaling to put all the features and the dependent variable on the same scale.
# We need to apply feature scaling to the dependent variable as well because the model will be sensitive to the scale of the dependent variable and the feature might be neglected. 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # Create an object of the StandardScaler class.
sc_y = StandardScaler() # Create an object of the StandardScaler class.
X = sc_X.fit_transform(X) # Fit and transform the matrix of features. The fit_transform method is used to transform the matrix of features. It takes the matrix of features as an argument and returns the transformed matrix of features.
y = sc_y.fit_transform(y) # Fit and transform the dependent variable. We don't need to have a numpy array here because the dependent variable is always going to be a vector.
#print(X)
#print(y)

# Training the Support Vector Regression model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Create an object of the SVR class and specify the kernel. The kernel parameter specifies the kernel type to be used in the algorithm. The rbf kernel is used for non-linear regression.
regressor.fit(X, y) # Fit the model to the dataset. This method trains the model on the dataset. It takes the matrix of features and the dependent variable as arguments. The model will learn the correlations between the matrix of features and the dependent variable.

# Predicting a new result
# We need to apply feature scaling to the new data because the model was trained on the scaled data. We need to transform the new data using the same StandardScaler object that was used to train the model.
# We also need to inverse the scaling to get the original scale of the dependent variable.
pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)) # The predict method expects a 2D array as the input. We need to pass the level as a 2D array. The result will be the predicted salary of an employee with a level of 6.5.
#print("Support Vector Regression Prediction")
#print(pred)

# Visualising the Support Vector Regression results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red') # Create a scatter plot of the dataset. The scatter method takes the x and y coordinates of the points to be plotted as arguments. The color parameter specifies the color of the points.
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue') # Create a line plot of the predictions. The predict method is used to generate predictions for the input data. It takes the input features and returns the predicted values.
plt.title('Truth or Bluff (Support Vector Regression)') # Set the title of the plot.
plt.xlabel('Position Level') # Set the x-axis label.
plt.ylabel('Salary') # Set the y-axis label.
plt.show() # Display the plot.

# Visualising the Support Vector Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1) # Create a range of values from the minimum to the maximum of X with a step of 0.1. The arange function is used to create a range of values. It takes the start, stop, and step as arguments and returns an array of evenly spaced values.
X_grid = X_grid.reshape((len(X_grid), 1)) # The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array.
y_grid = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)) # The predict method is used to generate predictions for the input data. It takes the input features and returns the predicted values.
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red') # Create a scatter plot of the dataset. The scatter method takes the x and y coordinates of the points to be plotted as arguments. The color parameter specifies the color of the points.
plt.plot(X_grid, y_grid, color = 'blue') # Create a line plot of the predictions. The predict method is used to generate predictions for the input data. It takes the input features and returns the predicted values.
plt.title('Truth or Bluff (Support Vector Regression)') # Set the title of the plot.
plt.xlabel('Position Level') # Set the x-axis label.
plt.ylabel('Salary') # Set the y-axis label.
plt.show() # Display the plot.

# Evaluating the Model Performance
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Support Vector Regression Evaluation")
print("Mean Squared Error: ", mean_squared_error(sc_y.inverse_transform(y), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)))) # The mean_squared_error function is used to calculate the mean squared error. It takes the actual values and the predicted values as arguments and returns the mean squared error. The mean squared error is a measure of how well the model is performing. It ranges from 0 to infinity, where 0 indicates a perfect model.
print("R2 Score: ", r2_score(sc_y.inverse_transform(y), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1))) ) # The r2_score function is used to calculate the coefficient of determination. It takes the actual values and the predicted values as arguments and returns the coefficient of determination. The coefficient of determination is a measure of how well the model is performing. It ranges from 0 to 1, where 1 indicates a perfect model.
