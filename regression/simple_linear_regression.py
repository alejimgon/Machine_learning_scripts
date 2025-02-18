# Simple linear regression

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

# Splitting the dataset into the Training set and Test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # We split the dataset into 80% training set and 20% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

# Training the Simple Linear Regression model on the Training set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Create an object of the LinearRegression class.
regressor.fit(X_train, y_train) # Fit the model to the training set. This method trains the model on the training set. It takes the matrix of features and the dependent variable as arguments. The model will learn the correlations between the matrix of features and the dependent variable.

# Predicting the Test set results.
y_pred = regressor.predict(X_test) # Predict the test set results. This method takes the matrix of features of the test set as an argument and returns the predicted dependent variable values.

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print("Simple Linear Regression Evaluation")
print("R2 Score: ", r2_score(y_test, y_pred)) # The r2_score function is used to calculate the coefficient of determination. It takes the actual values and the predicted values as arguments and returns the coefficient of determination. The coefficient of determination is a measure of how well the model is performing. It ranges from 0 to 1, where 1 indicates a perfect model.

# Visualising the Training set results.
plt.scatter(X_train, y_train, color = 'red') # Create a scatter plot of the training set. The scatter method takes the x and y coordinates of the points to be plotted as arguments. The color parameter specifies the color of the points.
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Create a line plot of the training set predictions. The predict method is used to generate predictions for the input data. It takes the input features and returns the predicted values. 
plt.title('Salary vs Experience (Training set)') # Set the title of the plot.
plt.xlabel('Years of Experience') # Set the x-axis label.
plt.ylabel('Salary') # Set the y-axis label.
plt.show() # Display the plot.

# Visualising the Test set results.
plt.scatter(X_test, y_test, color = 'red') # Create a scatter plot of the test set.
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Create a line plot of the training set predictions. The predict method is used to generate predictions for the input data. It takes the input features and returns the predicted values. 
plt.title('Salary vs Experience (Test set)') # Set the title of the plot.
plt.xlabel('Years of Experience') # Set the x-axis label.
plt.ylabel('Salary') # Set the y-axis label.
plt.show() # Display the plot.

# Making a single prediction (for example the salary of an employee with 12 years of experience)
#print(regressor.predict([[12]])) # The predict method expects a 2D array as the input. We need to pass the years of experience as a 2D array. The result will be the predicted salary of an employee with 12 years of experience.

# Getting the final linear regression equation with the values of the coefficients
#print(regressor.coef_) # Get the value of the coefficient (b1 or slope coefficient).
#print(regressor.intercept_) # Get the value of the intercept (b0 or constant).

# The equation of the simple linear regression model is:
# Salary = regressor.coef * YearsExperience + regression.intercept_