# Support Vector Regression

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
y = y.reshape(len(y),1) # We need to reshape the dependent variable because the StandardScaler class expects a 2D array as the input. The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array. 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # We split the dataset into 80% training set and 20% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

# Feature Scaling
# Support Vector Regression does not have a built-in feature scaling method. We need to apply feature scaling to the dataset before training the model. 
# When the dependent variable takes high values with respect to the other features, we need to apply feature scaling to put all the features and the dependent variable on the same scale.
# We need to apply feature scaling to the dependent variable as well because the model will be sensitive to the scale of the dependent variable and the feature might be neglected. 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # Create an object of the StandardScaler class.
sc_y = StandardScaler() # Create an object of the StandardScaler class.
X_train = sc_X.fit_transform(X_train) # Fit and transform the matrix of features. The fit_transform method is used to transform the matrix of features. It takes the matrix of features as an argument and returns the transformed matrix of features.
y_train = sc_y.fit_transform(y_train) # Fit and transform the dependent variable. We don't need to have a numpy array here because the dependent variable is always going to be a vector.

# Training the Support Vector Regression model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Create an object of the SVR class and specify the kernel. The kernel parameter specifies the kernel type to be used in the algorithm. The rbf kernel is used for non-linear regression.
regressor.fit(X_train, y_train) # Fit the model to the dataset. This method trains the model on the dataset. It takes the matrix of features and the dependent variable as arguments. The model will learn the correlations between the matrix of features and the dependent variable.

# Predicting the Test set results
# We need to apply feature scaling to the new data because the model was trained on the scaled data. We need to transform the new data using the same StandardScaler object that was used to train the model.
# We also need to inverse the scaling to get the original scale of the dependent variable.
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1,1)) # Predict the test set results. This method takes the matrix of features of the test set as an argument and returns the predicted dependent variable values. The reshape method is used to change the shape of the array (from horizontal to vertical).
np.set_printoptions(precision=2) # Set the number of decimal places to 2.
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # Concatenate the predicted values and the actual values. The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array. The 1 argument specifies the axis along which the arrays will be joined.

# Evaluating the Model Performance
from sklearn.metrics import r2_score
print("Support Vector Regression Evaluation")
print("R2 Score: ", r2_score(y_test, y_pred)) # The r2_score function is used to calculate the coefficient of determination. It takes the actual values and the predicted values as arguments and returns the coefficient of determination. The coefficient of determination is a measure of how well the model is performing. It ranges from 0 to 1, where 1 indicates a perfect model.
