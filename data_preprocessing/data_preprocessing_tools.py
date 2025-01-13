# Data Preprocessing Tools

# Importing the libraries
import numpy as np # Allows us to work with arrays
import matplotlib.pyplot as plt # Allows us to plot charts
import pandas as pd # Allows us to import datasets and create the matrix of features and dependent variable
import os

# Get the directory of the current script. 
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory.
os.chdir(script_dir) 

# Importing the dataset. 
# The way we are going to create our machine learning models expect exactly X and y entities in their input.
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and :-1 all the columns except the last one.
y = dataset.iloc[:, -1].values # : means all the rows and -1 the last column
#print(X)
#print(y)

# Taking care of missing data. 
# One way is to ignore the missing data, another way is to replace the missing data with the average of the column.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # Create an object of the SimpleImputer class and replace the missing values with the mean of the column.
imputer.fit(X[:, 1:3]) # This method calculates the necessary statistics (e.g., mean, median, mode) required to fill in missing values in the specified columns.
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Replace the missing data with the mean of the column. The transform method returns a new array with the missing values filled in.
#print(X)

# Encoding categorical data (creation of dummy variable). 
# We need to encode the categorical data into numbers. We can use the OneHotEncoder (create binary vectors) class to encode the independent variable and the LabelEncoder class to encode the dependent variable.

# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # Create an object of the ColumnTransformer class and specify the transformation to be applied to the specified columns.
X = np.array(ct.fit_transform(X)) # Apply the transformation to the specified columns and convert the result into a NumPy array.
#print(X)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # Create an object of the LabelEncoder class.
y = le.fit_transform(y) # Fit and transform the dependent variable. We don't need to have a numpy array here because the dependent variable is always going to be a vector.
#print(y)

# Splitting the dataset into the Training set and Test set. 
# We need to split the dataset into a training set and a test set. We will train the model on the training set and test the model on the test set.
# We have to apply feature scaling after splitting the dataset because we don't want to fit the test set to the training set. We want to fit the training set to the training set and the test set to the test set. This will prevent information leakage.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1) # We split the dataset into 80% training set and 20% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

# Feature Scaling
# Feature scaling is a method used to standardize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.
# We need to apply feature scaling to the independent variables because we want to avoid the domination of one independent variable over the other. We don't want the model to be biased towards one independent variable.
# There are two common ways to apply feature scaling: Standardization and Normalization.
# Standardization: x_stand = (x - mean(x)) / standard deviation(x). It works all the time.
# Normalization: x_norm = (x - min(x)) / (max(x) - min(x)). It recommended when you have a normal distribution in most of your features.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # Create an object of the StandardScaler class.
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) # Fit and transform the training set. We don't apply feature scaling to the dummy variables. We only apply feature scaling to the numerical variables.
X_test[:, 3:] = sc.transform(X_test[:, 3:]) # We only apply the transform method to the test set. We don't need to fit the test set because the StandardScaler object is already fitted to the training set.
#print(X_train)
#print(X_test)
