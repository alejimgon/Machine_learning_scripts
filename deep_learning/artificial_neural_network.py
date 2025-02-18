# Artificial Neural Network
# ANN is a computational model that is inspired by the way biological neural networks in the human brain work.
# The basic building block of an ANN is a neuron. A neuron receives input signals, processes them, and produces an output signal.
# The input signals are weighted, summed, and passed through an activation function to produce the output signal.
# The weights are adjusted during the training process to minimize the error between the predicted output and the actual output.
# The training process involves feeding the input data through the network, comparing the predicted output with the actual output, and adjusting the weights to minimize the error.
# The process is repeated until the error is minimized, and the network can make accurate predictions.

# Importing the libraries
import numpy as np # Allows us to work with arrays
import pandas as pd # Allows us to import datasets and create the matrix of features and dependent variable
import tensorflow as tf # Allows us to build and train the artificial neural network

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Importing the dataset
dataset = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv')
X = dataset.iloc[:, 3:-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and :-1 all the columns except the last one.
y = dataset.iloc[:, -1].values # : means all the rows and -1 the last column

## Encoding categorical data
### Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # Create an object of the LabelEncoder class.
X[:, 2] = le.fit_transform(X[:, 2]) # Fit and transform the gender column. We don't need to have a numpy array here because the dependent variable is always going to be a vector.
### One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough') # Create an object of the ColumnTransformer class and specify the transformation to be applied to the specified columns.
X = np.array(ct.fit_transform(X)) # Apply the transformation to the specified columns and convert the result into a NumPy array.

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # We split the dataset into 80% training set and 20% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

## Feature Scaling
## Feature scaling is fundamental for deep learning. It is important to scale the features so that the model can learn the weights more effectively. It is applied to all the features, including the dummy variables.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # Create an object of the StandardScaler class.
X_train = sc.fit_transform(X_train) # Fit and transform the training set. We apply the fit_transform method to the training set to calculate the necessary statistics (e.g., mean, standard deviation) required to scale the features.
X_test = sc.transform(X_test) # We only apply the transform method to the test set. We don't need to fit the test set because the StandardScaler object is already fitted to the training set.

# Part 2 - Building the ANN
## Initializing the ANN as a sequence of layers
ann = tf.keras.models.Sequential() # Create an object of the Sequential class. This class allows us to build the artificial neural network as a sequence of layers.

## Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # Add a dense layer to the ANN. Dense stands for fully connected.
# The units parameter specifies the number of neurons in the layer. 
# The activation parameter specifies the activation function to be used in the layer.

## Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) 

## Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Add a dense layer to the ANN. 
# The units parameter specifies the number of neurons in the layer. 
# The activation parameter specifies the activation function to be used in the layer. 
# For binary classification problems, we use the sigmoid activation function.
# For non-binary classification problems, we use the softmax activation function.
# For regression problems, we don't use any activation function.
# For regression problems, the loss function is mean_squared_error.

# Part 3 - Training the ANN
## Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile the ANN. 
# The optimizer parameter specifies the optimization algorithm to be used to update the weights. 
# The loss parameter specifies the loss function to be used to calculate the error. 
# For non-binary classication the loss must be category_crossentropy. 
# The metrics parameter specifies the metric to be used to evaluate the model.

## Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100) # Train the ANN on the training set. 
# The batch_size parameter specifies the number of samples to be used in each batch. 
# The epochs parameter specifies the number of times the entire training set is passed through the ANN.

# Part 4 - Making the predictions and evaluating the model
## Predicting the result of a single observation
# We need to scale the features before making the prediction. 
# The predict method returns the probability of the customer leaving the bank. 
# We use the threshold of 0.5 to determine if the customer is going to leave the bank.
# print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5) 

## Predicting the Test set results
y_pred = ann.predict(X_test) # Predict the test set results. The predict method returns the probability of the customer leaving the bank. We use the threshold of 0.5 to determine if the customer is going to leave the bank.
y_pred = (y_pred > 0.5) # Convert the probabilities into True or False values.
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # Concatenate the predicted values and the actual values. 

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred) # Create the confusion matrix.
print(cm)
accuracy_score = accuracy_score(y_test, y_pred) # Calculate the accuracy of the model.
print(accuracy_score)
