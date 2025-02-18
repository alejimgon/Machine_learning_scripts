# Recurrent Neural Network
# RNN is a type of artificial neural network that is designed to model sequential data.
# RNN is particularly useful for tasks such as natural language processing, speech recognition, and time series prediction.
# RNN has a feedback loop that allows information to persist over time, making it suitable for modeling sequences of data.
# RNN can be used to model sequences of data with variable lengths, such as sentences, audio signals, and time series data.
# RNN has a hidden state that captures the context of the input sequence, allowing it to make predictions based on the entire sequence.
# RNN can be trained using backpropagation through time, which is a variant of the backpropagation algorithm that takes into account the temporal dependencies in the data.
# RNN can suffer from the vanishing gradient problem, which makes it difficult to learn long-range dependencies in the data.
# LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are two types of RNN architectures that are designed to address the vanishing gradient problem.
# LSTM networks have a more complex architecture than standard RNNs, with additional gates that control the flow of information in the network.

# Importing the libraries
import numpy as np # For numerical computations
import matplotlib.pyplot as plt # For plotting the data
import pandas as pd # For data manipulation and analysis
import math # For mathematical operations
from sklearn.metrics import mean_squared_error # For calculating the RMSE
from sklearn.preprocessing import MinMaxScaler # For noralizing the data.
from tensorflow.keras.models import Sequential # type: ignore # For initializing the neural network
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input # type: ignore # For adding the layers, LSTM, and dropout regularization

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Part 1 - Data Preprocessing
## Importing the training set
dataset_train = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv')
training_set = dataset_train.iloc[:, 1:2].values # ilock stands for locate indexes. [rows, columns] : means all the rows and 1:2 means the second column only.

## Feature Scaling
# For RNN, it is recommended to apply normalization to the data. 
# Whenever you build an RNN and especially if there is a sigmoid function as the activation function in the output layer, it is recommended to apply normalization to the data.
sc = MinMaxScaler(feature_range = (0, 1)) # Create an object of the MinMaxScaler class. The feature_range parameter specifies the range of the scaled data.
training_set_scaled = sc.fit_transform(training_set) # Fit and transform

## Creating a data structure with 60 timesteps and 1 output
# 60 timesteps means that at each time T, the RNN is going to look at the 60 data points before time T, and based on the trends, it is capturing during these 60 previous time steps, it will try to predict the next output.
# The number of timesteps is a hyperparameter that you can tune.
X_train = [] # A list that will contain the input sequences.
y_train = [] # A list that will contain the output sequences.
for i in range(60, len(training_set_scaled)): # Loop over the training set. The first 60 data points are not considered because we need 60 previous data points to predict the next one.
    X_train.append(training_set_scaled[i-60:i, 0]) # Append the last 60 data points to the input list.
    y_train.append(training_set_scaled[i, 0]) # Append the next data point to the output list.
X_train, y_train = np.array(X_train), np.array(y_train) # Convert the lists to numpy arrays.

## Reshaping
# Reshape the data to add a new dimension. 
# The new dimension corresponds to the number of predictors.
# This predictors are the indicators that can help predict the output.
# Thanks to this new data structure, we can add more indicators to the input data.
# The new shape will be (number of samples, number of timesteps, number of features)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2])) # The new dimension is the number of features
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # The new dimension is 1 because we only have one indicator.

# Part 2 - Building the RNN
## Initializing the RNN
regressor = Sequential() # Create an object of the Sequential class.

## Adding the Input layer
# The input layer requires the following parameters:
# shape: the shape of the input data.
regressor.add(Input(shape=(X_train.shape[1], X_train.shape[2]))) # Add the input layer.

## Adding the second LSTM layer and some Dropout regularization
# The second LSTM layer requires the following parameters:
# units: the number of LSTM cells or units in the LSTM layer.
# return_sequences: set to True because we will add more LSTM layers.
regressor.add(LSTM(units = 50, return_sequences = True)) # Add the second LSTM layer
regressor.add(Dropout(0.2)) # Add some dropout regularization to avoid overfitting.

## Adding the third LSTM layer and some Dropout regularization
# The third LSTM layer requires the following parameters:
# units: the number of LSTM cells or units in the LSTM layer.
# return_sequences: set to True because we will add more LSTM layers.
regressor.add(LSTM(units = 50, return_sequences = True)) # Add the third LSTM layer
regressor.add(Dropout(0.2)) # Add some dropout regularization to avoid overfitting.

## Adding the fourth LSTM layer and some Dropout regularization
# The fourth LSTM layer requires the following parameters:
# units: the number of LSTM cells or units in the LSTM layer.
# return_sequences: set to False because we will not add more LSTM layers.
regressor.add(LSTM(units = 50)) # Add the fourth LSTM layer
regressor.add(Dropout(0.2)) # Add some dropout regularization to avoid overfitting.

## Adding the output layer
# The output layer requires the following parameters:
# units: the number of neurons in the output layer.
regressor.add(Dense(units = 1)) # Add the output layer.

## Compiling the RNN
# The compile method requires the following parameters:
# optimizer: the optimization algorithm to be used to update the weights. 
# There are several optimization algorithms available, such as Adam, RMSprop, and SGD.
# Adam is the most popular optimization algorithm used in deep learning.
# RMSprop is useful when we have sparse data.
# SGD is the simplest optimization algorithm.
# loss: the loss function to be used to calculate the error. 
# There are several loss functions available, such as mean_squared_error, binary_crossentropy, and categorical_crossentropy.
# mean_squared_error is used for regression problems.
# binary_crossentropy is used for binary classification problems.
# categorical_crossentropy is used for non-binary classification problems.
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # Compile the RNN.

## Fitting the RNN to the Training set
# The fit method requires the following parameters:
# X_train: the input data.
# y_train: the output data.
# batch_size: the number of samples to be used in each batch.
# epochs: the number of times the entire training set is passed through the RNN.
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) # Fit the RNN to the training set.

# Part 3 - Making the predictions and visualizing the results
## Importing the test set
dataset_test = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values # ilock stands for locate indexes. [rows, columns] : means all the rows and 1:2 means the second column only.

## Getting the predicted stock price of 2017
# We need to concatenate the training set and the test set because we need the 60 previous data points to predict the next one.
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # Concatenate the training and test sets.
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # Get the inputs.
inputs = inputs.reshape(-1,1) # Reshape the inputs.
inputs = sc.transform(inputs) # Apply feature scaling.
X_test = [] # A list that will contain the input sequences.
for i in range(60, 80): # Loop over the inputs. The upper bound is dependent on the number of data points in the test set.
    X_test.append(inputs[i-60:i, 0]) # Append the last 60 data points to the input list.
X_test = np.array(X_test) # Convert the list to a numpy array.
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # Reshape the data.
predicted_stock_price = regressor.predict(X_test) # Predict the stock price.
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # Inverse the scaling.

## Calculating the RMSE
# The RMSE is a measure of the difference between the predicted values and the real values.
# The lower the RMSE, the better the model.
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(f'Root Mean Squared Error: {rmse}')

## Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price') # Plot the real stock price.
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price') # Plot the predicted stock price.
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show() # Show the plot.