# Hybrid Deep Learning Model

# Importing the libraries
import numpy as np # For numerical computations
import matplotlib.pyplot as plt # For data visualization
import pandas as pd # For data manipulation and analysis
import tensorflow as tf # For building and training the neural network
from sklearn.preprocessing import MinMaxScaler # For normalizing the data.
from sklearn.preprocessing import StandardScaler # For scaling the features
from minisom import MiniSom # For training the SOM
from pylab import bone, pcolor, colorbar, plot, show # For plotting the map

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Part 1 Identify the Anomalies with the Self-Organizing Map

# Importing the dataset
# In this example the dataset is split into X and y not because training purposes. y will be used to color the markers in the map.
# In this example, the dataset is not split into training and test sets because the SOM is an unsupervised learning algorithm.
# The SOM does not require a test set because it does not make predictions on new data.
dataset = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv')
X = dataset.iloc[:, :-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and :-1 means all columns except the last one.
y = dataset.iloc[:, -1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and -1 means the last column only.

# Feature Scaling
# Whenever you build a SOM, it is recommended to apply normalization to the data.
sc = MinMaxScaler(feature_range = (0, 1)) # Create an object of the MinMaxScaler class. The feature_range parameter specifies the range of the scaled data.
X = sc.fit_transform(X) # Fit and transform

# Training the SOM
# The SOM requires the following parameters:
# x: the number of nodes in the x-axis.
# y: the number of nodes in the y-axis.
# input_len: the number of features in the input data.
# sigma: the radius of influence of each node.
# learning_rate: the speed of learning.
# decay_function: the function that decreases the learning rate and the radius over time.
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5) # Create an object of the MiniSom class.
som.random_weights_init(X) # Initialize the weights of the SOM.
som.train_random(data=X, num_iteration=100) # Train the SOM.

# Visualizing the results
# The markers represent the data points.
# The colors represent the classes.
bone() # Create a window.
pcolor(som.distance_map().T) # Create a color map. distance_map() method returns the mean inter-neuron distances. T is the transpose of the matrix.
colorbar() # Add a color bar.
markers = ['o', 's'] # Define the markers.
colors = ['r', 'g'] # Define the colors.
for i, x in enumerate(X): # Loop over the data points.
    w = som.winner(x) # Get the winning node for the data point.
    plot(w[0] + 0.5, # Add 0.5 to center the marker.
         w[1] + 0.5, # Add 0.5 to center the marker.
         markers[y[i]], # Get the marker for the data point.
         markeredgecolor = colors[y[i]], # Get the edge color for the marker.
         markerfacecolor = 'None', # Set the face color to None.
         markersize = 10, # Set the size of the marker.
         markeredgewidth = 2) # Set the width of the marker.
show() # Show the plot.

# Identifying the anomalies
distance_map = som.distance_map().T # Get the distance map.
threshold = np.percentile(distance_map, 95) # Set a threshold for outliers based on the 95th percentile.
outlier_neurons = np.argwhere(distance_map > threshold) # Identify outlier neurons.

# Get the mappings for the outlier neurons
mappings = som.win_map(X)
anomalies = np.concatenate([mappings[tuple(neuron)] for neuron in outlier_neurons if tuple(neuron) in mappings], axis=0)
anomalies = sc.inverse_transform(anomalies) # Inverse the scaling.

# Part 2 - Going from Unsupervised to Supervised Deep Learning

## Creating the matrix of features
features = dataset.iloc[:, 1:].values # ilock stands for locate indexes. [rows, columns] : means all the rows and 1: means all columns except the first one.

## Creating the dependent variable
is_anomaly = np.zeros(len(dataset)) # Create a vector of zeros.
for i in range(len(dataset)): # Loop over the data points.
    if dataset.iloc[i, 0] in anomalies: # Check if the data point is in the anomalies list.
        is_anomaly[i] = 1 # Set the value to 1 if the data point is in the anomalies list.

## Feature Scaling
## Feature scaling is absolutely necessary for deep learning models. 
## It is so fundamental that it is applied to every single feature in the dataset, including the dummy variables.
sc = StandardScaler() # Create an object of the StandardScaler class.
features = sc.fit_transform(features) # Fit and transform

## Initializing the ANN
ann = tf.keras.models.Sequential() # Create an object of the Sequential class. 
# This class allows us to build the artificial neural network as a sequence of layers.

## Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=2, activation='relu')) # Add a dense layer to the ANN. Dense stands for fully connected.
# The units parameter specifies the number of neurons in the layer. 
# The number of units can be estimated by different ways. One of them is to begin with a small number of units and gradually increase them. 
# For example, you can start with 8, 16, or 32 units.
# A second approach is the Heuristic Rules: The number of units should be 2/3 of the input size plus the output size.
# The activation parameter specifies the activation function to be used in the layer. 
# relu stands for rectified linear unit.

## Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Add a dense layer to the ANN. 
# The units parameter specifies the number of neurons in the layer. 
# The activation parameter specifies the activation function to be used in the layer. 
# For binary classification problems, we use the sigmoid activation function.
# For non-binary classification problems, we use the softmax activation function.
# For regression problems, we don't use any activation function.
# For regression problems, the loss function is mean_squared_error.

## Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile the ANN. 
# The optimizer parameter specifies the optimization algorithm to be used to update the weights. 
# The loss parameter specifies the loss function to be used to calculate the error. 
# For non-binary classication the loss must be category_crossentropy. 
# The metrics parameter specifies the metric to be used to evaluate the model.

## Training the ANN on the Training set
ann.fit(features, is_anomaly, batch_size = 1, epochs = 2) # Train the ANN on the training set. 
# The batch_size parameter specifies the number of samples to be used in each batch. 
# The epochs parameter specifies the number of times the entire training set is passed through the ANN.

## Predicting the results
y_pred = ann.predict(features) # Predict the results.
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1) # Concatenate the IDs and the probabilities.
y_pred = y_pred[y_pred[:, 1].argsort()[::-1]] # Sort the data points by probability in descending order.

# Printing the IDs
print('IDs')
for i in y_pred:
    print(f"{i[0]}\t{i[1]}")