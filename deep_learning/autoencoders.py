# Autoencoders
# Autoencoders are a type of neural network that can be used to learn efficient representations of data.
# Autoencoders consist of an encoder and a decoder, which are used to compress and decompress the input data.
# The encoder is used to compress the input data into a lower-dimensional representation, while the decoder is used to reconstruct the input data from the compressed representation.
# Autoencoders are trained to minimize the reconstruction error, which is the difference between the input data and the reconstructed data.
# Autoencoders can be used for tasks such as data denoising, dimensionality reduction, and anomaly detection.

# Importing the libraries
import numpy as np # For numerical computations
import pandas as pd # For data manipulation and analysis
import torch # For building the neural network
import torch.nn as nn # For building the neural network
import torch.nn.parallel # For parallel computations
import torch.optim as optim # For optimization
import torch.utils.data # For data manipulation and analysis
from torch.autograd import Variable # For neural network training (stochastic gradient descent)

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Classes
class SAE(nn.Module): # nn.Module is the base class for all neural network modules in PyTorch.
    '''Class to create the Stacked AutoEncoder. 
    SAE stands for Stacked AutoEncoder.'''
    def __init__(self, ):
        '''Function to initialize the Stacked AutoEncoder'''
        super(SAE, self).__init__() # Super function to inherit the properties of the parent class nn.Module
        self.fc1 = nn.Linear(nb_movies, 20) # First encoding layer with 20 neurons (encoding layer)
        self.fc2 = nn.Linear(20, 10) # Second encoding layer with 10 neurons (encoding layer)
        self.fc3 = nn.Linear(10, 20) # First decoding layer with 10 neurons (decoding layer)
        self.fc4 = nn.Linear(20, nb_movies) # Second decoding layer with 20 neurons (decoding layer)
        self.activation = nn.Sigmoid() # Activation function for the neural network

    def forward(self, x): 
        '''Function to perform the forward propagation'''
        x = self.activation(self.fc1(x)) # Encoding the input vector and getting the first encoding layer
        x = self.activation(self.fc2(x)) # Encoding the input vector and getting the second encoding layer
        x = self.activation(self.fc3(x)) # Decoding the input vector and getting the first decoding layer
        x = self.fc4(x) # Decoding the input vector and getting the second decoding layer
        return x

# Functions
def convert(data):
    '''Function to convert the data into an array with users in lines, movies in columns and ratings as values'''
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

# Importing the dataset
#movies = pd.read_csv(f'{data_folder}/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#users = pd.read_csv(f'{data_folder}/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#ratings = pd.read_csv(f'{data_folder}/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int'))

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines, movies in columns and ratings as values
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
# Tensor is a multi-dimensional matrix containing elements of a single data type.
# The number of dimensions of a tensor is called its rank.
training_set = torch.FloatTensor(training_set) 
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
sae = SAE()
criterion = nn.MSELoss() # Mean Squared Error Loss
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # RMSprop optimizer with L2 regularization (weight_decay)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1): 
    train_loss = 0
    s = 0.
    for id_user in range(nb_users): # Looping through the number of users
        input = Variable(training_set[id_user]).unsqueeze(0) # Adding a new dimension corresponding to the batch
        target = input.clone() # Cloning the input
        if torch.sum(target.data > 0) > 0: # Checking if the target has non-zero ratings
            output = sae(input) # Getting the output
            target.require_grad = False # Not computing the gradients with respect to the target
            output[target == 0] = 0 # Not computing the loss with respect to the movies that were not rated
            loss = criterion(output, target) # Computing the loss
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # Normalizing the loss
            loss.backward() # Backpropagation
            train_loss += np.sqrt(loss.data*mean_corrector) # Updating the loss
            s += 1. # Updating the counter
            optimizer.step() # Updating the weights
    print(f'epoch: {epoch} loss: {train_loss/s}') 

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users): # Looping through the number of users
    input = Variable(training_set[id_user]).unsqueeze(0) # Adding a new dimension corresponding to the batch
    target = Variable(test_set[id_user]).unsqueeze(0) # Adding a new dimension corresponding to the batch
    if torch.sum(target.data > 0) > 0: # Checking if the target has non-zero ratings
        output = sae(input) # Getting the output
        target.require_grad = False # Not computing the gradients with respect to the target
        output[target == 0] = 0 # Not computing the loss with respect to the movies that were not rated
        loss = criterion(output, target) # Computing the loss
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # Normalizing the loss
        test_loss += np.sqrt(loss.data*mean_corrector) # Updating the loss
        s += 1. # Updating the counter
print(f'Test Loss: {test_loss/s}')
