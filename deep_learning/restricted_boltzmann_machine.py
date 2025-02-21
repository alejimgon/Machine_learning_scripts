# Restricted Boltzmann Machine
# Restricted Boltzmann Machine is a generative stochastic neural network that can learn a probability distribution over its set of inputs.
# It is a two-layer neural network that learns a probability distribution over its set of inputs.
# It has visible and hidden layers, but no hidden-to-hidden connections.
# The visible units represent the input data and the hidden units model the features.
# The network is trained using the contrastive divergence algorithm.
# The network can be used to generate new samples from the learned distribution.
# The network can also be used for feature learning.
# The network can be used for dimensionality reduction, classification, regression, anomaly detection among other tasks.

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
class RBM():
    '''Class to create the Restricted Boltzmann Machine'''
    def __init__(self, nv, nh): # nv: number of visible nodes, nh: number of hidden nodes
        '''Function to initialize the Restricted Boltzmann Machine'''
        self.W = torch.randn(nh, nv) # Initializing the weights. This initializes the weights for the probability of the visible nodes given the hidden nodes.
        self.a = torch.randn(1, nh) # Initializing the bias for hidden nodes. This initializes the bias for the probability of the hidden nodes given the visible nodes.
        self.b = torch.randn(1, nv) # Initializing the bias for visible nodes. This initializes the bias for the probability of the visible nodes given the hidden nodes.
    
    def sample_h(self, x): # x: corresponds to the visible neurons.
        '''Function to sample the hidden nodes. 
        This function returns the probability of the hidden nodes given the visible nodes and the hidden nodes.'''
        wx = torch.mm(x, self.W.t()) # wx: product of the visible nodes and the weights.
        activation = wx + self.a.expand_as(wx) # activation: product of the visible nodes and the weights plus the bias for the hidden nodes.
        p_h_given_v = torch.sigmoid(activation) # p_h_given_v: probability of the hidden nodes given the visible nodes, 
        return p_h_given_v, torch.bernoulli(p_h_given_v) # torch.bernoulli: returns a binary sample of the probability of the hidden nodes given the visible nodes.
    
    def sample_v(self, y): # y: corresponds to the hidden neurons.
        '''Function to sample the visible nodes.
        This function returns the probability of the visible nodes given the hidden nodes and the visible nodes.'''
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk): # v0: input vector, vk: visible nodes after k samplings, ph0: probability of the hidden nodes at the first iteration, phk: probability of the hidden nodes after k samplings given the value of the visible nodes (vk).
        '''Function to train the model.
        This function updates the weights and biases of the Restricted Boltzmann Machine.'''
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t() # Updating the weights
        self.b += torch.sum((v0 - vk), 0) # Updating the bias for the visible nodes
        self.a += torch.sum((ph0 - phk), 0) # Updating the bias for the hidden nodes

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

def binarize_ratings(data):
    '''Function to convert ratings into binary ratings 1 (Liked) or 0 (Not Liked)'''
    for i in range(data.size(0)):  # Looping through the number of users
        for j in range(data.size(1)):  # Looping through the number of movies
            if data[i, j] == 0:
                data[i, j] = -1
            elif data[i, j] == 1 or data[i, j] == 2:
                data[i, j] = 0
            elif data[i, j] >= 3:
                data[i, j] = 1
    return data

# Importing the dataset
#movies = pd.read_csv(f'{data_folder}/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#users = pd.read_csv(f'{data_folder}/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#ratings = pd.read_csv(f'{data_folder}/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv(f'{data_folder}/YOUR_DATASET.csv', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

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

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set = binarize_ratings(training_set)
test_set = binarize_ratings(test_set)

# Creating the architecture of the Neural Network
nv = len(training_set[0]) # Number of visible nodes.
nh = 100 # Number of hidden nodes (can be tuned).
batch_size = 100 # Batch size (can be tuned). The number of samples to work through before updating the internal model parameters.
rbm = RBM(nv, nh)

# Training the Restricted Boltzmann Machine
nb_epoch = 10 # Number of epochs (can be tuned). An epoch is a measure of the number of times all of the training vectors are used once to update the weights.
for epoch in range(1, nb_epoch + 1): 
    train_loss = 0 
    train_rmse = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size] # vk: visible nodes after k samplings.
        v0 = training_set[id_user:id_user+batch_size] # v0: input vector.
        ph0,_ = rbm.sample_h(v0) # ph0: probability of the hidden nodes at the first iteration. ,_ is used to ignore the second output of the function.
        for k in range(10):
            _,hk = rbm.sample_h(vk) # _, is used to ignore the first output of the function. hk: hidden nodes after k samplings.
            _,vk = rbm.sample_v(hk) # _, is used to ignore the first output of the function. vk: visible nodes after k samplings.
            vk[v0<0] = v0[v0<0] # We do not want to learn the missing data.
        phk,_ = rbm.sample_h(vk) # phk: probability of the hidden nodes after k samplings given the value of the visible nodes (vk).
        rbm.train(v0, vk, ph0, phk) # Training the model.
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) # Computing the loss.
        train_rmse += torch.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # Computing the root mean squared error.
        s += 1. 
    print(f'Epoch: {epoch} Train Loss (Average Distance): {train_loss/s} Train RMSE: {train_rmse/s}')

# Testing the Restricted Boltzmann Machine
test_loss = 0
test_rmse = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1] # Training set is used to activate the neurons.
    vt = test_set[id_user:id_user+1] # Test set is used to compare the predictions.
    if len(vt[vt>=0]) > 0: # We want to learn only the data that is available.
        _,h = rbm.sample_h(v) 
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) # Computing the loss.
        test_rmse += torch.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # Computing the root mean squared error.
        s += 1.
print(f'Test Loss (Average Distance): {test_loss/s}')
print(f'Test RMSE: {test_rmse/s}')

# Saving the model
#torch.save(rbm, f'{main_repo_folder}/models/rbm.pth')

# Making predictions for a specific user
user_id = 1  # Change this to the ID of the user you want to make predictions for
user_ratings = training_set[user_id-1:user_id] # Getting the ratings of the user
_, hidden = rbm.sample_h(user_ratings) # Getting the hidden neurons
_, predictions = rbm.sample_v(hidden)  # Getting the predicted ratings
predictions = predictions.detach().numpy().flatten()  # Detaching the tensor from the computation graph and converting it into a numpy array
print(f'Predictions for user {user_id}: {predictions}')
