# Thompsom sampling algorithm for multi-armed bandit problem
# This algorithm is based on the principle of Bayesian inference.
# The algorithm selects the action that has the highest sample from the posterior distribution.
# The posterior distribution is calculated as the Beta distribution.
# The Beta distribution is a probability distribution that describes the probability of the success of a Bernoulli experiment.
# The Beta distribution has two parameters: alpha and beta.
# The alpha parameter is the number of successes plus 1.
# The beta parameter is the number of failures plus 1.

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
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement the Thompsom Sampling algorithm
import random
N = 10000 # Number of rounds
d = 10 # Number of ads. Each ad has a different conversion rate. The goal is to find the ad with the highest conversion rate.
ads_selected = [] # List of ads selected at each round
numbers_of_rewards_1 = [0] * d # Number of times the ad got a reward of 1 up to round n. This correspond to N1_i(n).
numbers_of_rewards_0 = [0] * d # Number of times the ad got a reward of 0 up to round n. This correspond to N0_i(n).
total_reward = 0 # Total reward. This correspond to the sum of all the rewards up to round n.
for n in range(0, N): # Loop through all the rounds n.
    ad = 0 # Selected ad at each round n. 
    max_random = 0 # Maximum random draw at each round n. This correspond to theta_i(n).
    for i in range(0, d): # Loop through all the ads to find the ad with the maximum random draw.
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1) # Random draw from the Beta distribution of ad i up to round n. This correspond to theta_i(n).
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad) # Append the selected ad to the list of ads selected at each round n.
    reward = dataset.values[n, ad] # Get the reward of the selected ad at round n.
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1 # Increment the number of times the ad got a reward of 1.
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1 # Increment the number of times the ad got a reward of 0.
    total_reward = total_reward + reward # Increment the total reward.

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
