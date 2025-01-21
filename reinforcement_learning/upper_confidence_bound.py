# Upper Confidence Bound algorithm

# Upper Confidence Bound algorithm is a multi-armed bandit algorithm that balances exploration and exploitation.
# It is based on the principle of optimism in the face of uncertainty.
# The algorithm selects the action that has the highest upper confidence bound.
# The upper confidence bound is calculated as the sum of the average reward and the confidence interval.
# The average reward is calculated as the sum of the rewards of the action divided by the number of times the action was selected.
# The confidence interval is calculated as the square root of the log of the number of total rounds divided by the number of times the action was selected.

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

# Implement the UCB algorithm
import math
N = 10000 # Number of rounds
d = 10 # Number of ads
ads_selected = [] # List of ads selected at each round
numbers_of_selections = [0] * d # Number of times each ad was selected up to round n. This correspond to N_i(n)
sums_of_rewards = [0] * d # Sum of rewards of each ad up to round n. This correspond to R_i(n)
total_reward = 0 # Total reward. This correspond to the sum of all the rewards up to round n.
for n in range(0, N): # Loop through all the rounds n.
    ad = 0 # Selected ad at each round n. This correspond to a_i(n)
    max_upper_bound = 0 # Maximum upper bound at each round n. This correspond to UCB_i(n)
    for i in range(0, d): # Loop through all the ads to find the ad with the maximum upper bound
        if (numbers_of_selections[i] > 0): # If the ad i was selected at least once then calculate the upper bound.
            average_reward = sums_of_rewards[i] / numbers_of_selections[i] # Average reward of ad i up to round n. This correspond to r_i(n)
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i]) # Confidence interval of ad i up to round n. This correspond to delta_i(n)
            upper_bound = average_reward + delta_i # Upper bound of ad i up to round n. This correspond to UCB_i(n)
        else:
            upper_bound = 1e400 # Set the upper bound to a very large number if the ad i was not selected at least once.
        if upper_bound > max_upper_bound: 
            max_upper_bound = upper_bound
            ad = i # Select the ad with the maximum upper bound
            
    ads_selected.append(ad) # Append the selected ad to the list of ads selected at each round n. 
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1 # Increment the number of times the selected ad was selected.
    reward = dataset.values[n, ad] # Get the reward of the selected ad at round n. 
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward # Increment the sum of rewards of the selected ad.
    total_reward = total_reward + reward # Increment the total reward.

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
