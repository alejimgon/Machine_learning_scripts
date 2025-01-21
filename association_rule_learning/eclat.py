# Eclat 
# Eclat is an algorithm for frequent item set mining and association rule learning over transactional databases.
# It is an alternative to the Apriori algorithm.
# It is faster than the Apriori algorithm.
# It uses a depth-first search strategy.
# It is used to find associations between different products.

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
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) # We don't have a header in our dataset.
transactions = [] # Create an empty list to store the transactions.
for i in range(0, 7501): # We have 7501 transactions in our dataset.
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)]) # We have 20 products in our dataset. dataset.values[i, j] returns the value of the cell in the ith row and jth column.

# Training Eclat on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2) # The apriori function takes the transactions as input and returns the rules. 
# The min_support parameter specifies the minimum support of the itemsets.
# The min_confidence parameter specifies the minimum confidence of the rules.
# The min_lift parameter specifies the minimum lift of the rules.
# The min_length parameter specifies the minimum number of items in the rules.
# The max_length parameter specifies the maximum number of items in the rules.

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results sorted by descending supports
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))