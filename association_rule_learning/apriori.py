# Apriori algorithm implementation
# Apriori algorithm is a classical algorithm in data mining. 
# It is used for mining frequent itemsets and relevant association rules.
# It is devised to operate on a database containing a lot of transactions, for instance, items brought by customers in a store.
# The Apriori algorithm assumes that any subset of a frequent itemset must be frequent.
# The Apriori algorithm is mainly used to find associations between different products.
# The algorithm is based on the concept that a subset of a frequent itemset must also be a frequent itemset.
# The algorithm uses a breadth-first search strategy to count the support of itemsets and uses a candidate generation function to generate all possible itemsets.
# The algorithm has three main parts: Support, Confidence, and Lift.
# Support: The support of an itemset is the number of transactions that contain the itemset. It is calculated by dividing the number of transactions that contain the itemset by the total number of transactions.
# Confidence: The confidence of a rule is the number of transactions that contain all the items in the antecedent and the consequent. It is calculated by dividing the number of transactions that contain all the items in the antecedent and the consequent by the number of transactions that contain all the items in the antecedent.
# Lift: The lift of a rule is the ratio of the observed support to that expected if the two rules were independent. It is calculated by dividing the confidence of the rule by the support of the consequent.
# The Apriori algorithm has three main steps:
# 1. Generate all frequent itemsets.
# 2. Generate all association rules.
# 3. Evaluate the rules.

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

# Training Apriori on the dataset
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
#print(results)

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results] # The first item in the tuple is the left hand side of the rule.
    rhs         = [tuple(result[2][0][1])[0] for result in results] # The second item in the tuple is the right hand side of the rule.
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
#print(resultsinDataFrame)

## Displaying the results sorted by descending lifts
#print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift', keep='first'))