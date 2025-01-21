# K-Means clustering algorithm
# K-Means is a clustering algorithm that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
# K-Means is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). 
# The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. 
# The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. 
# Data points are clustered based on feature similarity.

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
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 1:].values # ilock stands for locate indexes. [rows, columns] : means all the rows and 1: all the columns except the first one.

# Encoding categorical data (creation of dummy variable).
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # Create an object of the ColumnTransformer class and specify the transformation to be applied to the specified columns.
X = np.array(ct.fit_transform(X)) # Apply the transformation to the specified columns and convert the result into a NumPy array.

# Using the elbow method to find the optimal number of clusters.
from sklearn.cluster import KMeans
wcss = [] # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) # Create an object of the KMeans class and specify the number of clusters.
    kmeans.fit(X) # Fit the KMeans model to the dataset.
    wcss.append(kmeans.inertia_) # Append the inertia value to the wcss list. The inertia value is the sum of squared distances of samples to their closest cluster center.
plt.plot(range(1, 11), wcss) # Plot the number of clusters against the wcss values.
plt.title('The Elbow Method') # Set the title of the plot.
plt.xlabel('Number of clusters') # Set the x-axis label of the plot.
plt.ylabel('WCSS') # Set the y-axis label of the plot.
plt.show() # Display the plot.

# Training the K-Means model on the dataset.
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42) # Create an object of the KMeans class and specify the number of clusters.
y_kmeans = kmeans.fit_predict(X) # Fit the KMeans model to the dataset and predict the cluster for each data point.

# Adding the cluster labels to the original dataset
dataset['Cluster'] = y_kmeans 

# Print the updated dataset to verify
print(dataset)

# Save the updated dataset with the new dependent variable
#dataset.to_csv('Mall_Customers_with_Clusters.csv', index=False)
