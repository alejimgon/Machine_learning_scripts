# Hieralchical clustering algorithm
# Hierarchical clustering is a type of unsupervised machine learning algorithm used to cluster unlabeled data points. 
# Like K-Means clustering, hierarchical clustering also groups together the data points with similar characteristics.
# There are two types of hierarchical clustering: Agglomerative and Divisive.

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
X = dataset.iloc[:, 1:].values # ilock stands for locate indexes. [rows, columns] : means all the rows and 3:4 all the columns except the first one.

# Encoding categorical data (creation of dummy variable).
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # Create an object of the ColumnTransformer class and specify the transformation to be applied to the specified columns.
X = np.array(ct.fit_transform(X)) # Apply the transformation to the specified columns and convert the result into a NumPy array.

# Using the dendrogram to find the optimal number of clusters.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward')) # Create a dendrogram object and specify the linkage method. The ward method minimizes the variance of the clusters being merged.
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model on the dataset.
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'ward') # Create an object of the AgglomerativeClustering class and specify the number of clusters, the affinity, and the linkage method. The metric parameter specifies the distance metric to use. The linkage parameter specifies the linkage criterion to use.
y_hc = hc.fit_predict(X) # Fit the AgglomerativeClustering model to the dataset and predict the cluster for each data point.

# Visualising the clusters.
plt.scatter(X[y_hc == 0, 3], X[y_hc == 0, 4], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 3], X[y_hc == 1, 4], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 3], X[y_hc == 2, 4], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 3], X[y_hc == 3, 4], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 3], X[y_hc == 4, 4], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Adding the cluster labels to the original dataset
dataset['Cluster'] = y_hc 

# Print the updated dataset to verify
print(dataset)