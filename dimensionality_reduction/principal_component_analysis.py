# Principal Component Analysis (PCA) 
# PCA is a dimensionality reduction technique that is widely used in machine learning.
# PCA is used to reduce the number of features in a dataset by projecting the data onto a lower-dimensional space.
# PCA is used to reduce the dimensionality of the data while preserving as much variance as possible.
# PCA is used to identify the most important features in a dataset.
# PCA is used to reduce the computational complexity of a machine learning model.

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
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, :-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and :-1 all the columns except the last one.
y = dataset.iloc[:, -1].values # : means all the rows and -1 the last column

# Spliting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # We split the dataset into 80% training set and 20% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # Create an object of the StandardScaler class.
X_train = sc.fit_transform(X_train) # Fit and transform the training set. We don't apply feature scaling to the dummy variables. We only apply feature scaling to the numerical variables.
X_test = sc.transform(X_test) # We only apply the transform method to the test set. We don't need to fit the test set because the StandardScaler object is already fitted to the training set.

# Applying PCA
# We need to apply PCA before training the machine learning model because we want to reduce the dimensionality of the dataset.
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # Create an object of the PCA class and specify the number of components we want to keep. We want to keep the two most important components.
X_train = pca.fit_transform(X_train) # Fit and transform the training set. We don't apply PCA to the dependent variable.
X_test = pca.transform(X_test) # We only apply the transform method to the test set. We don't need to fit the test set because the PCA object is already fitted to the training set.

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) # Create an object of the LogisticRegression class.
classifier.fit(X_train, y_train) # Fit the classifier to the training set.

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test) # Predict the test set results.
cm = confusion_matrix(y_test, y_pred) # Create a confusion matrix to evaluate the model's performance.
print(cm)
accuracy_score = accuracy_score(y_test, y_pred) # Calculate the accuracy of the model. The accuracy in the test set is the number of correct predictions divided by the total number of predictions in the test set.
print(accuracy_score)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()