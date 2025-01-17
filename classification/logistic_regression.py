# Logistic Regression

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
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values # ilock stands for locate indexes. [rows, columns] : means all the rows and :-1 all the columns except the last one.
y = dataset.iloc[:, -1].values # : means all the rows and -1 the last column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) # We split the dataset into 75% training set and 25% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

# Feature Scaling
# We need to apply feature scaling to the independent variables because we want to avoid the domination of one independent variable over the other. We don't want the model to be biased towards one independent variable.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # Create an object of the StandardScaler class.
X_train = sc.fit_transform(X_train) # Fit and transform the training set. We need to fit the training set to the StandardScaler object and then transform it.
X_test = sc.transform(X_test) # We don't need to fit the test set to the StandardScaler object because it's already fitted to the training set.

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) # Create an object of the LogisticRegression class.
classifier.fit(X_train, y_train) # Fit the classifier to the training set.

# Predicting a new result
# print(classifier.predict(sc.transform([[30,87000]]))) # We need to scale the input values because the model was trained on scaled values. The predict method expects a 2D array.
# predict method returns a binary value. 
# predict_proba method returns the probability of the prediction. The first column is the probability of the prediction being 0 and the second column is the probability of the prediction being 1.

# Predicting the Test set results
y_pred = classifier.predict(X_test) # Predict the test set results.
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # Concatenate the predicted values and the actual values. The reshape method is used to change the shape of the array (from horizontal to vertical). The len function returns the length of the array. The 1 argument specifies the axis along which the arrays will be joined.

# Making the Confusion Matrix
# The confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.
# The confusion matrix will show us the number of correct and incorrect predictions.
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred) # Create a confusion matrix to evaluate the model's performance.
print(cm)
accuracy_score = accuracy_score(y_test, y_pred) # Calculate the accuracy of the model. The accuracy in the test set is the number of correct predictions divided by the total number of predictions in the test set.
print(accuracy_score)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train # We need to inverse the scaling to get the original values. We don't want to plot the scaled values.
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25)) # Create a grid with all the pixels.
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue'))) # Plot the decision boundary.
plt.xlim(X1.min(), X1.max()) # Set the limits of the x-axis.
plt.ylim(X2.min(), X2.max()) # Set the limits of the y-axis.
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j) # Plot the data points.
plt.title('Logistic Regression (Training set)') # Set the title of the plot.
plt.xlabel('Age') # Set the label of the x-axis.
plt.ylabel('Estimated Salary') # Set the label of the y-axis.
plt.legend() # Show the legend.
plt.show() # Display the plot.

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()