# Natural Language Processing
# NLP is a field in machine learning with the ability of a computer to understand, analyze, manipulate, and potentially generate human language.

# Importing the libraries
import numpy as np # Allows us to work with arrays
import matplotlib.pyplot as plt # Allows us to plot charts
import pandas as pd # Allows us to import datasets and create the matrix of features and dependent variable
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define a function to train, predict, and evaluate a model
def train_evaluate_model(classifier, X_train, y_train, X_test, y_test, model_name):
    classifier.fit(X_train, y_train)  # Train the model
    y_pred = classifier.predict(X_test)  # Predict the test set results
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy
    precision = precision_score(y_test, y_pred, average='weighted')  # Calculate the precision
    recall = recall_score(y_test, y_pred, average='weighted')  # Calculate the recall
    f1 = f1_score(y_test, y_pred, average='weighted')  # Calculate the F1 score
    return accuracy, precision, recall, f1

# List of classifiers to train and evaluate
classifiers = [
    (LogisticRegression(random_state=0), "Logistic Regression"),
    (KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2), "K-NN"),
    (SVC(kernel='linear', random_state=0), "SVM"),
    (SVC(kernel='rbf', random_state=0), "Kernel SVM"),
    (GaussianNB(), "Naive Bayes"),
    (DecisionTreeClassifier(criterion='entropy', random_state=0), "Decision Tree"),
    (RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0), "Random Forest"),
    (GradientBoostingClassifier(random_state=0), "Gradient Boosting"),
    (AdaBoostClassifier(random_state=0), "AdaBoost"),
    (ExtraTreesClassifier(n_estimators=100, random_state=0), "Extra Trees")
]

# Get the directory of the current script.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory.
os.chdir(script_dir)

# Importing the dataset.
dataset=pd.read_csv('ENTER_THE_NAME_OF_YOUR_DATASET_HERE.tsv', sep='\t', quoting=3) # quoting=3 ignores the double quotes in the dataset.

# Cleaning the texts
import re # Regular Expression library
import nltk # Natural Language Toolkit
import ssl # Secure Sockets Layer. The Python ssl module uses the OpenSSL library, which is a secure sockets layer implementation.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords') # Download the stopwords dataset. Stopwords are the words that are not relevant in the text. For example, 'the', 'is', 'and', etc.
from nltk.corpus import stopwords # Import the stopwords dataset.
from nltk.stem.porter import PorterStemmer # Import the PorterStemmer class. This class is used to stem the words in the text. Stemming is the process of reducing a word to its root form. For example, 'loved' to 'love'.
corpus = [] # Create an empty list to store the cleaned texts. It will be used to create the Bag of Words model. 
for i in range(0, 1000): # Loop through all the reviews in the dataset.
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # Replace all the characters that are not letters with a space. We don't want to remove the letters because we need them to create the Bag of Words model.
    review = review.lower() # Convert all the letters to lowercase.
    review = review.split() # Split the review into words.
    ps = PorterStemmer() # Create an object of the PorterStemmer class.
    all_stopwords = stopwords.words('english') # Get all the stopwords in English.
    all_stopwords.remove('not') # Remove the word 'not' from the stopwords list because it is important for sentiment analysis.
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)] # Stem the words in the review and remove the stopwords. We use a set to make the search faster.
    review = ' '.join(review) # Join the words back together.
    corpus.append(review) # Append the cleaned review to the corpus list.

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer # Import the CountVectorizer class. This class is used to create the Bag of Words model.
cv = CountVectorizer(max_features=1500) # Create an object of the CountVectorizer class and specify the maximum number of words to keep.
X = cv.fit_transform(corpus).toarray() # Fit the corpus to the CountVectorizer object and convert it to an array. This array will be the independent variable in the model.
y = dataset.iloc[:, -1].values # Get the dependent variable from the dataset.

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # We split the dataset into 80% training set and 20% test set. The random_state parameter is used to ensure that we get the same results every time we run the code.

# Train and evaluate each classifier
results = {}
for classifier, name in classifiers:
    accuracy, precision, recall, f1 = train_evaluate_model(classifier, X_train, y_train, X_test, y_test, name)
    results[name] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

# Print the results in a tabular format
print("\nModel Performance Metrics:")
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
for name, metrics in results.items():
    print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")

