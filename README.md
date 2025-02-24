# Machine Learning and Data Science Projects

## Table of Contents
- [Description](#description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description
This repository contains various machine learning and data science scripts, organized by different techniques and algorithms based on the courses Machine Learning A-Z and Deep Learning A-Z by SuperDataScience Team. Each project includes code for data preprocessing, model training, evaluation, and visualization (when possible).

## Project Structure

The repository is organized into the following directories:

- `association_rule_learning/`: Contains scripts related to association rule learning algorithms like Apriori and Eclat.
- `classification/`: Contains scripts related to classification algorithms like Decision Tree, K-Nearest Neighbors, Kernel SVM, Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machine.
- `clustering/`: Contains projects related to clustering algorithms like K-Means and Hierarchical Clustering.
- `data_preprocessing/`: Contains scripts for data preprocessing tasks.
- `deep_learning/`: Contains scripts related to deep learning algorithms like Artificial Neural Networks, Convolutional Neural Networks, Recurrent Neural Networks, Self Organizing Maps, a hybrid model SOM and ANN, a Restricted Boltzmann Machine, and an Autoencoder.
- `dimensionality_reduction/`: Contains scripts related to dimensionality reduction techniques like PCA, LDA, and Kernel PCA.
- `model_selection/`: Contains scripts for model selection techniques like k-Fold Cross Validation and Grid Search.
- `natural_language_processing/`: Contains scripts related to natural language processing tasks.
- `regression/`: Contains scripts related to regression algorithms like Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, Support Vector Regression, Decision Tree Regression, and Random Forest Regression.
- `reinforcement_learning/`: Contains scripts related to reinforcement learning algorithms like Thompson Sampling and Upper Confidence Bound.
- `XGBoost/`: Contains scripts related to the XGBoost and CATBoost algorithms.

## Getting Started

### Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/alejimgon/Machine_learning_scripts.git
    cd Machine_learning_scripts
    ```

2. **Set up the conda environment**:
    ```sh
    conda create --name ml_env python=3.12.7
    conda activate ml_env
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

Each script is self-contained and can be run independently. Follow these steps to use the scripts:

1. **Prepare Your Dataset**:
    - Ensure your dataset is in CSV format.
    - Place your dataset in the `data` folder within the respective project directory. For example, if you are working on a classification project, place your dataset in [data](http://_vscodecontentref_/0).

2. **Modify the Script**:
    - Open the script you want to run and make any necessary changes. For example, you may need to update the file path to your dataset or select different columns for analysis.

3. **Run the Script**:
    - Navigate to the project directory and run the script. For example, to run a classification project, use the following commands:
    ```sh
    cp DATASET.csv classification/data/
    cd classification
    python logistic_regression.py
    ```

## Contributing

Contributions are welcome! If you have any improvements or new projects to add, feel free to open a pull request.

## License

This repository is licensed under the MIT License. See the LICENSE file for more information.