# Machine Learning and Data Science Projects

This repository contains various machine learning and data science scripts, organized by different techniques and algorithms. Each project includes code for data preprocessing, model training, evaluation, and visualization (when possible).

## Project Structure

The repository is organized into the following directories:

- `association_rule_learning/`: Contains scripts related to association rule learning algorithms like Apriori and Eclat.
- `classification/`: Contains scripts related to classification algorithms like Decision Tree, K-Nearest Neighbors, Kernel SVM, Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machine.
- `clustering/`: Contains projects related to clustering algorithms like K-Means and Hierarchical Clustering.
- `data_preprocessing/`: Contains scripts for data preprocessing tasks.
- `deep_learning/`: Contains scripts related to deep learning algorithms like Artificial Neural Networks, Convolutional Neural Networks, and Recurrent Neural Networks.
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

3. **Install dependencies**:M
    ```sh
    pip install -r requirements.txt
    ```

### Usage

Each script is self-contained and can be run independently. Simply copy your dataset into the data folder within each main folder, make the necessary changes in the script (if you need to select different columns, for example), and execute the script to see the results. For example, to run a classification project, navigate to the classification directory and run the desired script:

```sh
cp DATASET.csv classification/data/
cd classification
python logistic_regression.py
```

## Contributing

Contributions are welcome! If you have any improvements or new projects to add, feel free to open a pull request.

## License

This repository is licensed under the MIT License. See the LICENSE file for more information.
