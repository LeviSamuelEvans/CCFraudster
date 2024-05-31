# Credit Card Fraud Detection API

## Overview
This project provides a machine learning-based API for detecting fraudulent credit card transactions. The aim is to identify potentially fraudulent transactions based on transaction details using the best performing model from a selection of methods, e.g Isolation Forests, Logistic Regression, etc. The project involves data preprocessing, training, and evaluation of the model, as well as an API for making predictions.

We will be using a dataset from Kaggle `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`(thanks!).

From the dataset description:

- This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.


## Features
- **Data Loading:** Load transaction data into a SQL database.
- **Data Preprocessing:** Handle imbalanced data using SMOTE and preprocess the data for training. We create training, validation and testing sets. The validation set is required to tune model hyperparameters, such that the test-set can then give completely unbiased view of our model's performance!
- **Model Training:** Train a selection of models for anomaly detection.
- **Model Evaluation:** Evaluate the model using AUC-ROC, precision, and recall.
- **API:** A Flask application to serve the model predictions.

## Project Structure

### app.py
This file contains the Flask web application that serves the model's predictions through an API endpoint.

### Scripts
A brief outline of each script, and the order in which to run:

#### load_data.py
This script loads the dataset from a CSV file and saves it into a PostgreSQL database.

#### preprocess.py
This script preprocesses the data by handling imbalances using SMOTE, and splits the data into training, validation and testing sets.

#### train_model.py
This script trains the models on the preprocessed data and saves the trained model. The models currently include:
```
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

References
----------
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression

```

#### evaluate_model.py
This script evaluates the trained model on the test set using metrics like AUC-ROC, precision, and recall, and logs the evaluation results.

------

The trained models will be saved to the `models` directory, with the best performing model on the test set saved as `best_model.joblib`. We use `joblib` here since it faster and more efficient than pickling. The best performing model will be available in the application for inference. The data with the transactions is stored in a .csv file inside the `data` folder.


## Future Aims:

- Find more datasets
- Play around with more models
- More visualisation of the data
- Some notebooks
- CI/CD