import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
from config import engine


def load_data():
    """Load data from the database"""
    data = pd.read_sql("SELECT * FROM credit_card_fraud", engine)
    return data


def handle_imbalanced_data(X, y):
    """Handle imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique).

    Reference
    ---------
        https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
    """
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def perform_train_val_test_split(X, y):
    """Perform train-validation-test split on the data."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_data_to_database(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save preprocessed data to the database."""
    X_train.to_sql("X_train", engine, if_exists="replace", index=False)
    X_val.to_sql("X_val", engine, if_exists="replace", index=False)
    X_test.to_sql("X_test", engine, if_exists="replace", index=False)
    y_train.to_sql("y_train", engine, if_exists="replace", index=False)
    y_val.to_sql("y_val", engine, if_exists="replace", index=False)
    y_test.to_sql("y_test", engine, if_exists="replace", index=False)


def preprocess_data():
    """Method to preprocess data.

    This method loads data from the database, handles imbalanced data, performs train-test split,
    and saves the preprocessed data to the database.

    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load data
    logging.info("Loading data from the database...")
    data = load_data()

    # Features and target
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # Handle imbalanced data
    logging.info("Handling imbalanced data using SMOTE...")
    X_resampled, y_resampled = handle_imbalanced_data(X, y)

    # Perform train-test split
    logging.info("Performing train-test split...")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = perform_train_val_test_split(X_resampled, y_resampled)

    # Save preprocessed data to the database
    logging.info("Saving preprocessed data to the database...")
    save_data_to_database(X_train, X_val, X_test, y_train, y_val, y_test)

    logging.info("Data preprocessing completed successfully!")

    # Quick look at our database
    print(data)


if __name__ == "__main__":
    preprocess_data()
