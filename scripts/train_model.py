import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import dump
import logging
from config import engine

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_database():
    return engine

def load_data(engine):
    logging.info('Loading preprocessed data...')
    X_train = pd.read_sql('SELECT * FROM "X_train"', engine)
    y_train = pd.read_sql('SELECT * FROM "y_train"', engine)
    X_val = pd.read_sql('SELECT * FROM "X_val"', engine)
    y_val = pd.read_sql('SELECT * FROM "y_val"', engine)
    return X_train, y_train, X_val, y_val

def train_and_save_models(X_train, y_train, X_val, y_val):
    y = y.values.ravel()

    models = {
        "IsolationForest": IsolationForest(contamination=0.01, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    }

    for name, model in models.items():
        logging.info(f'Training model: {name}')
        model.fit(X_train, y_train)

        # let's save all our models in case we want to use them later...
        model_path = f'models/{name}.joblib'
        dump(model, model_path)
        logging.info(f'Saved model: {model_path}')

        # test the trained model in the validation sets
        y_pred = model.predict(X_val)

        # show the classification report ( lovely little feature this! :D)
        logging.info(f'Classification Report for {name}:')
        print(f'Classification Report for {name}:')
        print(classification_report(y_val, y_pred, zero_division=0))

        # now log the all important AUC score
        if name in ["RandomForestClassifier", "LogisticRegression"]:
            y_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
            logging.info(f'AUC for {name}: {auc}')
            print(f'AUC for {name}: {auc}')

    return models

def main():
    setup_logging()
    engine = connect_to_database()
    X, y = load_data(engine)
    train_and_save_models(X, y)

if __name__ == "__main__":
    main()
