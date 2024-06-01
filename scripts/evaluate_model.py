import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from joblib import load, dump
import logging
import os
from config import engine


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def connect_to_database():
    return engine


def load_test_data(engine):
    logging.info("Loading test data...")
    X_test = pd.read_sql('SELECT * FROM "X_test"', engine)
    y_test = pd.read_sql('SELECT * FROM "y_test"', engine)
    return X_test, y_test


def evaluate_models(models_path, X_test, y_test):

    # need to esnrue y_test is a 1D array
    y_test = y_test.values.ravel()

    best_model_name = None
    best_model = None
    best_score = 0

    for model_file in os.listdir(models_path):
        if model_file.endswith(".joblib"):
            model_name = model_file.split(".")[0]
            model_path = os.path.join(models_path, model_file)
            model = load(model_path)
            logging.info(f"Evaluating model: {model_name}")

            y_pred = model.predict(X_test)

            # classification reports
            logging.info(f"Classification Report for {model_name}:")
            print(f"Classification Report for {model_name}:")
            print(classification_report(y_test, y_pred, zero_division=0))

            # calc accuracy
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Accuracy for {model_name}: {accuracy}")
            print(f"Accuracy for {model_name}: {accuracy}")

            # if the model has predict_proba method, calculate AUC
            auc = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                logging.info(f"AUC for {model_name}: {auc}")
                print(f"AUC for {model_name}: {auc}")

            # Use AUC as the primary metric for selecting the best model,
            # fall back to accuracy if AUC is not available
            if auc is not None:
                score = auc
            else:
                score = accuracy

            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_model = model

    logging.info(f"Best model: {best_model_name} with score: {best_score}")
    return best_model_name, best_model


def save_best_model(model, model_name):
    logging.info(f"Saving the best performing model: {model_name}")
    dump(model, "models/best_model.joblib")


def main():
    setup_logging()
    engine = connect_to_database()
    X_test, y_test = load_test_data(engine)
    best_model_name, best_model = evaluate_models("models", X_test, y_test)
    save_best_model(best_model, best_model_name)


if __name__ == "__main__":
    main()
