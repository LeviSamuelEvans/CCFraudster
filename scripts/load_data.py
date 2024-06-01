import pandas as pd
from config import engine

def load_data_to_database():
    """Load data to the database.

    This method reads the data from the CSV file and loads it to the PostgreSQL database.
    """

    data = pd.read_csv('data/creditcard.csv')
    data.to_sql('credit_card_fraud', engine, if_exists='replace', index=False)

