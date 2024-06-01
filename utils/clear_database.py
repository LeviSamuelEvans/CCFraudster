from sqlalchemy import text
from config import engine

def clear_database():
    """Clear the database by dropping the relevant tables."""

    tables_to_clear = ['credit_card_fraud', 'X_train', 'X_test', 'y_train', 'y_test']

    with engine.connect() as conn:
        for table in tables_to_clear:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
            print(f"Table {table} has been dropped.")

if __name__ == '__main__':
    clear_database()
