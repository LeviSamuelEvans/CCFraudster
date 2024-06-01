import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(dotenv_path='secrets/.env')

db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST', 'localhost')
db_name = os.getenv('DB_NAME')

engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}')
