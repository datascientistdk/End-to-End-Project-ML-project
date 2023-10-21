import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from dotenv import load_dotenv 
import psycopg2



load_dotenv()

host=os.getenv("host")
database=os.getenv("db")
user=os.getenv("user")
password=os.getenv('password')

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        logging.info("Connection Established",mydb)
        df=pd.read_sql_query("Select * from students",mydb)
        print(df.head)

        return df


    except Exception as ex:
        raise CustomException(ex)
