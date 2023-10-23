import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd #Pandas
from dotenv import load_dotenv 
import psycopg2

import pickle
import numpy as np


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

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)