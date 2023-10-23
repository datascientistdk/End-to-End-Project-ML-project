import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


from src.mlproject.utils import save_object
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import os

@dataclass
class DataTransfromationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','prepocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.DataTransfromationConfig=DataTransfromationConfig()


    def get_data_transfromer_object(self):
        '''
        this function is reponsible for data transformation'''

        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
            ])    

            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])    

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("Num_pipepline",num_pipeline,numerical_columns),
                    ("Cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj=self.get_data_transfromer_object()

            target_columns_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            # Divided the train dataset into independent and dependent features

            input_feature_train_df=train_df.drop(columns=[target_columns_name],axis=1)
            target_feature_train_df=train_df[target_columns_name]

            # Divided the test dataset into independent and dependent features

            input_feature_test_df=test_df.drop(columns=[target_columns_name],axis=1)
            target_feature_test_df=test_df[target_columns_name]

            logging.info("Appling preprocessing on training and test dataframes")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                 input_feature_train_arr, np.array(target_feature_train_df)
             ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing objects")

            save_object(

                file_path=self.DataTransfromationConfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (

                train_arr,
                test_arr,
                self.DataTransfromationConfig.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(sys,e)
        
