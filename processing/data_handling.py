import os
import pandas as pd
import joblib
import pickle

from util import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    return _data

#Save column
def Save_column(columns,COLUMN_NAME):
    filepath = os.path.join(config.SAVE_MODEL_PATH,COLUMN_NAME)

    # Save the list of column names to a pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(columns, f)

#Load column
def Load_column(COLUMN_NAME):
    filepath = os.path.join(config.SAVE_MODEL_PATH,COLUMN_NAME)

    # Load the list of column names from the pickle file
    with open(filepath, 'rb') as f:
        loaded_column_names = pickle.load(f)

    return loaded_column_names


#Serialization
def save_pipeline(pipeline_to_save,MODEL_NAME):
    save_path = os.path.join(config.SAVE_MODEL_PATH,MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {MODEL_NAME}")

#Deserialization
def load_pipeline(MODEL_NAME):
    save_path = os.path.join(config.SAVE_MODEL_PATH,MODEL_NAME)
    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded")
    return model_loaded