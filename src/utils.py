import re
from tqdm import tqdm
from typing import List
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import os
import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def convert_ctfidf(ctfidf_json:dict):
    ctf_idf_converted = {}
    for key in ctfidf_json.keys():
        ctf_idf_converted[int(key)] = list_to_dict(ctfidf_json[key])
    return ctf_idf_converted

def list_to_dict(input_list):
    result_dict = {}
    for item in input_list:
        result_dict[item[0]] = item[1]
    return result_dict

def create_folders_if_not_exist(folder_path):
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # If not, create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def load_20newsgroups_and_save_csv(path:str):
    # Load the 20 Newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # Create a DataFrame with 'data' and 'target' columns
    df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})

    if not path[-1]=="/": path+="/"
    # Save the DataFrame to a CSV file
    df.to_csv(path+'20newsgroups.csv', index=False)

def clean_text(text):
    text = re.sub(r"\\[a-zA-Z]", " ", text)  # Remove escape sequences
    text = re.sub(r"\S+@\S+", " ", text)  # Remove email addresses
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation
    text = " ".join(text.split())  # Remove extra whitespace

    return text

def clean_dataset(data: List[str]) -> List[str]:
    """Clean the dataset and return.

    Args:
        data (List): List of cleaned strings.
    """
    docs = []
    for text in tqdm(data):
        docs.append(clean_text(text))

def read_csv_column(csv_file: str, column_name:str) -> List[str]:
    """
    Reads a CSV file and returns a specified column as a list of strings.

    Parameters:
    - csv_file (str): The path to the CSV file.
    - column_name (str): The name of the column to extract.

    Returns:
    - list: A list of strings representing the specified column.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if the specified column exists
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

        # Extract the specified column as a list of strings
        column_values = df[column_name].astype(str).tolist()

        return column_values

    except Exception as e:
        print(f"Error: {e}")
        return None