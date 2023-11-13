# utils / data_forge.py

import pandas as pd
import numpy as np
import io
import json


def file_preview(file_uploaded, enforce_float=True):
    if enforce_float:
        pd.options.display.float_format = "{:.2f}".format
    df = pd.read_csv(file_uploaded.file_path, index_col=False)
    file_uploaded.file_path.close()
    uploaded_file_details = {
        "head": df.head(10)
        .to_html(index=False)
        .replace('class="dataframe"', 'class="table table-sm table-bordered"'),
        "desc": df.describe()
        .to_html()
        .replace('class="dataframe"', 'class="table table-sm table-bordered"'),
        "path": file_uploaded.file_path.path,
    }
    return uploaded_file_details


# Exploratory data analysis

uploaded_file_path = r"C:\Users\AK\Projects\code\coding-projects\transaction-data-analysis\dataset\sample.csv"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class DataAnalysis:
    def exploratory_analysis(self):
        df = self.source_df
        df.drop("isFlaggedFraud", axis=1)
        # Create a buffer
        buffer = io.StringIO()
        # Call df.info() and write the output to the buffer
        # Basic summary of the dataset
        df.info(buf=buffer)
        # Get the output from the buffer
        info_output = buffer.getvalue()
        print(info_output)
        # Check for missing values
        df.isnull().sum()
        # Initialize an empty dictionary to store the results
        results = {}

        # Select only the numerical columns
        numerical_df = df.select_dtypes(include=[np.number])

        # Iterate over each column in the numerical DataFrame
        for column in numerical_df.columns:
            # Get the minimum and maximum values for the column
            min_value = numerical_df[column].min()
            max_value = numerical_df[column].max()

            # Store the minimum and maximum values in the dictionary
            results[column] = [min_value, max_value]

        # Convert the results dictionary into a JSON string
        results_json = json.dumps(results, cls=NpEncoder)
        # Print the results
        print(results_json)

    def __init__(self, file_path):
        self.source_df = pd.read_csv(file_path, index_col=False)
        print("Source file initialised into dataframe")


da = DataAnalysis(uploaded_file_path)
da.exploratory_analysis()
