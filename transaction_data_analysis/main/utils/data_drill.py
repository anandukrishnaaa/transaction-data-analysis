import pandas as pd


def perform_analysis(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Get first few rows
    first_rows = df.head(5).to_html()

    # Perform the analysis
    mean = df.mean()
    median = df.median()
    mode = df.mode()

    # Return the results
    return {"sample": first_rows, "mean": mean, "median": median, "mode": mode}
