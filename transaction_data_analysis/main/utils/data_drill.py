import pandas as pd


def perform_analysis(file_path):
    # Read the CSV file into a pandas DataFrame
    # Test with static dataset
    file_path = r"C:\Users\AK\Projects\code\coding-projects\transaction-data-analysis\dataset\sample.csv"
    pd.options.display.float_format = "{:.2f}".format

    df = pd.read_csv(file_path)

    # Get first few rows
    no_of_rows = 5
    first_x_rows = df.head(no_of_rows).to_html()

    # Process only numeric cols
    df_cleaned = df.select_dtypes(["number"])

    # print(df_cleaned.describe())

    # Perform the analysis
    mean = df_cleaned.mean()
    median = df_cleaned.median()
    mode = df_cleaned.mode()

    total_amount = df["amount"].sum()
    transactions_per_customer = df["nameOrig"].value_counts()
    avg_amount_per_customer = df.groupby("nameOrig")["amount"].mean()

    data = {
        "first_x_rows": first_x_rows,
        "stats": {"mean": mean, "median": median, "mode": mode},
    }

    # print(data)

    # Return the results
    return data


# # Test runs
# file_path = r"C:\Users\AK\Projects\code\coding-projects\transaction-data-analysis\dataset\sample.csv"
# perform_analysis(file_path)
