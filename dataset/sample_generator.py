import pandas as pd

# Set the chunk size and number of files
chunk_size = 100
num_files = 10

# Read the input dataset
input_file_path = r"online_transactions_dataset.csv"
input_reader = pd.read_csv(input_file_path, chunksize=chunk_size)

# Generate and save 10 CSV files
for i in range(num_files):
    # Read the next chunk
    df_chunk = next(input_reader)

    # Save to a new CSV file with headers
    output_file_path = f"test_statement_{i + 1}.csv"
    df_chunk.to_csv(output_file_path, index=False)

