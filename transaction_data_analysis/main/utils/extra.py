# utils / data_forge.py

import pandas as pd


def file_preview(file_uploaded, size, enforce_float=True):
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
        "size": size,
        "path": file_uploaded.file_path.path,
    }
