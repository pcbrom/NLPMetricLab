import pandas as pd
import re

def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")

def detect_han_characters(df):
    han_regex = re.compile(r'[\u4e00-\u9fff]')
    return df.applymap(lambda x: bool(han_regex.search(str(x)))).any().any()
