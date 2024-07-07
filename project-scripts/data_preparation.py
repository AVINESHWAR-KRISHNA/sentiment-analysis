import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def preprocess_data(data):
    try:
        data = data.dropna()
        data['text'] = data['text'].str.lower()  # Convert text to lowercase
        return data
    except KeyError:
        print("The 'text' column is missing in the DataFrame.")
        return None

def split_data(data, test_size=0.8, random_state=42):
    try:
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    except KeyError:
        print("The 'text' or 'label' column is missing in the DataFrame.")
        return None
