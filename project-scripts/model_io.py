import joblib

def save_model(model, file_path):
    try:
        joblib.dump(model, file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def load_model(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")