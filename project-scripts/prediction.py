import os
import pandas as pd
from conf import PREDICTION_PATH, MODELS_DIR, OUTPUT_DIR
from data_preparation import preprocess_data
from feature_extraction import extract_features
from model_io import load_model

def load_models(model_names):
    try:
        models = {}
        for name in model_names:
            model_path = os.path.join(MODELS_DIR, f'{name}.joblib')
            if os.path.exists(model_path):
                models[name] = load_model(model_path)
            else:
                raise FileNotFoundError(f"Model {name} not found at {model_path}")
        return models
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def predict_and_append(data_file, model_names):
    try:
        data = pd.read_csv(data_file)
        data = preprocess_data(data)
        
        X = data['text']
        _, X_tfidf = extract_features(X, X)
        
        models = load_models(model_names)
        
        if models is None:
            print("No models loaded. Exiting prediction.")
            return None
        
        for name, model in models.items():
            predictions = model.predict(X_tfidf)
            data[f'{name}_prediction'] = predictions
        
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    model_names = ['Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest']
    result_data = predict_and_append(PREDICTION_PATH, model_names)

    if result_data is not None:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        output_file_path = os.path.join(OUTPUT_DIR, 'predictions_with_models.csv')
        result_data.to_csv(output_file_path, index=False)
        print(f"Predictions saved to {output_file_path}")
    else:
        print("No predictions were made.")
