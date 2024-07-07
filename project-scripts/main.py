import os
from conf import DATASET_PATH, MODELS_DIR, TRAIN_MODELS
from data_preparation import load_data, preprocess_data, split_data
from feature_extraction import extract_features
from model_training import train_models
from model_evaluation import evaluate_models
from model_io import load_model

def main():
    
    data = load_data(DATASET_PATH)
    data = preprocess_data(data)
    
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)

    try:
        model_names = ['Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest']
        models = {}
        if TRAIN_MODELS:
            models = train_models(X_train_tfidf, y_train, save_path=MODELS_DIR)
        else:
            for name in model_names:
                model_path = os.path.join(MODELS_DIR, f'{name}.joblib')
                if os.path.exists(model_path):
                    models[name] = load_model(model_path)
                else:
                    print(f"Model {name} not found. Training all models.")
                    models = train_models(X_train_tfidf, y_train, save_path=MODELS_DIR)
                    break
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    results = evaluate_models(models, X_test_tfidf, y_test)

    for name, result in results.items():
        print(f"Model: {name}")
        print(f"Accuracy: {result['accuracy']}")
        print(f"Classification Report:\n{result['report']}")
        print("-" * 60)

if __name__ == "__main__":
    main()