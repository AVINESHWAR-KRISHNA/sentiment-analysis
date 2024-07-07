from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from model_io import save_model, load_model
    
def train_models(X_train, y_train, save_path):
    try:
        models = {
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(probability=True),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

        for name, model in models.items():
            model_file_path = os.path.join(save_path, f'{name}.joblib')
            if os.path.exists(model_file_path):
                existing_model = load_model(model_file_path)
                existing_model.fit(X_train, y_train)
                save_model(existing_model, model_file_path)
                models[name] = existing_model  # Update models dict with retrained model
                print(f"Existing model '{name}' retrained.")
            else:
                model.fit(X_train, y_train)
                save_model(model, model_file_path)
                print(f"New model '{name}' trained and saved.")
        return models
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None