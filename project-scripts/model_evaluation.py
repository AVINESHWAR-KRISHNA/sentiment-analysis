from sklearn.metrics import accuracy_score, classification_report
def evaluate_models(models, X_test, y_test):
    try:
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'report': report
            }
        
        return results
    except Exception as e:
        print(f"An error occurred: {e}")
        return None