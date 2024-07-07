from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(X_train, X_test):
    try:
        vectorizer = TfidfVectorizer(max_features=6000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
