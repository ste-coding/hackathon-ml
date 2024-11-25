import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from joblib import dump

categories_keywords = {
    'hotel': ['hotel', 'hoteis', 'business', 'executive', 'inn'],
    'pousada': ['pousada', 'charme', 'romântico'],
    'hostel/albergue': ['hostel', 'albergue', 'mochileiros', 'hostels', 'dormitório', 'cama compartilhada'],
    'resort': ['resort', 'spa', 'all-inclusive', 'hotel resort', 'praia', 'piscina'],
    'hotel fazenda': ['fazenda', 'rural', 'campo', 'eco', 'chale', 'fazenda ecológica', 'hospedagem rural'],
    'flat/apart hotel': ['flat', 'apart hotel', 'apart-hotel', 'apartamento', 'residencial', 'suite']
}


def add_keyword_score(data, categories_keywords):
    def calculate_score(row, keywords_dict):
        scores = {cat: sum(word in row.lower() for word in words) 
                  for cat, words in keywords_dict.items()}
        return max(scores, key=scores.get)
    
    data["keyword_score"] = data["description"].apply(lambda x: calculate_score(x, categories_keywords))
    return data

def balance_with_smote(X, y):
    class_counts = Counter(y)
    
    valid_classes = [class_label for class_label, count in class_counts.items() if count > 1]
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]
    
    min_samples = min(class_counts.values())
    n_neighbors = max(min_samples - 1, 1)
    
    smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def train_model(data):
    data = add_keyword_score(data, categories_keywords)
    data["text"] = data["name"] + " " + data["description"]
    X = data["text"]
    y = data["category"]
    
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    X_tfidf = tfidf.fit_transform(X)
    
    X_resampled, y_resampled = balance_with_smote(X_tfidf, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )
    
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    cross_val_scores = cross_val_score(classifier, X_resampled, y_resampled, cv=5)
    print("Cross-validation scores:", cross_val_scores)
    print("Average cross-validation score:", cross_val_scores.mean())
    
    return classifier, tfidf

def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    print(f"Modelo salvo em {model_path}")
    print(f"Vetor TF-IDF salvo em {vectorizer_path}")

if __name__ == "__main__":
    data_path = "data/cleaned_classification_data.csv"
    model_path = "models/classification_model.joblib"
    vectorizer_path = "models/tfidf_vectorizer.joblib"
    
    data = pd.read_csv(data_path)
    
    model, tfidf = train_model(data)
    
    save_model_and_vectorizer(model, tfidf, model_path, vectorizer_path)
