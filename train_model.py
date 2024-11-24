import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from collections import Counter
from joblib import dump

categories_keywords = {
    'hotel': ['hotel', 'hoteis', 'business', 'executive', 'inn'],
    'pousada': ['pousada', 'charme'],
    'hostel/albergue': ['hostel', 'albergue', 'mochileiros', 'hostels'],
    'resort': ['resort', 'spa', 'all-inclusive', 'hotel resort'],
    'hotel fazenda': ['fazenda', 'rural', 'campo', 'eco', 'chale'],
    'flat/apart hotel': ['flat', 'apart hotel', 'apart-hotel', 'apartamento', 'residencial', 'suite']
}

def add_keyword_score(data, categories_keywords):
    def calculate_score(row, keywords_dict):
        scores = {cat: sum(word in row.lower() for word in words) 
                  for cat, words in keywords_dict.items()}
        return max(scores, key=scores.get)
    
    data["keyword_score"] = data["description"].apply(lambda x: calculate_score(x, categories_keywords))
    return data

# Balanceamento de dados com ADASYN
def balance_with_adasyn(X, y):
    adasyn = ADASYN(random_state=42, n_neighbors=1)  # Reduzido para 1 vizinho
    class_counts = Counter(y)
    
    # Verificar se todas as classes têm amostras suficientes para o número de vizinhos
    valid_classes = [class_label for class_label, count in class_counts.items() if count > 1]  # A classe deve ter mais de 1 amostra
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]
    
    # Balanceamento com ADASYN
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    # Se houver classes com poucas amostras, pode-se usar RandomOverSampler para balanceamento
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_resampled, y_resampled)
    
    return X_resampled, y_resampled

def train_model(data):
    data = add_keyword_score(data, categories_keywords)
    data["text"] = data["name"] + " " + data["description"]
    X = data["text"]
    y = data["category"]
    
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    X_tfidf = tfidf.fit_transform(X)
    
    X_resampled, y_resampled = balance_with_adasyn(X_tfidf, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )
    
    classifier = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Avaliação
    y_pred = classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Validação cruzada
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
