import logging
from joblib import load
import os

# Caminhos para o modelo e vetorizador
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/classification_model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "../../models/tfidf_vectorizer.joblib")

try:
    # Verifique se os arquivos realmente existem
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vetorizador não encontrado em {VECTORIZER_PATH}")
    
    classification_model = load(MODEL_PATH)
    tfidf_vectorizer = load(VECTORIZER_PATH)
    logging.info("Modelo e vetorizador carregados com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar o modelo ou vetorizador: {e}")
    classification_model, tfidf_vectorizer = None, None

def classify_establishment_ml(name, description):
    """
    Classifica um estabelecimento utilizando o modelo treinado.
    """
    if not classification_model or not tfidf_vectorizer:
        raise ValueError("Modelo ou vetorizador não foram carregados corretamente.")
    
    # Preparar texto combinado
    text = f"{name} {description}"
    
    # Transformar o texto em vetores TF-IDF
    text_tfidf = tfidf_vectorizer.transform([text])
    
    # Fazer a previsão
    category = classification_model.predict(text_tfidf)[0]
    logging.info(f"Estabelecimento classificado como: {category}")
    return category
