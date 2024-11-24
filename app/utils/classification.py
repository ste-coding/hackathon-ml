import logging
import unicodedata

def normalize_text(text):
    """
    Normaliza o texto para facilitar a comparação, removendo acentos e caracteres especiais.
    """
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text

def classify_by_keywords(text, categories_keywords):
    """
    Classifica um texto com base em palavras-chave de categorias fornecidas.
    """
    for category, keywords in categories_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return category
    return None

def classify_establishment(name, description):
    """
    Classifica um estabelecimento com base no nome e na descrição.
    """
    categories_keywords = {
        'hotel': ['hotel', 'hoteis', 'business', 'executive', 'inn'],
        'pousada': ['pousada', 'charme'],
        'hostel/albergue': ['hostel', 'albergue', 'mochileiros', 'hostels'],
        'resort': ['resort', 'spa', 'all-inclusive', 'hotel resort'],
        'hotel fazenda': ['fazenda', 'rural', 'campo', 'eco', 'chale'],
        'flat/apart hotel': ['flat', 'apart hotel', 'apart-hotel', 'apartamento', 'residencial', 'suite']
    }

    normalized_name = normalize_text(name)
    normalized_description = normalize_text(description)

    category = classify_by_keywords(normalized_name, categories_keywords)
    if category:
        logging.info(f"{name} classificado como {category} com base no nome.")
        return category

    category = classify_by_keywords(normalized_description, categories_keywords)
    if category:
        logging.info(f"{name} classificado como {category} com base na descrição.")
        return category

    logging.warning(f"{name} não classificado.")
    return 'não classificado'
