import logging

def classify_establishment(name, description):
    categories_keywords = {
        'hotel': ['hotel', 'hoteis', 'business', 'executive', 'inn'],
        'pousada': ['pousada', 'charmosa', 'estilo', 'romântica'],
        'hostel/albergue': ['hostel', 'albergue', 'mochileiros', 'hostels'],
        'resort': ['resort', 'spa', 'all-inclusive', 'hotel resort'],
        'hotel fazenda': ['fazenda', 'rural', 'campo', 'eco', 'chale'],
        'flat/apart hotel': ['flat', 'apart hotel', 'apart-hotel' 'apartamento', 'residencial', 'suite']
    }

    name = name.lower()
    
    if description is None:
        description = ""
    else:
        description = description.lower()

    for category, keywords in categories_keywords.items():
        for keyword in keywords:
            if keyword in name:
                logging.info(f"{name} classificado como {category} com base no nome.")
                return category
    
    for category, keywords in categories_keywords.items():
        for keyword in keywords:
            if keyword in description:
                logging.info(f"{name} classificado como {category} com base na descrição.")
                return category
    
    logging.warning(f"{name} não classificado.")
    return 'não classificado'

def classify_all(df):
    logging.info("Iniciando a classificação de todos os estabelecimentos.")
    df['category'] = df.apply(lambda row: classify_establishment(row['name'], row['description']), axis=1)
    logging.info("Classificação concluída.")
    return df
