def classify_establishment(name, description):
    categories_keywords = {
        'hotel': ['hotel', 'hoteis', 'luxo', 'hospedagem'],
        'pousada': ['pousada', 'charmosa', 'estilo'],
        'hostel/albergue': ['hostel', 'albergue', 'mochileiros', 'hostels'],
        'resort': ['resort', 'praia', 'luxo', 'spa', 'all-inclusive'],
        'hotel fazenda': ['fazenda', 'rural', 'campo', 'eco'],
        'flat/apart hotel': ['flat', 'apart hotel', 'apartamento', 'residencial']
}

    name = name.lower()
    
    if description is None:
        description = ""
    else:
        description = description.lower() 

    for category, keywords in categories_keywords.items():
        for keyword in keywords:
            if keyword in name or keyword in description:
                return category
    
    return 'não classificado'

def classify_all(df):
    df['category'] = df.apply(lambda row: classify_establishment(row['name'], row['description']), axis=1)
    return df
