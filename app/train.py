import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.db import fetch_data, update_category_in_db
from app.utils.classification import classify_all


def main():
    df = fetch_data()

    classified_df = classify_all(df)

    update_category_in_db(classified_df)

    print("Classificação concluída e dados atualizados no banco de dados.")

if __name__ == "__main__":
    main()

