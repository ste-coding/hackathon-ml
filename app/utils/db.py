
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

def connect_to_db():
    
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    database = os.getenv("DB_NAME")

    return pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4"
    )

def fetch_data():
    conn = connect_to_db()

    cursor = conn.cursor(pymysql.cursors.DictCursor)

    query = """
    SELECT id, name, description
    FROM Hotels
    WHERE name IS NOT NULL
    """
    cursor.execute(query)
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    import pandas as pd
    df = pd.DataFrame(result)
    return df

def update_category_in_db(df):
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="123456",
        database="hotelsdb",
        charset="utf8mb4"
    )
    cursor = conn.cursor()

    # Adicionar a coluna 'category' à tabela, caso ela não exista
    try:
        cursor.execute("ALTER TABLE Hotels ADD COLUMN category VARCHAR(255)")
    except pymysql.MySQLError as e:
        print(f"Erro ao adicionar a coluna: {e}")
    
    for index, row in df.iterrows():
        cursor.execute("""
            UPDATE Hotels
            SET category = %s
            WHERE id = %s
        """, (row['category'], row['id']))

    conn.commit()
    cursor.close()
    conn.close()
