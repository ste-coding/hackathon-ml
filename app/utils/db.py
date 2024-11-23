import pymysql
import logging
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def connect_to_db():
    try:
        host = os.getenv("DB_HOST")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD")
        database = os.getenv("DB_NAME")
        
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4"
        )
        logging.info("Conexão com o banco de dados estabelecida com sucesso.")
        return connection
    
    except pymysql.MySQLError as e:
        logging.error(f"Erro ao conectar ao banco de dados: {e}")
        raise

def fetch_data():
    conn = connect_to_db()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    query = """
    SELECT id, name, description
    FROM Hotels
    WHERE name IS NOT NULL
    """
    
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        logging.info(f"{len(result)} registros encontrados.")
        
        df = pd.DataFrame(result)
        return df
    except pymysql.MySQLError as e:
        logging.error(f"Erro ao buscar dados: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def update_category_in_db(df):
    conn = connect_to_db()
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE Hotels ADD COLUMN IF NOT EXISTS category VARCHAR(255)")
        logging.info("Coluna 'category' verificada/criada com sucesso.")
        
        for index, row in df.iterrows():
            cursor.execute("""
                UPDATE Hotels
                SET category = %s
                WHERE id = %s
            """, (row['category'], row['id']))
        
        conn.commit()
        logging.info(f"{len(df)} registros atualizados com sucesso.")
    
    except pymysql.MySQLError as e:
        logging.error(f"Erro ao atualizar o banco de dados: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
