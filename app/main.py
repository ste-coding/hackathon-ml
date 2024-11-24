from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.utils.classification import classify_establishment
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

app = FastAPI(
    title="Classificador de Estabelecimentos Hoteleiros",
    description="API para classificar estabelecimentos em categorias como hotel, pousada, resort, etc.",
    version="1.0.0"
)

class Establishment(BaseModel):
    name: str
    description: str

@app.post("/classify", response_model=dict)
async def classify(establishment: Establishment):
    """
    Classifica um estabelecimento com base no nome e descrição.
    """
    try:
        category = classify_establishment(establishment.name, establishment.description)
        return {"category": category}
    except Exception as e:
        logging.error(f"Erro ao classificar: {e}")
        raise HTTPException(status_code=500, detail="Erro interno no servidor")
