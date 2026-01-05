from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# cargar el modelo al arrancar
model = joblib.load("modelo_sentimientos.pkl")

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment")
def sentiment(req: SentimentRequest):

    text = req.text.strip()

    if len(text) < 5:
        raise HTTPException(
            status_code=400,
            detail="El campo 'text' es obligatorio y debe tener al menos 5 caracteres"
        )

    try:
        pred = model.predict([text])[0]
        prob = model.predict_proba([text]).max()

        return {
            "prevision": "Positivo" if pred == 1 else "Negativo",
            "probabilidad": float(prob)
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No se pudo procesar el texto"
        )

