from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Sentiment Analysis API")

# Modelo global
model = None

# Esquema de entrada con validaciones
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Texto a analizar (mínimo 5 caracteres)")
    threshold: float = Field(0.6, ge=0.0, le=1.0, description="Umbral de confianza para clasificar")

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("modelo_sentimientos.pkl")
    logging.info("Modelo cargado correctamente")

@app.post("/sentiment")
def predict_sentiment(request: SentimentRequest):
    # Validación de entrada
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    # Predicción
    prediction = model.predict([text])[0]
    print("Prediction:", prediction)
    proba = model.predict_proba([text])[0]
    print("Probabilities:", proba)
    confidence = max(proba)
    print("Confidence:", confidence)
    # Aplicar umbral de confianza
    if confidence > 0.49 and confidence < 0.6:
        sentiment = "Neutro"
    else:
        sentiment = "Positivo" if prediction == 3 else "Negativo"

    # Logging
    logging.info(f"Texto: {text}, Predicción: {sentiment}, Confianza: {confidence:.2f}")

    return {
        "texto": text,
        "prevision": sentiment,
        "probabilidades": {
            "Positivo": float(proba[1]),
            "Negativo": float(proba[0])
        },
        "confianza": float(confidence),
        "umbral": request.threshold
    }
