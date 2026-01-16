from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import logging
from deep_translator import GoogleTranslator

# Configuración de logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Sentiment Analysis API")

# Modelo global
model = None

#Traduccion
def traducir_a_ingles(texto: str) -> str:
    try:
        translator = GoogleTranslator(source="auto", target="en")
        return translator.translate(texto)
    except Exception as e:
        logger.error(f"Error en traducción: {str(e)}")
        return texto

    try:
        texto_traducido = traducir_a_ingles(entrada.texto)
        texto_limpio = limpiar_texto(texto_traducido)
        prediccion = modelo.predict([texto_limpio])
        sentimiento_id = int(prediccion[0])
        sentimiento = mapa_sentimientos[sentimiento_id]

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



