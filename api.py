from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Sentiment Analysis API")

model = None  # 🔑 NO cargar aquí

class SentimentRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("modelo_sentimientos.pkl")

@app.post("/sentiment")
def predict_sentiment(request: SentimentRequest):

    prediction = model.predict([request.text])[0]
    probability = model.predict_proba([request.text])[0].max()

    return {
        "prevision": "Positivo" if prediction == 1 else "Negativo",
        "probabilidad": float(probability)
    }


