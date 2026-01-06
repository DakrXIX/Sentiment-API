from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Sentiment Analysis API")

model = joblib.load("modelo_sentimientos.pkl")

class SentimentRequest(BaseModel):
    text: str

@app.post("/sentiment")
def predict_sentiment(request: SentimentRequest):

    prediction = model.predict([request.text])[0]
    probability = model.predict_proba([request.text])[0].max()

    return {
        "prevision": "Positivo" if prediction == 1 else "Negativo",
        "probabilidad": float(probability)
    }

