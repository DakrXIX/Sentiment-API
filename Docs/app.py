from fastapi import FastAPI 
from pydantic import BaseModel
from fastapi.responses import FileResponse 
from fastapi.staticfiles import StaticFiles
import joblib
from fastapi import HTTPException 
import re#Importa el módulo de expresiones regulares para limpiar y procesar texto
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from fastapi import Query
from database import engine
from database import SessionLocal
from database import Base
from models import Prediccion
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
from deep_translator import GoogleTranslator 

#Crear tablas en la base de datos, esto nos sirve para crear estadisticas
Base.metadata.create_all(bind=engine)

# -------------------
# Logging
# -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment_api")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "logs/app.log",
    maxBytes=1_000_000, 
    backupCount=3 
)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

handler.setFormatter(formatter)#Aplica el formato definido al manejador de logs
logger.addHandler(handler)#Agrega el manejador al registrador para que los mensajes se escriban en el archivo con rotación
# -------------------
# App
# -------------------
app = FastAPI()

# -------------------
# Static files
# -------------------
app.mount("/static", StaticFiles(directory="static"), name="static")#Monta el directorio "static" para servir archivos estáticos (HTML, CSS, JS) en la ruta "/static"

# -------------------
# Load model
# -------------------
modelo = joblib.load("modelo_sentimientos.pkl")#Carga el modelo de machine learning previamente entrenado desde el archivo "modelo_sentimientos.pkl"

# ------------------
#Traductor
# ------------------
def traducir_a_ingles(texto: str) -> str:
    try:
        translator = GoogleTranslator(source="auto", target="en")
        return translator.translate(texto)
    except Exception as e:
        logger.error(f"Error en traducción: {str(e)}")
        return texto  # fallback: usa texto original si falla
# -------------------
# Text cleaning
# -------------------
def limpiar_texto(texto: str) -> str:#Función para limpiar y preprocesar el texto de entrada
    texto = texto.lower()#Convierte el texto a minúsculas
    texto = re.sub(r"http\S+", "", texto)# Elimina URLs
    texto = re.sub(r"@\w+", "", texto)# Elimina menciones de usuarios
    texto = re.sub(r"#", "", texto)# Elimina el símbolo de hashtag
    texto = re.sub(r"[^a-z\s]", "", texto)# Elimina caracteres no alfabéticos
    texto = re.sub(r"\s+", " ", texto).strip()# Elimina espacios extras y recorta espacios al inicio y final
    return texto#Retorna el texto limpio

# -------------------
# Sentiment map
# -------------------
mapa_sentimientos = {#Mapea los IDs de sentimiento a sus descripciones correspondientes
    1: "Negativo",
    2: "Neutral",
    3: "Positivo"
}

# -------------------
# Input schema
# -------------------
class TextoEntrada(BaseModel):#Define el esquema de entrada para la predicción de sentimientos
    texto: str

# -------------------
# Frontend
# -------------------
@app.get("/")#Ruta raíz que sirve el archivo HTML principal del frontend
def frontend():
    return FileResponse("static/index.html")#Retorna el archivo HTML ubicado en "static/index.html"
# -------------------
#stats endpoint (DB)
@app.get("/stats")# Define el endpoint "/stats" que recibe una solicitud GET para obtener estadísticas de predicciones almacenadas en la base de datos
def obtener_estadisticas(limit: int = Query(100, ge=1)):
    db: Session = SessionLocal()

    registros = (
        db.query(Prediccion)
        .order_by(Prediccion.fecha.desc())
        .limit(limit)
        .all()
    )

    db.close()

    total = len(registros)
    if total == 0:
        return {"mensaje": "No hay datos para analizar"}

    conteo = {"Negativo": 0, "Neutral": 0, "Positivo": 0}

    for r in registros:
        conteo[r.sentimiento] += 1

    return {
        "total": total,
        "conteo": conteo,
        "porcentaje": {
            k: round(v * 100 / total, 2) for k, v in conteo.items()
        }
    }

# -------------------
# Prediction endpoint
# -------------------
@app.post("/predict")
def predecir_sentimiento(entrada: TextoEntrada):#Define el endpoint "/predict" que recibe una solicitud POST con texto para predecir su sentimiento
# Validar texto vacío
    if not entrada.texto.strip():
        logger.warning("Texto vacío recibido")
        raise HTTPException(
            status_code=400,
            detail="El texto no puede estar vacío"
        )
# Limpieza del texto y predicción 
    try:
        texto_traducido = traducir_a_ingles(entrada.texto)
        texto_limpio = limpiar_texto(texto_traducido)
        prediccion = modelo.predict([texto_limpio])
        sentimiento_id = int(prediccion[0])
        sentimiento = mapa_sentimientos[sentimiento_id]

#Predicción de probabilidades
        probabilidades = modelo.predict_proba([texto_limpio])[0]
        clases = modelo.classes_
#Mapeo de probabilidades a etiquetas
        probabilidades_dict = {
            mapa_sentimientos[int(clase)]: round(float(prob), 4)
            for clase, prob in zip(modelo.classes_, probabilidades)
        }
  #  Guardar en DB
        db = SessionLocal()
        registro = Prediccion(
            texto=entrada.texto,# Para almacenar el texto original sin limpiar, es mejor para auditoría y análisis futuros
            sentimiento_id=sentimiento_id,
            sentimiento=sentimiento,
            fecha=datetime.now(timezone.utc)
        )
        db.add(registro)
        db.commit()
        db.close()

        logger.info(
            f"Predicción exitosa | Texto: '{entrada.texto[:30]}' | Resultado: {sentimiento}"
        )
        
# Retornar resultado de sentimiento y también de probabilidades
        return {
            "texto_original": entrada.texto,
            "texto_traducido": texto_traducido,
            "sentimiento_id": sentimiento_id,
            "sentimiento": sentimiento,
            "probabilidades": probabilidades_dict
        }
# Manejo de errores en la predicción
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno al procesar el texto"
        )
    
