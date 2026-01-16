from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import logging
import csv
from datetime import datetime
import os
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# Configuración de logging con UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# FastAPI con encoding UTF-8 explícito
app = FastAPI(
    title="Sentiment Analysis API",
    description="API para análisis de sentimientos con soporte UTF-8 y traducción automática"
)

# Modelo global
model = None

# Esquema de entrada para 1 texto (SIN threshold)
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Texto a analizar (mínimo 5 caracteres)")
    
    class Config:
        json_encoders = {
            str: lambda v: v
        }

# Esquema de entrada para 1 texto con resultado esperado
class SentimentRequestWithExpected(BaseModel):
    text: str = Field(..., min_length=5, description="Texto a analizar (mínimo 5 caracteres)")
    expected_result: str = Field(..., description="Resultado esperado (Positivo, Negativo, Neutral)")
    
    class Config:
        json_encoders = {
            str: lambda v: v
        }

# Esquema para un texto con resultado esperado (bulk)
class TextWithExpectedResult(BaseModel):
    text: str = Field(..., description="Texto a analizar")
    expected_result: str = Field(..., description="Resultado esperado (Positivo, Negativo, Neutral)")
    
    class Config:
        json_encoders = {
            str: lambda v: v
        }

# Esquema para múltiples textos con resultados esperados
class BulkSentimentRequestWithExpected(BaseModel):
    texts: List[TextWithExpectedResult] = Field(..., min_items=1, description="Lista de textos con resultados esperados")
    
    class Config:
        json_encoders = {
            str: lambda v: v
        }

# Esquema antiguo (sin expected_result) - para mantener compatibilidad
class BulkSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Lista de textos a analizar")
    
    class Config:
        json_encoders = {
            str: lambda v: v
        }

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("modelo_sentimientos.pkl")
    logging.info("Modelo cargado correctamente")

def detect_language(text: str) -> str:
    """
    Detecta el idioma del texto usando langdetect.
    """
    try:
        lang = detect(text)
        return lang
    except LangDetectException as e:
        logging.warning(f"No se pudo detectar el idioma: {str(e)}")
        return "unknown"

def translate_to_english(text: str) -> dict:
    """
    Traduce el texto a inglés si no está en inglés.
    Retorna un diccionario con el texto original, texto traducido, y el idioma detectado.
    """
    try:
        # Detectar idioma
        detected_lang = detect_language(text)
        
        # Si ya está en inglés o no se pudo detectar, no traducir
        if detected_lang == 'en':
            logging.info(f"Texto ya está en inglés: {text[:50]}...")
            return {
                "original_text": text,
                "translated_text": text,
                "detected_language": detected_lang,
                "was_translated": False
            }
        
        if detected_lang == 'unknown':
            logging.warning(f"Idioma desconocido, usando texto original: {text[:50]}...")
            return {
                "original_text": text,
                "translated_text": text,
                "detected_language": detected_lang,
                "was_translated": False
            }
        
        # Traducir a inglés usando deep-translator
        translator = GoogleTranslator(source=detected_lang, target='en')
        translated_text = translator.translate(text)
        
        logging.info(f"Traducido de {detected_lang} a inglés: '{text[:50]}...' -> '{translated_text[:50]}...'")
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "detected_language": detected_lang,
            "was_translated": True
        }
    
    except Exception as e:
        logging.error(f"Error en traducción: {str(e)} - Usando texto original")
        # Si falla la traducción, usar el texto original
        return {
            "original_text": text,
            "translated_text": text,
            "detected_language": "error",
            "was_translated": False
        }

def normalize_sentiment(sentiment: str) -> str:
    """
    Normaliza el sentimiento para que coincida con el formato esperado.
    """
    sentiment_lower = sentiment.lower()
    
    if sentiment_lower in ['positivo', 'positive']:
        return 'Positivo'
    elif sentiment_lower in ['negativo', 'negative']:
        return 'Negativo'
    elif sentiment_lower in ['neutro', 'neutral']:
        return 'Neutral'
    else:
        return sentiment  # Devolver original si no coincide

@app.post("/sentiment")
def predict_sentiment(request: SentimentRequest):
    """
    Endpoint para analizar el sentimiento de UN SOLO texto.
    Detecta automáticamente el idioma y traduce a inglés si es necesario.
    """
    # Validación de entrada
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

    # Traducir a inglés
    translation_result = translate_to_english(text)
    text_for_prediction = translation_result["translated_text"]
    
    logging.info(f"Texto para predicción: {text_for_prediction}")
    
    # Predicción
    prediction = model.predict([text_for_prediction])[0]
    print(f"Prediction: {prediction}")
    proba = model.predict_proba([text_for_prediction])[0]
    print(f"Probabilities: {proba}")
    confidence = max(proba)
    print(f"Confidence: {confidence}")
    
    # Aplicar umbral de confianza
    if prediction == 3:
        sentiment = "Positivo"
    elif prediction == 2:
        sentiment = "Negativo"
    else: 
        sentiment = "Neutral" 

    # Logging
    logging.info(f"Texto original: {text}, Idioma: {translation_result['detected_language']}, "
                f"Traducido: {translation_result['was_translated']}, "
                f"Predicción: {sentiment}, Confianza: {confidence:.2f}")

    return {
        "texto_original": text,
        "texto_traducido": text_for_prediction,
        "idioma_detectado": translation_result["detected_language"],
        "fue_traducido": translation_result["was_translated"],
        "prevision": sentiment,
        "probabilidades": {
            "Positivo": float(proba[2]),
            "Negativo": float(proba[1]),
            "Neutral": float(proba[0])
        },
        "confianza": float(confidence)
    }

@app.post("/sentiment-with-expected")
def predict_sentiment_with_expected(request: SentimentRequestWithExpected):
    """
    Endpoint para analizar el sentimiento de UN SOLO texto CON resultado esperado.
    Detecta automáticamente el idioma y traduce a inglés si es necesario.
    Incluye comparación con el resultado esperado.
    """
    # Validación de entrada
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="El texto no puede estar vacío")
    
    expected = normalize_sentiment(request.expected_result)

    # Traducir a inglés
    translation_result = translate_to_english(text)
    text_for_prediction = translation_result["translated_text"]
    
    logging.info(f"Texto para predicción: {text_for_prediction}")
    
    # Predicción
    prediction = model.predict([text_for_prediction])[0]
    print(f"Prediction: {prediction}")
    proba = model.predict_proba([text_for_prediction])[0]
    print(f"Probabilities: {proba}")
    confidence = max(proba)
    print(f"Confidence: {confidence}")
    
    # Aplicar umbral de confianza
    if prediction == 3:
        sentiment = "Positivo"
    elif prediction == 2:
        sentiment = "Negativo"
    else: 
        sentiment = "Neutral"
    
    # Verificar si la predicción es correcta
    is_correct = (sentiment == expected)

    # Logging
    correctness = "✓" if is_correct else "✗"
    logging.info(f"Texto original: {text}, Idioma: {translation_result['detected_language']}, "
                f"Traducido: {translation_result['was_translated']}, "
                f"Esperado: {expected}, Predicción: {sentiment} {correctness}, "
                f"Confianza: {confidence:.2f}")

    return {
        "texto_original": text,
        "texto_traducido": text_for_prediction,
        "idioma_detectado": translation_result["detected_language"],
        "fue_traducido": translation_result["was_translated"],
        "expected_result": expected,
        "prevision": sentiment,
        "es_correcto": is_correct,
        "probabilidades": {
            "Positivo": float(proba[2]),
            "Negativo": float(proba[1]),
            "Neutral": float(proba[0])
        },
        "confianza": float(confidence)
    }

@app.post("/sentiment/bulk")
def predict_sentiment_bulk(request: BulkSentimentRequest):
    """Endpoint para compatibilidad con el formato antiguo (sin expected_result)"""
    results = []
    
    # Procesar cada texto
    for idx, text in enumerate(request.texts):
        text = text.strip()
        
        # Validar que el texto no esté vacío
        if not text or len(text) < 5:
            logging.warning(f"Texto #{idx+1} omitido (vacío o muy corto): {text}")
            continue
        
        try:
            # Traducir a inglés
            translation_result = translate_to_english(text)
            text_for_prediction = translation_result["translated_text"]
            
            logging.info(f"Procesando texto #{idx+1}: '{text[:50]}...' -> '{text_for_prediction[:50]}...'")
            
            # Predicción
            prediction = model.predict([text_for_prediction])[0]
            proba = model.predict_proba([text_for_prediction])[0]
            confidence = max(proba)
            
            # Aplicar umbral de confianza
            if prediction == 3:
                sentiment = "Positivo"
            elif prediction == 2:
                sentiment = "Negativo"
            else: 
                sentiment = "Neutral"
            
            # Agregar resultado
            result = {
                "texto_original": text,
                "texto_traducido": text_for_prediction,
                "idioma_detectado": translation_result["detected_language"],
                "fue_traducido": translation_result["was_translated"],
                "prevision": sentiment,
                "Positivo": float(proba[2]),
                "Negativo": float(proba[1]),
                "Neutral": float(proba[0]),
                "confianza": float(confidence)
            }
            results.append(result)
            
            # Logging
            logging.info(f"Texto #{idx+1}: Idioma: {translation_result['detected_language']} | "
                        f"Predicción: {sentiment} | Confianza: {confidence:.2f}")
        
        except Exception as e:
            logging.error(f"Error procesando texto #{idx+1}: {str(e)}")
            continue
    
    # Crear archivo CSV con encoding UTF-8 explícito
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_results_{timestamp}.csv"
    
    # Crear directorio si no existe
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    # Escribir CSV con UTF-8 y BOM para mejor compatibilidad con Excel
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['texto_original', 'texto_traducido', 'idioma_detectado', 'fue_traducido', 
                     'prevision', 'Positivo', 'Negativo', 'Neutral', 'confianza']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    logging.info(f"Resultados guardados en {filepath} ({len(results)} textos procesados)")
    
    return {
        "total_procesados": len(results),
        "archivo_generado": filepath,
        "resultados": results
    }

@app.post("/sentiment/bulk-with-expected")
def predict_sentiment_bulk_with_expected(request: BulkSentimentRequestWithExpected):
    """Endpoint que acepta textos con resultados esperados y genera CSV con accuracy"""
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    # Procesar cada texto
    for idx, item in enumerate(request.texts):
        text = item.text.strip()
        expected = normalize_sentiment(item.expected_result)
        
        # Validar que el texto no esté vacío
        if not text or len(text) < 5:
            logging.warning(f"Texto #{idx+1} omitido (vacío o muy corto): {text}")
            continue
        
        try:
            # Traducir a inglés
            translation_result = translate_to_english(text)
            text_for_prediction = translation_result["translated_text"]
            
            logging.info(f"Procesando texto #{idx+1}: '{text[:50]}...' -> '{text_for_prediction[:50]}...'")
            
            # Predicción
            prediction = model.predict([text_for_prediction])[0]
            proba = model.predict_proba([text_for_prediction])[0]
            confidence = max(proba)
            
            # Aplicar umbral de confianza
            if prediction == 3:
                sentiment = "Positivo"
            elif prediction == 2:
                sentiment = "Negativo"
            else: 
                sentiment = "Neutral"
            
            # Verificar si la predicción es correcta
            is_correct = (sentiment == expected)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Agregar resultado
            result = {
                "texto_original": text,
                "texto_traducido": text_for_prediction,
                "idioma_detectado": translation_result["detected_language"],
                "fue_traducido": translation_result["was_translated"],
                "expected_result": expected,
                "prevision": sentiment,
                "es_correcto": is_correct,
                "Positivo": float(proba[2]),
                "Negativo": float(proba[1]),
                "Neutral": float(proba[0]),
                "confianza": float(confidence)
            }
            results.append(result)
            
            # Logging
            correctness = "✓" if is_correct else "✗"
            logging.info(f"Texto #{idx+1}: Idioma: {translation_result['detected_language']} | "
                        f"Esperado: {expected} | Predicción: {sentiment} {correctness} | "
                        f"Confianza: {confidence:.2f}")
        
        except Exception as e:
            logging.error(f"Error procesando texto #{idx+1}: {str(e)}")
            continue
    
    # Calcular accuracy
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Crear archivo CSV con encoding UTF-8 explícito
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sentiment_results_with_expected_{timestamp}.csv"
    
    # Crear directorio si no existe
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    # Escribir CSV con UTF-8 y BOM para mejor compatibilidad con Excel
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['texto_original', 'texto_traducido', 'idioma_detectado', 'fue_traducido', 
                     'expected_result', 'prevision', 'es_correcto', 
                     'Positivo', 'Negativo', 'Neutral', 'confianza']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    logging.info(f"Resultados guardados en {filepath} ({len(results)} textos procesados)")
    logging.info(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    return {
        "total_procesados": len(results),
        "predicciones_correctas": correct_predictions,
        "accuracy_porcentaje": round(accuracy, 2),
        "archivo_generado": filepath,
        "resultados": results
    }

# Endpoint de prueba para verificar UTF-8 y traducción
@app.get("/test-utf8")
def test_utf8():
    test_strings = [
        "Español: ñáéíóú ¿¡",
        "Português: ãõçÃÕÇ",
        "Símbolos: €£¥©®™"
    ]
    return {
        "encoding": "UTF-8",
        "test_strings": test_strings,
        "message": "Si ves correctamente estos caracteres, UTF-8 funciona bien"
    }

@app.get("/test-translation")
def test_translation():
    """Endpoint para probar la traducción"""
    test_texts = [
        "La vacuna es muy efectiva",
        "A vacina é segura",
        "This vaccine works great",
        "Esta vacuna no sirve para nada",
        "Me siento muy feliz con los resultados"
    ]
    
    results = []
    for text in test_texts:
        translation_result = translate_to_english(text)
        results.append(translation_result)
    
    return {
        "message": "Prueba de traducción",
        "results": results
    }
