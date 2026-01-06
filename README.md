# Sentiment-API
MVP de análisis de sentimiento que integra Data Science (Python, TF-IDF + Logistic Regression) con una API REST en Java mediante un contrato JSON.
# 📊 Sentiment Analysis API – Machine Learning & FastAPI

## 🧠 Descripción del Proyecto

Este proyecto implementa un sistema de **análisis de sentimientos** basado en técnicas de **Machine Learning**, capaz de clasificar textos en tres categorías:

- **Negativo (1)**
- **Neutral (2)**
- **Positivo (3)**

El modelo fue entrenado con datos reales de tweets relacionados con vacunas COVID-19 y desplegado como una **API REST** utilizando **FastAPI**.

---

## 🎯 Objetivos

- Construir un modelo de clasificación de texto
- Optimizar el modelo mediante validación cruzada
- Exponer el modelo a través de una API
- Permitir su consumo desde aplicaciones externas
- Demostrar el flujo completo de ML en producción

---

## 🏗️ Arquitectura del Sistema

Texto del usuario

↓

Limpieza de texto

↓

TF-IDF Vectorizer

↓

Logistic Regression

↓

API FastAPI (/predict)

↓

Respuesta JSON

## 🤖 Modelo de Machine Learning

- **Algoritmo:** Logistic Regression
- **Vectorización:** TF-IDF
- **Optimización:** GridSearchCV
- **Métrica:** F1-score macro
- **Implementación:** Pipeline de scikit-learn

---

## ⚙️ Tecnologías Utilizadas

- [Python](https://www.python.org/) 3.12
- [FastAPI](https://fastapi.tiangolo.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [Uvicorn](https://www.uvicorn.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [Pytest](https://docs.pytest.org/)

---

## 📡 Endpoints Principales

### `POST /predict`

Recibe un texto y retorna el sentimiento detectado.

**Ejemplo de Request:**
```json
{
  "texto": "La vacuna es muy efectiva"
}

**Ejemplo de Response:**
{
  "sentimiento_id": 3,
  "sentimiento": "Positivo"
}

🧪 Pruebas y Validación

El proyecto incluye tests automáticos que validan:

Funcionamiento del endpoint /predict

Manejo de errores (texto vacío)

Respuesta en formato JSON

Ejecutar tests:

pytest

🌐 Despliegue (pendiente)


Este proyecto esta siendo desarrollado de manera colaborativa por:

- Cesar Araya  
- Cesar Londono  
- Gloria Gutiérrez  
- Marcos Perez  
- Victor Araya  
- Yober Cieza  
- Carlos Gaston Fernandez  
- José Luis Planes  
- Lester Hernandez  
- Wilmer Acosta  



  
