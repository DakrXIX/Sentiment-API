 📊 Sentiment Analysis API – Machine Learning & FastAPI

## 🧠 Descripción del Proyecto

Este proyecto implementa un sistema de **análisis de sentimientos** basado en técnicas de **Machine Learning**, capaz de clasificar textos en tres categorías:

- **Negativo (1)**
- **Neutral (2)**
- **Positivo (3)**

El usuario ingresa un texto desde una interfaz web y el sistema determina automáticamente el sentimiento asociado, entregando además probabilidades por clase para mayor transparencia del resultado.

🌍 Soporte multilingüe (ES / PT)

Aunque el modelo de Machine Learning fue entrenado en inglés, la API acepta textos en español y portugués.
Para lograrlo, el sistema incorpora una capa de traducción automática a inglés antes de la inferencia, permitiendo reutilizar el modelo sin necesidad de reentrenamiento.

Este diseño equilibra:

eficiencia técnica

menor costo computacional

facilidad de uso para usuarios finales

Además del análisis en tiempo real, el sistema almacena cada predicción en una base de datos, guardando el texto original del usuario junto con el resultado. Esto permite:

Trazabilidad

Análisis histórico

Estadísticas agregadas

Futuras extensiones analíticas

El modelo fue entrenado con datos reales de tweets relacionados con vacunas COVID-19 y desplegado como una API REST utilizando FastAPI, integrando Machine Learning, backend y persistencia de datos en una solución completa.
---

🎯 Objetivos del Proyecto

Construir un modelo de clasificación de texto usando NLP

Optimizar el modelo mediante validación cruzada

Exponer el modelo a través de una API REST

Permitir su consumo desde aplicaciones web

Demostrar un flujo completo de ML en producción
(Machine Learning + API + Base de Datos + Frontend)

Servir como base para proyectos de:

feedback de clientes

encuestas de satisfacción

monitoreo de opiniones

👥 Público Objetivo

Personas del área tecnológica
(desarrollo, data, TI, ciencia de datos)

Personas no técnicas
(negocio, gestión, usuarios finales)

La documentación y el diseño del sistema están pensados para ser comprensibles por ambos perfiles, explicando tanto el qué como el por qué de cada componente.
---

## 🏗️ Arquitectura del Sistema


Texto del usuario (ES/PT)

↓

Traducción automática a inglés

↓

Limpieza y normalización de texto

↓

Pipeline de Machine Learning (TF-IDF + Clasificador)

↓

API FastAPI (/predict)

↓

Respuesta JSON + Persistencia en DB

Para mayor comprensión de nuestro proyecto en personas que no son del área de la tecnología presentamos el siguiente Diagrama de Flujo:

🔄 Flujo Paso a Paso

1️⃣ Persona usuaria 👤

Ingresa un texto en el formulario web.

Ejemplo:

“Es muy buena y efectiva”

2️⃣ Página Web 🌐

Recibe el texto del usuario.

No realiza ningún análisis.

Envía el texto a la API.

3️⃣ JavaScript (script.js) 🔁

Lee el texto ingresado.

Envía una solicitud POST a la API usando fetch.

Recibe la respuesta

Muestra:

Sentimiento final

Texto traducido al inglés

Probabilidades por clase

4️⃣ API (app.py) ⚙️

Recibe el texto original desde el frontend.

Traduce automáticamente a inglés

Limpia el texto traducido

Ejecuta el modelo de Machine Learning

Guarda el resultado en la base de datos

5️⃣ Modelo de Machine Learning 🤖

Analiza el texto utilizando técnicas de NLP y lo guarda en la base de datos.

Clasifica el sentimiento como:

Positivo

Negativo

Neutral 

Calcula probabilidades por clase

6️⃣ Respuesta 📦

La API devuelve una respuesta en formato JSON:

{
  "texto_original": "Es muy buena y efectiva",
  
  "texto_traducido": "It is very good and effective",
  
  "sentimiento_id": 3,
  
  "sentimiento": "Positivo",
  
  "probabilidades": {
  
    "Negativo": 0.02,
    
    "Neutral": 0.08,
    
    "Positivo": 0.90
  }
}

**Ejemplos**

## Ejemplos

### Positivo (Español)
![Positivo Español](https://github.com/user-attachments/assets/a4f1d027-e2d5-4e92-b919-ea325b21b9c2)

### Positivo (Portugués)
![Positivo Portugués](https://github.com/user-attachments/assets/3f7eac53-827c-453d-849e-03ef1f1b58c2)

### Neutro
![Neutro](https://github.com/user-attachments/assets/02ac9c33-b34b-4aeb-8b4a-8fe75e393d36)

### Negativo
![Negativo](https://github.com/user-attachments/assets/5a24e672-885b-49f9-a5dd-ca9a237776c4)

---

## 🤖 Modelo de Machine Learning

Algoritmo: Logistic Regression (pipeline)

Vectorización: TF-IDF

Optimización: GridSearchCV

Métrica principal: F1-score macro

Implementación: Pipeline de scikit-learn

Idioma de entrenamiento: Inglés

---

🛠️ Tecnologías Utilizadas

Frontend: HTML, JavaScript

Backend: Python (Flask o FastAPI)

Machine Learning: Scikit-learn

NLP: TF-IDF, Logistic Regression / Naive Bayes

Traducción: deep-translator (GoogleTranslator)

Base de Datos: SQLite

ORM: SQLAlchemy

Comunicación: API REST (JSON)

Logging: logging + RotatingFileHandler

  ## 🧪 Dataset

- **Fuente:** Kaggle
- **Nombre:** COVID-19 Vaccine Tweets with Sentiment
- **Formato:** CSV
- **Codificación:** latin1

🔗 Enlace al dataset:  
https://www.kaggle.com/datasets/gpreda/covid19-vaccine-tweets-with-sentiment

---

## 📡 Endpoints Principales

### `POST /predict`

Recibe un texto y retorna el sentimiento detectado, el texto traducido y las probabilidades.

GET /stats

Entrega estadísticas agregadas de las predicciones almacenadas en la base de datos.

🧪 Pruebas y Validación

El proyecto incluye tests automáticos que validan:

Funcionamiento del endpoint /predict

Manejo de errores (texto vacío)

Respuestas JSON estructuradas

Persistencia correcta en base de datos


Este proyecto está siendo desarrollado de manera colaborativa por:

- Carlos Gastón Fernández 
- Cesar Araya  
- Cesar Londono  
- Gloria Gutiérrez  
- José Luis Planes  
- Lester Hernández 
- Marcos Pérez  
- Víctor Araya  
- Yober Cieza  
- Wilmer Acosta


  
