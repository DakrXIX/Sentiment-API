from sqlalchemy import Column, Integer, String, DateTime # Importa los tipos de columna necesarios para definir el modelo
from datetime import datetime, timezone # Importa datetime y timezone para manejar marcas de tiempo
from database import Base  # Base viene desde database.py

class Prediccion(Base): 
    __tablename__ = "predicciones"

    id = Column(Integer, primary_key=True, index=True)#Clave primaria única para cada registro
    texto = Column(String, nullable=False)
    sentimiento_id = Column(Integer, nullable=False)
    sentimiento = Column(String, nullable=False)
    fecha = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
