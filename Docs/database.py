from sqlalchemy import create_engine #Crea el motor de conexión a la base de datos
from sqlalchemy.orm import sessionmaker #Crea una fábrica de sesiones para interactuar con la base de datos
from sqlalchemy.orm import declarative_base #Define la clase base para los modelos ORM

# URL de la base de datos (SQLite local)
DATABASE_URL = "sqlite:///./sentiment.db"

# Crea el motor de conexión
engine = create_engine( 
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Crea una fábrica de sesiones
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Clase base para los modelos ORM
Base = declarative_base()
