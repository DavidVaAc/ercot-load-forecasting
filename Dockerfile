# 1. Usamos una imagen oficial de Python ligera como base
FROM python:3.11-slim

# 2. Establecemos la carpeta de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalamos herramientas de Linux necesarias para LightGBM (compilación y paralelismo)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiamos primero el requirements.txt para aprovechar la caché de Docker
COPY requirements.txt .

# 5. Instalamos las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiamos todo el resto del proyecto (código, carpeta data con el modelo, etc.)
COPY . .

# 7. Exponemos el puerto estándar que usa Streamlit
EXPOSE 8501

# 8. Comando definitivo que arrancará la app cuando el contenedor se encienda
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]