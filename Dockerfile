# Base image lightweight Python 3.9
FROM python:3.9-slim

# Container ke andar working directory set karein
WORKDIR /app

# Sabse pehle requirements.txt copy karein taaki Docker cache ka fayda mile
COPY requirements.txt .

# Dependencies install karein
RUN pip install --no-cache-dir -r requirements.txt

# Baaki saara project code (serve.py, static folder, mlflow.db) copy karein
COPY . .

# Port 8000 open karein
EXPOSE 8000

# Application start karne ki command
CMD ["python", "serve.py"]