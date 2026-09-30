# Base image ko upgrade kiya gaya hai taaki naye packages aasani se install ho sakein
FROM python:3.11-slim

# Container ke andar working directory set karein
WORKDIR /app

# Sabse pehle requirements.txt copy karein taaki Docker cache ka fayda mile
COPY requirements.txt .

# Pip ko update karein aur dependencies install karein
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Baaki saara project code (serve.py, static folder, mlflow.db) copy karein
COPY . .

# Port 8000 open karein
EXPOSE 8000

# Application start karne ki command
CMD ["python", "serve.py"]