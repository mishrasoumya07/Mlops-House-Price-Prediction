FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Pip upgrade zaroori hai build errors se bachne ke liye
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .
COPY static ./static

# Yeh dono add karna sabse important hai trained model load karne ke liye
COPY mlruns ./mlruns
COPY mlflow.db .

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]