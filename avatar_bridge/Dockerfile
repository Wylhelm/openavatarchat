FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY avatar_bridge.py .
COPY .env .env

EXPOSE 9000

CMD ["python3", "avatar_bridge.py"]
