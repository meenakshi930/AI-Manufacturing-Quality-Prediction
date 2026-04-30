FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/backend

EXPOSE 5000

CMD ["python", "backend/src/api/main.py"]
