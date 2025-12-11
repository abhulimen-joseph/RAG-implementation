FROM python:3.11 AS slim

ENV  PYTHONBUFFERED=1

WORKDIR /app

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY RAG.py .
COPY 41488-30243-PB.pdf .

EXPOSE 8000

CMD ["uvicorn", "RAG:app", "--host:", "0.0.0.0", "--port", "8000"]
