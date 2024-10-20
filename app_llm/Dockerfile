FROM python:3.12.7-slim

WORKDIR /app_llm

COPY app_llm/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app_llm app_llm

EXPOSE 8000

CMD ["uvicorn", "app_llm.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

