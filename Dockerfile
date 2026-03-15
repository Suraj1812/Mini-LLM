FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MINI_LLM_HOST=0.0.0.0
ENV MINI_LLM_PORT=8000

EXPOSE 8000

CMD ["python", "server.py"]
