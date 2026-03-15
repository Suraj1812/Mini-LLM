FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

ARG TORCH_VERSION=2.8.0

COPY requirements-runtime.txt requirements-runtime.txt

RUN pip install --upgrade pip \
    && pip install -r requirements-runtime.txt \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VERSION}

COPY . .

ENV MINI_LLM_HOST=0.0.0.0

EXPOSE 8000

CMD ["python", "server.py"]
