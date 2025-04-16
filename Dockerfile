FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:5000/_stcore/health || exit 1

CMD ["python","app.py"]
