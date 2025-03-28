FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r docker_requirements.txt

CMD ["python", "app.py"]