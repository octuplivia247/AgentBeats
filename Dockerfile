FROM python:3.11-slim

WORKDIR /app

COPY . /app
RUN pip install -e .

EXPOSE 8080

ENTRYPOINT ["python", "main.py", "green"]