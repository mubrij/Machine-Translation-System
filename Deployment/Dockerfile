FROM --platform=linux/amd64 python:3.10.14-slim

WORKDIR /app

# Dependencies
COPY ./serve-requirements.txt .
RUN pip install -r serve-requirements.txt

# Trained model and definition with main script
COPY ./dyu_to_fr /app/dyu_to_fr
COPY ./main.py /app/main.py

# Set entrypoint
ENTRYPOINT ["python", "-m", "main"]