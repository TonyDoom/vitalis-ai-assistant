FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit>=1.40,<2.0 \
    requests>=2.32,<3.0

COPY frontend/app.py /app/app.py

EXPOSE 8501

CMD [
    "streamlit",
    "run",
    "app.py",
    "--server.address=0.0.0.0",
    "--server.port=8501",
    "--server.headless=true"
]
