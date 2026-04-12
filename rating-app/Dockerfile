FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn jinja2==3.1.4 python-multipart
COPY server.py .
COPY templates/ templates/
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
