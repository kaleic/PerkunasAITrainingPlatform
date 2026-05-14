FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY docs ./docs

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "kvserve.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
