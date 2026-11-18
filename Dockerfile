FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Separate layer so script dependencies don't invalidate the one above.
COPY requirements-scripts.txt .
RUN pip install --no-cache-dir -r requirements-scripts.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
