# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and source code
COPY requirements.txt ./
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables (optional, for Streamlit)
ENV PYTHONUNBUFFERED=1

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]