FROM python:3.11-slim

# Install system dependencies required for rasterio / GDAL
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL include paths
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

COPY . /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install -r requirements.txt

EXPOSE 10000

CMD ["streamlit", "run", "AppV2.py", "--server.port=10000", "--server.address=0.0.0.0"]
