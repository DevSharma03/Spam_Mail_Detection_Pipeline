FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system dependencies for NLTK + LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        curl \
        unzip \
        git \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up NLTK data directory
RUN mkdir -p /usr/local/nltk_data

# ✅ Download NLTK data explicitly using curl instead of nltk.downloader (more reliable in slim images)
RUN wget -q -P /usr/local/nltk_data/corpora/ \
    https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip && \
    wget -q -P /usr/local/nltk_data/corpora/ \
    https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip && \
    wget -q -P /usr/local/nltk_data/corpora/ \
    https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip && \
    unzip -qo /usr/local/nltk_data/corpora/stopwords.zip -d /usr/local/nltk_data/corpora/ && \
    unzip -qo /usr/local/nltk_data/corpora/wordnet.zip -d /usr/local/nltk_data/corpora/ && \
    unzip -qo /usr/local/nltk_data/corpora/omw-1.4.zip -d /usr/local/nltk_data/corpora/ && \
    rm /usr/local/nltk_data/corpora/*.zip

# Set environment variable for NLTK
ENV NLTK_DATA=/usr/local/nltk_data

# Expose FastAPI port
EXPOSE 9000

# Run app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
