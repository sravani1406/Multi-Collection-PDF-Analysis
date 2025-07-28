FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY . .

# Install Python dependencies (all should be pre-cached)
RUN pip install --no-cache-dir sentence-transformers==2.2.2 \
    && pip install --no-cache-dir PyMuPDF==1.23.4 \
    && pip install --no-cache-dir huggingface_hub==0.16.4

# Set offline mode (no internet access allowed during execution)
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

CMD ["python", "run_1b.py"]
