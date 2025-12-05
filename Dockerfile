# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies first for better layer caching.
# When building from NCWP/ as context, requirements.txt is in the context root.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r /workspace/requirements.txt

# Copy project into /workspace/NCWP
COPY . /workspace/NCWP
WORKDIR /workspace

# Expose Jupyter
EXPOSE 8888

# Default command: start a shell (use jupyter/cli inside)
CMD ["bash"]


