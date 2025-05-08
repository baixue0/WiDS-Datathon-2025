# Use nvidia/cuda as a base image to enable GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install essential packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    && apt-get clean

# Set up Python environment
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    openpyxl \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Copy the script and any other necessary files
COPY config/SETTINGS.json /app/config/SETTINGS.json
COPY autoencoder.py /app/autoencoder.py
COPY prepare_data.py /app/prepare_data.py
COPY train.py /app/train.py
COPY predict.py /app/predict.py
COPY predict_new.py /app/predict_new.py
COPY run.sh /app/run.sh

# Set the working directory
WORKDIR /app

# Define the entry point
ENTRYPOINT ["sh", "run.sh"]