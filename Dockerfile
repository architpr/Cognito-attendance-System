FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libv4l-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
# We use dlib-bin for faster/safer installation, but we installed cmake just in case
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user . .

# Expose the port Hugging Face Spaces uses
EXPOSE 7860

# Command to run the application using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "--workers", "2", "app:app"]
