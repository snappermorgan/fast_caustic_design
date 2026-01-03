# Use a multi-stage build to keep the final image clean
FROM ubuntu:22.04 as builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
# We need cmake, g++, and the libraries: libpng, libjpeg, libceres, libsuitesparse
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libpng-dev \
    libjpeg-dev \
    libceres-dev \
    libsuitesparse-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
WORKDIR /app
COPY . .

# Build the application
# We use the flags from the README/CMakeLists suitable for linux
RUN mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    && make caustic_design -j$(nproc)

# --- Runtime Stage ---
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies for C++ app and Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libpng16-16 \
    libjpeg8 \
    libceres2 \
    libsuitesparse-dev \
    libgoogle-glog0v5 \
    libgflags2.2 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
WORKDIR /app
COPY requirements.txt .
# Use --break-system-packages because we are in a container and it's fine for now, 
# or we could use venv but simpler here.
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy build artifacts
COPY --from=builder /app/build/caustic_design /app/apps/build/caustic_design
COPY --from=builder /app/data /app/data
COPY --from=builder /app/tests /app/tests

# Copy the wrapper script
COPY cloud_runner.py /app/cloud_runner.py

# Ensure dynamic libraries are found
ENV LD_LIBRARY_PATH=/usr/local/lib

# Set entrypoint
# The command line arguments will be passed to this entrypoint
ENTRYPOINT ["python3", "cloud_runner.py"]
