# Optimized Dockerfile

# Use a slim base image
FROM python:3.10-slim

# --- 1. System dependencies (run as root) ---
# Install required C toolchain (build-essential) and audio libs (libsndfile1, ffmpeg).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- 2. Create non-root user ---
# Define the UID/GID build arguments
ARG UID=10001
ARG GID=10001
RUN groupadd -g $GID appgroup && \
    useradd -m -u $UID -g $GID -s /bin/bash appuser

# --- 3. Create application directory (still root) ---
# Create the app directory plus audit mount point and hand ownership to the non-root user.
# chown on empty directories is effectively instant.
RUN mkdir -p /app/audit && \
    chown -R $UID:$GID /app

# --- 4. Switch to non-root user ---
# Everything below runs as 'appuser'
USER appuser
WORKDIR /app

# --- 5. Install Python dependencies (as appuser) ---
# Create a virtualenv owned by 'appuser'
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Copy dependency manifest first to leverage Docker cache
COPY --chown=$UID:$GID requirements.txt .

# Install dependencies inside the virtualenv.
# This is the slow step (py-solc-x, etc.) but everything remains owned by appuser, avoiding a final chown.
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# --- 6. Copy application code (still appuser) ---
# Copy the rest of the source tree.
# chown is explicit here even though we're already appuser.
COPY --chown=$UID:$GID . .

# Create a placeholder for the contract info file that will be mounted
# Avoids runtime errors if the mount is missing
RUN touch /app/contract_info.json

# --- 7. Environment and ports ---
# Expose application and metrics ports
EXPOSE 8080
EXPOSE 9000

# Runtime command (using gunicorn)
# Update 'server.api:app' if your entry point changes
# Use the absolute path to the venv executable
CMD ["/app/venv/bin/gunicorn", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "240", "--bind", "0.0.0.0:8080", "server.api:app"]