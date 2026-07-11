FROM python:3.11-slim

LABEL description="VoxFrame Narration Synthesis Container"

# Prepare and install ffmpeg for video frame extraction
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pre-install dependencies to utilize Docker build cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Load core modules and entrypoint
COPY voxframe/ ./voxframe/
COPY web_dashboard/ ./web_dashboard/
COPY run.py .

# Instantiate external volume mounts
RUN mkdir -p /input /output

# Expose the web dashboard port
EXPOSE 7860

# Run the CLI engine by default (required for lablab.ai evaluation)
CMD ["python", "run.py"]