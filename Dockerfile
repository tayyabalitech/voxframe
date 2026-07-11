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


# Environment configurations
ENV FIREWORKS_API_KEY=fw_QLKdYmY6h4YsoZogznsT5j
ENV FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ENV GROQ_API_KEY=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
ENV GROQ_BASE_URL=https://api.groq.com/openai/v1
ENV GROQ_WHISPER_MODEL=whisper-large-v3

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