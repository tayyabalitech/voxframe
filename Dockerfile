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
# Secret keys are injected at runtime via `docker run -e ...` or an env file.
ENV AIMLAPI_BASE_URL=https://api.aimlapi.com/v1
ENV AIMLAPI_VISION_MODEL=google/gemini-2.5-pro
ENV AIMLAPI_TEXT_MODEL=google/gemini-2.5-pro
ENV GROQ_BASE_URL=https://api.groq.com/openai/v1
ENV GROQ_WHISPER_MODEL=whisper-large-v3
ENV REFINEMENT_ENABLED=1
ENV AIMLAPI_GRADER_MODEL=google/gemini-2.5-flash
ENV WEAK_STYLE_CANDIDATES=1

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