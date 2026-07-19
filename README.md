# VoxFrame

VoxFrame is a containerized multimodal video-captioning pipeline built for the AMD Developer Hackathon ACT II Track 2 challenge. It downloads video clips, extracts adaptive keyframes, optionally transcribes audio, grounds each scene with a vision-capable model, generates four stylistically distinct captions, audits them locally, and can refine weaker outputs with a second-pass grader.

The project is designed to run in a reproducible Docker environment and to produce structured output for automated evaluation.

## Overview

The pipeline processes each task from a JSON input file and writes a structured result set to an output JSON file. Each task produces four captions:

- formal
- sarcastic
- humorous_tech
- humorous_non_tech

## How the pipeline works

1. Read task definitions from the input file.
2. Download the target video clip.
3. Extract adaptive keyframes and optional speech transcription.
4. Generate a visual scene summary from the sampled frames.
5. Produce styled captions grounded in that scene summary.
6. Run local rule-based quality checks and targeted repairs.
7. Optionally evaluate outputs with a grader and refine weak styles.
8. Write the final results to the output file.

## Project structure

- [run.py](run.py) — main entrypoint for batch execution
- [voxframe/config/cfg.py](voxframe/config/cfg.py) — runtime configuration and environment variable handling
- [voxframe/config/defs.py](voxframe/config/defs.py) — Pydantic schemas for captions and evaluation results
- [voxframe/media_processing/media_utils.py](voxframe/media_processing/media_utils.py) — video download, frame extraction, audio demuxing, and speech transcription
- [voxframe/engines/author.py](voxframe/engines/author.py) — narration generation, audit, repair, and refinement stages
- [voxframe/engines/grader.py](voxframe/engines/grader.py) — caption quality evaluation against the sampled frames
- [web_dashboard/server.py](web_dashboard/server.py) — optional local web demo
- [sample_inputs/tasks.json](sample_inputs/tasks.json) — example task list for local testing

## Requirements

The application requires:

- Python 3.11+
- Docker (recommended for evaluation runs)
- FFmpeg available in the runtime environment
- API access for AIMLAPI and Groq

Python dependencies are listed in [requirements.txt](requirements.txt).

## Configuration

Create a local environment file before running outside Docker:

```bash
cp .env.example .env
```

The current implementation uses AIMLAPI for the vision and grading workflows and Groq for optional speech transcription. The main environment variables are:

```env
AIMLAPI_KEY=your_aimlapi_key
AIMLAPI_BASE_URL=https://api.aimlapi.com/v1
AIMLAPI_VISION_MODEL=google/gemini-2.5-pro
AIMLAPI_TEXT_MODEL=google/gemini-2.5-pro
AIMLAPI_GRADER_MODEL=google/gemini-2.5-flash

GROQ_API_KEY=your_groq_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_WHISPER_MODEL=whisper-large-v3

REFINEMENT_ENABLED=1
PER_CLIP_TIMEOUT_S=300
MAX_CONCURRENT_CLIPS=1
JSON_RETRY_ATTEMPTS=3
SCORE_THRESHOLD=0.85
WEAK_STYLE_CANDIDATES=3
CAPTION_MIN_WORDS=8
CAPTION_MAX_WORDS=70
INPUT_PATH=/input/tasks.json
OUTPUT_PATH=/output/results.json
```

## Running locally

### Option 1: Docker (recommended)

Build the image:

```bash
docker build -t voxframe .
```

Run the evaluation workflow:

```bash
docker run -it --rm -p 7860:7860 \
  -v "$(pwd)/sample_inputs:/input" \
  -v "$(pwd)/output:/output" \
  --name voxframe voxframe
```

The container reads tasks from /input/tasks.json and writes results to /output/results.json.

### Option 2: Local Python execution

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the engine directly:

```bash
python run.py
```

This expects the input and output paths to be available in the environment or the default filesystem locations.

## Web dashboard

A lightweight local web interface is available through [web_dashboard/server.py](web_dashboard/server.py). It can be served with Uvicorn:

```bash
uvicorn web_dashboard.server:app --host 0.0.0.0 --port 7860
```

## Input and output format

### Input

The task file should contain a list of objects with the following structure:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://example.com/clip.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

### Output

Each task in the output file will include a task identifier and a captions object:

```json
{
  "task_id": "v1",
  "captions": {
    "formal": "...",
    "sarcastic": "...",
    "humorous_tech": "...",
    "humorous_non_tech": "..."
  }
}
```

## Notes on implementation

- The pipeline uses a staged approach for robustness: scene grounding, caption generation, audit and repair, and optional refinement.
- JSON parsing and retry logic are included to handle model output that is incomplete or formatted inconsistently.
- The default concurrency is intentionally low for stable evaluation runs.
- The web dashboard is optional and is primarily intended for local demonstrations.

## Deployment notes

The project is prepared for container-based evaluation and can be pushed to a container registry such as GitHub Container Registry. The Docker image entrypoint is configured to run the batch engine with Python.

## License

This repository is intended for the AMD Developer Hackathon ACT II Track 2 submission workflow.
