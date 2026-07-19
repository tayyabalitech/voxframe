"""
VoxFrame FastAPI Demonstration Server
====================================
Bridges the web-based frontend dashboard with the backend processing engines.
"""
import sys
import os
import shutil
import tempfile

# Insert workspace root to sys.path to allow absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voxframe.config.cfg import AppConfig
from voxframe.media_processing.media_utils import (
    extract_video_keyframes,
    demux_audio_track,
    speech_to_text_inference,
    fetch_remote_video,
)
from voxframe.engines.author import synthesize_narratives


app = FastAPI(title="VoxFrame Web Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    video_url: str


def _process_video_and_generate(video_path: str, temp_dir: str) -> dict:
    """Helper pipeline to run keyframe extraction, speech transcription, and captioning."""
    # Pull 6 frames for the live dashboard demonstration to minimize latency
    sampled_frames = extract_video_keyframes(video_path, temp_dir, count=6)
    audio_path = demux_audio_track(video_path, temp_dir)
    transcribed_speech = speech_to_text_inference(audio_path)
    
    narrative_captions = synthesize_narratives(sampled_frames, transcribed_speech)
    
    return {
        "success": True,
        "transcript": transcribed_speech,
        "captions": {
            "formal": narrative_captions.formal,
            "sarcastic": narrative_captions.sarcastic,
            "humorous_tech": narrative_captions.humorous_tech,
            "humorous_non_tech": narrative_captions.humorous_non_tech,
        },
    }


@app.post("/api/generate")
def generate(req: GenerateRequest):
    temp_dir = tempfile.mkdtemp(prefix="api_url_")
    try:
        local_path = fetch_remote_video(req.video_url, temp_dir)
        return _process_video_and_generate(local_path, temp_dir)
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp(prefix="api_upload_")
    try:
        local_path = os.path.join(temp_dir, file.filename or "upload.mp4")
        with open(local_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return _process_video_and_generate(local_path, temp_dir)
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# Mount static frontend assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")