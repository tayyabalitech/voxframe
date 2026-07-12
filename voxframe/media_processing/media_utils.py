"""
VoxFrame Media Utility Suite
===========================
Handles the pipeline for fetching remote media files, analyzing video metrics,
extracting frame data, isolating audio tracks, and performing text-to-speech
transcriptions using the Groq Whisper API.
"""
import os
import re
import subprocess
import tempfile
import requests
from openai import OpenAI

from voxframe.config.cfg import AppConfig


# ── Remote Downloading ────────────────────────────────────────────────────────

def fetch_remote_video(url: str, storage_dir: str) -> str:
    """
    Downloads a video from a remote URL to a local folder in a chunked manner.

    Saves data to a localized path, raising an exception if download fails.
    """
    local_filename = os.path.join(storage_dir, "input_source.mp4")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(local_filename, "wb") as file_handle:
            for data_block in response.iter_content(chunk_size=16384):
                file_handle.write(data_block)
    return local_filename


# ── Video Duration Checking ────────────────────────────────────────────────────

def get_video_length(file_path: str) -> float:
    """Utilizes ffprobe to determine the duration of the media file in seconds."""
    ffprobe_arguments = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    process_result = subprocess.run(
        ffprobe_arguments,
        capture_output=True,
        text=True,
        check=True,
    )
    return float(process_result.stdout.strip())


# ── Frame Sampling Metrics ────────────────────────────────────────────────────

def determine_frame_count(length_in_sec: float, manual_count: int = 0) -> int:
    """
    Computes an appropriate keyframe count dynamically based on duration.

    Shorter clips require fewer samples; longer clips scale up.
    If a manual count override (>0) is supplied, it is returned directly.
    """
    if manual_count > 0:
        return manual_count
    
    if length_in_sec < 30.0:
        return 6
    elif length_in_sec < 60.0:
        return 8
    else:
        return 10


def _extract_scene_change_timestamps(video_source: str, scene_threshold: float = 0.18) -> list[float]:
    """Returns timestamps where ffmpeg detects a notable scene change."""
    ffmpeg_arguments = [
        "ffmpeg",
        "-hide_banner",
        "-i", video_source,
        "-vf", f"select='gt(scene\\,{scene_threshold})',showinfo",
        "-f", "null",
        "-",
    ]
    process_result = subprocess.run(ffmpeg_arguments, capture_output=True, text=True)
    combined_output = f"{process_result.stdout}\n{process_result.stderr}"

    timestamps: list[float] = []
    for match in re.finditer(r"pts_time:(\d+(?:\.\d+)?)", combined_output):
        timestamps.append(float(match.group(1)))

    unique_timestamps = sorted({round(timestamp, 2) for timestamp in timestamps})
    return unique_timestamps


def _build_sampling_timestamps(duration: float, count: int, scene_timestamps: list[float]) -> list[float]:
    """Combines uniform coverage and scene-change coverage into a compact sample set."""
    padding = duration * 0.05
    effective_range = max(duration - (2.0 * padding), 0.1)

    base_timestamps = [
        padding + (effective_range * idx / max(count - 1, 1))
        for idx in range(count)
    ]

    candidate_pool = sorted({
        round(timestamp, 2)
        for timestamp in base_timestamps + scene_timestamps
        if padding <= timestamp <= (duration - padding)
    })

    if len(candidate_pool) <= count:
        return candidate_pool

    selected: list[float] = []
    anchor_points = [candidate_pool[0], candidate_pool[len(candidate_pool) // 2], candidate_pool[-1]]
    for anchor in anchor_points:
        if anchor not in selected:
            selected.append(anchor)
        if len(selected) >= count:
            break

    while len(selected) < count:
        best_timestamp = None
        best_distance = -1.0
        for candidate in candidate_pool:
            if candidate in selected:
                continue
            nearest_distance = min(abs(candidate - existing) for existing in selected) if selected else float("inf")
            if nearest_distance > best_distance:
                best_distance = nearest_distance
                best_timestamp = candidate

        if best_timestamp is None:
            break
        selected.append(best_timestamp)

    return sorted(selected[:count])


def extract_video_keyframes(video_source: str, output_directory: str, count: int) -> list[str]:
    """
    Samples keyframes chronological from the video using FFmpeg.

    Combines uniform timeline sampling with scene-change candidates to improve coverage,
    then rescales images to 640px wide to optimize network usage.
    """
    clip_duration = get_video_length(video_source)
    scene_timestamps = _extract_scene_change_timestamps(video_source)
    timestamps = _build_sampling_timestamps(clip_duration, count, scene_timestamps)

    sampled_paths: list[str] = []
    for index, time_offset in enumerate(timestamps):
        output_file_name = os.path.join(output_directory, f"frame_sample_{index:02d}.jpg")
        ffmpeg_arguments = [
            "ffmpeg",
            "-y",
            "-ss", f"{time_offset:.4f}",
            "-i", video_source,
            "-frames:v", "1",
            "-q:v", "2",
            "-vf", "scale=640:-1",
            output_file_name,
        ]
        subprocess.run(ffmpeg_arguments, capture_output=True, check=True)
        sampled_paths.append(output_file_name)
        
    return sampled_paths


# ── Audio Track Extraction ────────────────────────────────────────────────────

def demux_audio_track(video_path: str, output_dir: str) -> str:
    """
    Strips the audio track out of the video and encodes it into a 16kHz mono WAV file.

    Returns an empty string if there is no audio track present or on subprocess failure.
    """
    output_audio_path = os.path.join(output_dir, "isolated_speech.wav")
    ffmpeg_arguments = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        output_audio_path,
    ]
    process_run = subprocess.run(ffmpeg_arguments, capture_output=True, text=True)
    return output_audio_path if process_run.returncode == 0 else ""


# ── Text transcription ────────────────────────────────────────────────────────

def speech_to_text_inference(audio_file: str) -> str:
    """
    Transcribes audio track file content utilizing the Groq ASR endpoint.

    Gracefully catches exceptions and checks for suspiciously short outputs to
    avoid Whisper hallucinations on silent/background noise clips.
    """
    if not audio_file or not os.path.exists(audio_file):
        return ""
    if not AppConfig.GROQ_KEY:
        return ""

    try:
        api_client = OpenAI(api_key=AppConfig.GROQ_KEY, base_url=AppConfig.GROQ_URL)
        with open(audio_file, "rb") as opened_file:
            api_response = api_client.audio.transcriptions.create(
                model=AppConfig.GROQ_MODEL,
                file=opened_file,
            )
        transcribed_text = (api_response.text or "").strip()
        
        # Discard transcripts with fewer than 3 words (common Whisper silence hallucination)
        if len(transcribed_text.split()) < 3:
            return ""
            
        return transcribed_text
    except Exception as error_msg:
        print(f"  [media_utils] Speech transcription skipped: {error_msg}")
        return ""


# ── Media Pipeline Coordination ───────────────────────────────────────────────

def run_media_extraction(url: str) -> dict:
    """
    Executes the complete media preprocessing workflow on a target URL.

    Downloads video, extracts dynamic keyframes, strips audio, and transcribes speech.
    Returns collected assets and temporary directory path for future manual cleanup.
    """
    temp_storage_dir = tempfile.mkdtemp(prefix="vf_media_")
    
    try:
        video_local_path = fetch_remote_video(url, temp_storage_dir)
        duration_seconds = get_video_length(video_local_path)
        
        needed_keyframes = determine_frame_count(duration_seconds, manual_count=AppConfig.KEYFRAME_LIMIT)
        extracted_frames = extract_video_keyframes(video_local_path, temp_storage_dir, needed_keyframes)
        
        audio_local_path = demux_audio_track(video_local_path, temp_storage_dir)
        text_transcript = speech_to_text_inference(audio_local_path)
        
        return {
            "frames": extracted_frames,
            "speech": text_transcript,
            "duration": duration_seconds,
            "_tmp": temp_storage_dir,
        }
    except Exception as processing_error:
        # Cleanup temporary files immediately if initial generation fails
        if os.path.exists(temp_storage_dir):
            import shutil
            shutil.rmtree(temp_storage_dir, ignore_errors=True)
        raise processing_error
