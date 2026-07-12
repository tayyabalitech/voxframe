"""
VoxFrame Application Configuration
==================================
Loads configuration secrets, limits, and behavior patterns from environment variables.
Supports automatic local configuration via standard dot-env files.
"""
import os
from dotenv import load_dotenv

# Automatically load the .env file if it exists at the project root level
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))


class AppConfig:
    # ── Inference Services Configuration ──────────────────────────────────────
    AIMLAPI_KEY: str = os.environ.get("AIMLAPI_KEY", "")
    AIMLAPI_URL: str = os.environ.get("AIMLAPI_BASE_URL", "https://api.aimlapi.com/v1")
    VISION_MODEL: str = os.environ.get("AIMLAPI_VISION_MODEL", "google/gemini-2.5-pro")
    TEXT_MODEL: str = os.environ.get("AIMLAPI_TEXT_MODEL", "google/gemini-2.5-pro")
    GRADER_MODEL: str = os.environ.get("AIMLAPI_GRADER_MODEL", "google/gemini-2.5-flash")
    REFINEMENT_ENABLED: bool = int(os.environ.get("REFINEMENT_ENABLED", "0")) == 1

    # ── Transcription Settings ────────────────────────────────────────────────
    GROQ_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GROQ_URL: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_MODEL: str = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")

    # ── Input/Output File Paths ────────────────────────────────────────────────
    TASKS_INPUT: str = os.environ.get("INPUT_PATH", "/input/tasks.json")
    RESULTS_OUTPUT: str = os.environ.get("OUTPUT_PATH", "/output/results.json")

    # ── Video Processing Parameters ──────────────────────────────────────────
    KEYFRAME_LIMIT: int = int(os.environ.get("NUM_KEYFRAMES", "0"))  # 0 indicates adaptive sampling
    PER_CLIP_TIMEOUT: int = int(os.environ.get("PER_CLIP_TIMEOUT_S", "300"))
    CONCURRENT_LIMIT: int = int(os.environ.get("MAX_CONCURRENT_CLIPS", "1"))
    JSON_RETRY_ATTEMPTS: int = int(os.environ.get("JSON_RETRY_ATTEMPTS", "3"))

    MIN_WORDS: int = int(os.environ.get("CAPTION_MIN_WORDS", "8"))
    MAX_WORDS: int = int(os.environ.get("CAPTION_MAX_WORDS", "70"))

    # ── Quality Refinement Rubrics ────────────────────────────────────────────
    EVAL_THRESHOLD: float = float(os.environ.get("SCORE_THRESHOLD", "0.85"))
    MAX_REGEN_TRIES: int = int(os.environ.get("WEAK_STYLE_CANDIDATES", "3"))

    # ── Target Formats ────────────────────────────────────────────────────────
    TARGET_STYLES: tuple[str, ...] = (
        "formal",
        "sarcastic",
        "humorous_tech",
        "humorous_non_tech",
    )

    @classmethod
    def validate_configuration(cls) -> None:
        """Confirms that required secret API keys are correctly set in the environment."""
        if not cls.AIMLAPI_KEY:
            raise RuntimeError(
                "Missing required configuration: AIMLAPI_KEY is not defined. "
                "Ensure it is set as an environment variable or present in a local .env file."
            )
        if not cls.VISION_MODEL:
            raise RuntimeError("Missing required configuration: AIMLAPI_VISION_MODEL is not defined.")
        if not cls.TEXT_MODEL:
            raise RuntimeError("Missing required configuration: AIMLAPI_TEXT_MODEL is not defined.")
        if not cls.GRADER_MODEL:
            raise RuntimeError("Missing required configuration: AIMLAPI_GRADER_MODEL is not defined.")
