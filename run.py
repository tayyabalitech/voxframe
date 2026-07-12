"""
VoxFrame Orchestration Runner
=============================
Main entry point script. Reads input tasks, executes the processing engine
concurrently, and saves structured narrative outputs to the target results file.
"""
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from voxframe.config.cfg import AppConfig
from voxframe.media_processing.media_utils import run_media_extraction
from voxframe.engines.author import synthesize_narratives


# ── Formatting Helpers ────────────────────────────────────────────────────────

def get_empty_captions_dict(error_reason: str) -> dict:
    """Yields a well-formed dictionary with error explanations for all target styles."""
    failure_message = f"Narration pipeline unavailable ({error_reason})."
    return {style: failure_message for style in AppConfig.TARGET_STYLES}


# ── Per-clip Pipeline Execution ────────────────────────────────────────────────

def execute_single_task(task: dict) -> dict:
    """
    Runs the complete processing pipeline for a single video task entry.

    Downloads video, extracts frame and speech content, composts styled captions,
    and returns a formatted dictionary. Catches all errors internally, falling back
    to empty captions to preserve task lists in outputs.
    """
    task_id = task.get("task_id", "?")
    video_url = task.get("video_url", "")
    temp_dir_path = None
    start_time = time.monotonic()

    try:
        print(f"\n{'-' * 60}")
        print(f"[{task_id}] Processing URL: {video_url}")
        print(f"{'-' * 60}")

        # Step 1: Pre-process media assets
        media_assets = run_media_extraction(video_url)
        temp_dir_path = media_assets["_tmp"]

        # Step 2: Display speech transcription preview
        speech_text = media_assets["speech"]
        if speech_text:
            text_preview = speech_text[:70] + ("..." if len(speech_text) > 70 else "")
            print(f"  [speech] Isolated {len(speech_text.split())} words | {text_preview}")
        else:
            print("  [speech] No voice data transcribed.")

        # Step 3: Call composition engine
        narratives = synthesize_narratives(media_assets["frames"], speech_text)

        execution_duration = time.monotonic() - start_time
        print(f"\n[{task_id}] Completed in {execution_duration:.1f}s")
        for style in AppConfig.TARGET_STYLES:
            caption_text = getattr(narratives, style)
            print(f"  {style:<22} {caption_text[:70]}...")

        return {
            "task_id": task_id,
            "captions": {
                "formal": narratives.formal,
                "sarcastic": narratives.sarcastic,
                "humorous_tech": narratives.humorous_tech,
                "humorous_non_tech": narratives.humorous_non_tech,
            },
        }

    except Exception as error:
        execution_duration = time.monotonic() - start_time
        print(f"[{task_id}] Process failed after {execution_duration:.1f}s: {error}")
        traceback.print_exc()
        return {"task_id": task_id, "captions": get_empty_captions_dict(str(error)[:120])}

    finally:
        # Guarantee that temporary files are cleared from the disk
        if temp_dir_path and os.path.exists(temp_dir_path):
            shutil.rmtree(temp_dir_path, ignore_errors=True)


def execute_task_with_timeout(task: dict, timeout_seconds: int) -> dict:
    """Executes execute_single_task using a dedicated thread pool to enforce a timeout."""
    task_id = task.get("task_id", "?")
    with ThreadPoolExecutor(max_workers=1) as single_thread_pool:
        future_result = single_thread_pool.submit(execute_single_task, task)
        try:
            return future_result.result(timeout=timeout_seconds)
        except FutureTimeout:
            print(f"[{task_id}] Timeout exceeded ({timeout_seconds}s)")
            return {"task_id": task_id, "captions": get_empty_captions_dict("process timeout")}


# ── Execution Entrypoint ───────────────────────────────────────────────────────

def main() -> None:
    AppConfig.validate_configuration()

    with open(AppConfig.TASKS_INPUT, encoding="utf-8") as file_handle:
        tasks: list[dict] = json.load(file_handle)

    print(f"VoxFrame Narration Engine | Loaded {len(tasks)} task(s)")
    print(
        f"  vision_model={AppConfig.VISION_MODEL.split('/')[-1]}"
        f"  text_model={AppConfig.TEXT_MODEL.split('/')[-1]}"
        f"  concurrency={AppConfig.CONCURRENT_LIMIT}"
        f"  timeout={AppConfig.PER_CLIP_TIMEOUT}s"
        f"  refine_threshold={AppConfig.EVAL_THRESHOLD}"
    )

    run_start_time = time.monotonic()
    results: list[dict | None] = [None] * len(tasks)

    # Submit tasks concurrently to the pool
    task_futures = []
    with ThreadPoolExecutor(max_workers=AppConfig.CONCURRENT_LIMIT) as thread_pool:
        for index, task in enumerate(tasks):
            future_object = thread_pool.submit(
                execute_task_with_timeout, task, AppConfig.PER_CLIP_TIMEOUT
            )
            task_futures.append((index, future_object))

        # Collect and index results as threads complete
        for idx, fut in task_futures:
            try:
                results[idx] = fut.result()
            except Exception as exc:
                tid = tasks[idx].get("task_id", "?")
                print(f"[{tid}] Executor error occurred: {exc}")
                results[idx] = {
                    "task_id": tid,
                    "captions": get_empty_captions_dict(f"executor error: {exc}"),
                }

    output_dir = os.path.dirname(AppConfig.RESULTS_OUTPUT)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(AppConfig.RESULTS_OUTPUT, "w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, indent=2, ensure_ascii=False)

    total_duration = time.monotonic() - run_start_time
    print(f"\n{'=' * 60}")
    print(f"Output File     : {AppConfig.RESULTS_OUTPUT}")
    print(f"Processed Tasks : {len(results)} | Total execution time: {total_duration:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as fatal_error:
        print(f"Fatal error encountered: {fatal_error}")
        traceback.print_exc()
        sys.exit(1)
