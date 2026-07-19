"""
VoxFrame Narrative Composition Engine
====================================
Coordinates Stage A scene grounding and Stage B narrative style composition,
with constraint-based validation and targeted cleanup.
"""
import base64
import json
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from voxframe.config.cfg import AppConfig
from voxframe.config.defs import FrameDescription, GeneratedCaptions
from voxframe.engines.grader import assess_caption_quality


# ── Persona Spec Registry ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class StylePersona:
    """Immutable specification outlining the target behavior of a caption style."""
    key: str
    persona: str
    rules: tuple[str, ...]
    specimen: str      # A gold-standard sample
    anti_pattern: str  # What to explicitly avoid


PERSONA_REGISTRY: tuple[StylePersona, ...] = (
    StylePersona(
        key="formal",
        persona="a wire news agency reporter filing a clear, factual, objective scene report",
        rules=(
            "Write 1-2 complete sentences in objective, neutral third-person voice",
            "State only observable visual facts: who/what is present, what they are doing, where they are",
            "Include specific visual details: colors, count, clothing, environment, motion",
            "Absolutely no exclamation points, no metaphors, no slang, no opinions",
            "Precise wording only — avoid vague phrases like 'something' or 'a person'",
        ),
        specimen="A young professional woman is seated at a desktop computer workstation in a bright, modern open-plan office, focused intently on her screen.",
        anti_pattern="Unbelievable! This rider is absolutely flying down the mountain trail!",
    ),
    StylePersona(
        key="sarcastic",
        persona="a deadpan, unimpressed bystander who narrates the blindingly obvious with dry irony",
        rules=(
            "Write exactly ONE short sentence — punchy, dry, and effortlessly unimpressed",
            "Comment on the scene as if the event is the most unremarkable thing imaginable",
            "Acknowledge something obvious in the video as if it is a profound observation",
            "NEVER describe the scene literally — react to it sarcastically from the sidelines",
            "No exclamation marks, no LOL, no obvious joke setups — the humor is in the understatement",
        ),
        specimen="A kitten outdoors, clearly plotting something elaborate and fully confident it will succeed.",
        anti_pattern="OMG this is so funny, I am literally dying at this hilarious scene!",
    ),
    StylePersona(
        key="humorous_tech",
        persona="a senior engineer who turns one clearly visible object or action into a precise tech pun, then lands the joke with sharp technical wordplay",
        rules=(
            "Anchor the joke in exactly one clearly visible object, person, action, or scene detail from the video",
            "Use one concrete tech term that directly supports the visible anchor, not a generic coding phrase",
            "The wordplay must be clever and specific to THIS video — not just a generic tech sentence",
            "1 sentence preferred, 2 sentences maximum, punchy and witty",
            "Include at least one real software/hardware term such as deployment, pipeline, kernel, API, agent, loop, thread, cache, render, packet, or schema",
        ),
        specimen="Nature's annual deployment: all leaf nodes updated to yellow simultaneously, no breaking changes reported.",
        anti_pattern="This runner is moving fast, which requires high bandwidth calculation.",
    ),
    StylePersona(
        key="humorous_non_tech",
        persona="a witty friend sending a clever voice note about what they just saw — relatable, funny, zero jargon",
        rules=(
            "Write ONE sentence — conversational, relatable, and genuinely funny to a general audience",
            "Comment on the scene with the energy of a group chat message — casual and clever",
            "Draw a relatable parallel: what does this scene remind everyone of in real life?",
            "Absolutely no technical, developer, or programming words of any kind",
            "Avoid sounding like a formal description — this is banter, not a report",
        ),
        specimen="The trees got together and decided to put on a show, and honestly they are the only ones putting in any effort.",
        anti_pattern="His velocity calculation iterates recursively without hitting any base condition.",
    ),
)

PERSONA_MAP: dict[str, StylePersona] = {p.key: p for p in PERSONA_REGISTRY}


# ── Vocabulary Dictionaries for Auditing ─────────────────────────────────────

TECHNICAL_DICTIONARY: frozenset[str] = frozenset({
    "algorithm", "server", "code", "bug", "debug", "deploy", "bandwidth",
    "cpu", "gpu", "ram", "kernel", "loop", "function", "api", "database",
    "cache", "stack", "overflow", "compile", "runtime", "latency", "git",
    "commit", "merge", "pipeline", "neural", "model", "token", "prompt",
    "cloud", "container", "docker", "kubernetes", "microservice", "binary",
    "pixel", "render", "shader", "matrix", "tensor", "gradient", "epoch",
    "leetcode", "recursion", "async", "thread", "memory", "pointer",
    "syntax", "exception", "http", "tcp", "socket", "404", "503",
    "throughput", "latency", "packet", "hash", "queue", "mutex",
    "concurrency", "distributed", "scaling", "throttle", "endpoint",
    "payload", "schema", "inference", "training", "weights", "parameters",
    "optimizer", "loss", "rate-limit", "rate limit",
})

RESTRICTED_ENGINEERING_JARGON: frozenset[str] = frozenset({
    "algorithm", "kubernetes", "docker", "microservice", "tensor", "gradient",
    "epoch", "leetcode", "recursion", "async", "tcp", "socket", "shader",
    "kernel", "compile", "runtime", "api", "database", "binary", "pointer",
    "mutex", "concurrency", "payload", "schema", "inference", "optimizer",
    "weights", "parameters", "distributed",
})


# ── Formatting Helpers ────────────────────────────────────────────────────────

def image_to_base64_uri(file_path: str) -> str:
    """Encodes a local JPEG image into a base64 Data URI string."""
    with open(file_path, "rb") as image_file:
        encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_data}"


def find_balanced_json_object(text: str) -> str | None:
    """Finds the first balanced JSON object span in a piece of text."""
    start_index = text.find("{")
    while start_index != -1:
        depth = 0
        in_string = False
        escape_next = False
        for index in range(start_index, len(text)):
            character = text[index]
            if escape_next:
                escape_next = False
                continue
            if character == "\\":
                escape_next = True
                continue
            if character == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start_index:index + 1]
        start_index = text.find("{", start_index + 1)
    return None


PROSE_RESPONSE_PREFIXES: tuple[str, ...] = (
    "the user",
    "looking at",
    "i need to",
    "let me",
    "based on",
    "i will",
    "i'll",
    "sure,",
    "okay,",
    "here is",
    "here's",
    "analyzing",
    "to analyze",
)


def looks_like_prose_response(text: str) -> bool:
    """Returns True when the model echoed instructions instead of emitting JSON."""
    if not text or "{" not in text:
        return True
    normalized = text.strip().lower()
    return any(normalized.startswith(prefix) for prefix in PROSE_RESPONSE_PREFIXES)


SCENE_FIELD_KEYS: tuple[str, ...] = ("subject", "action", "setting", "mood", "details")


def close_truncated_json_object(text: str) -> str | None:
    """Attempts to close a JSON object that was cut off mid-stream."""
    start_index = text.find("{")
    if start_index == -1:
        return None

    fragment = text[start_index:]
    in_string = False
    escape_next = False
    depth = 0
    for character in fragment:
        if escape_next:
            escape_next = False
            continue
        if character == "\\":
            escape_next = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1

    repaired = fragment.rstrip()
    if in_string:
        repaired += '"'
    while depth > 0:
        repaired += "}"
        depth -= 1
    return repaired if repaired != fragment else None


def salvage_json_string_fields(text: str, field_keys: tuple[str, ...]) -> dict[str, str]:
    """Extracts string field values from complete or truncated JSON-like text."""
    fields: dict[str, str] = {}
    for key in field_keys:
        pattern = rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1).replace('\\"', '"').strip()
        if value:
            fields[key] = value
    return fields


def salvage_scene_fields_from_text(text: str) -> dict | None:
    """Builds a scene report dict from salvaged JSON fragments."""
    fields = salvage_json_string_fields(text, SCENE_FIELD_KEYS)
    if len(fields) < 3:
        return None

    if "mood" not in fields:
        anchor = fields.get("subject") or fields.get("setting") or fields.get("action") or "the scene"
        fields["mood"] = f"an everyday atmosphere visible around {anchor[:100]}"
    if "details" not in fields:
        fields["details"] = fields.get("setting") or fields.get("action") or fields.get("subject", "")
    if "action" not in fields:
        fields["action"] = "activity and motion visible across the sampled frames"
    if "setting" not in fields:
        fields["setting"] = "the surrounding environment visible behind the main subjects"
    if "subject" not in fields:
        fields["subject"] = fields.get("setting") or "the primary visible subjects in the clip"

    return fields


def salvage_caption_fields_from_text(text: str) -> dict | None:
    """Builds a caption dict from salvaged JSON fragments."""
    fields = salvage_json_string_fields(text, AppConfig.TARGET_STYLES)
    if len(fields) < len(AppConfig.TARGET_STYLES):
        return None
    return fields


def extract_json_object(model_output: str) -> dict:
    """Extracts and parses the first JSON object block from the text response."""
    if not model_output:
        raise ValueError("Received empty model output.")
    
    clean_text = re.sub(r"^```(?:json)?\s*", "", model_output.strip())
    clean_text = re.sub(r"\s*```$", "", clean_text)
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        balanced_json = find_balanced_json_object(clean_text)
        if balanced_json:
            try:
                return json.loads(balanced_json)
            except json.JSONDecodeError:
                pass

        repaired_json = close_truncated_json_object(clean_text)
        if repaired_json:
            try:
                return json.loads(repaired_json)
            except json.JSONDecodeError:
                pass

        json_pattern_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_pattern_match:
            try:
                return json.loads(json_pattern_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to find valid JSON in text response: {clean_text[:200]}")


def has_placeholder_content(text: str) -> bool:
    """Detects placeholder-style outputs that should be rejected."""
    normalized_text = text.strip().lower()
    return normalized_text in {"...", "n/a", "na", "none", "unknown", "unspecified", "null"}


def validate_scene_report(scene_report: dict) -> None:
    """Raises if the structured scene report contains placeholders or empty fields."""
    required_keys = ("subject", "action", "setting", "mood", "details")
    for key in required_keys:
        value = str(scene_report.get(key, "")).strip()
        if not value:
            raise ValueError(f"Scene report field '{key}' is empty.")
        if has_placeholder_content(value):
            raise ValueError(f"Scene report field '{key}' contains placeholder content.")
        if len(value.split()) < 3:
            raise ValueError(f"Scene report field '{key}' is too short: {value}")


def validate_caption_structure(captions: GeneratedCaptions) -> None:
    """Raises only on structural parse failures. Style rules are checked in Stage C audit."""
    for style in AppConfig.TARGET_STYLES:
        value = getattr(captions, style).strip()
        if not value:
            raise ValueError(f"Caption field '{style}' is empty.")
        if has_placeholder_content(value):
            raise ValueError(f"Caption field '{style}' contains placeholder content.")
        if len(value.split()) < AppConfig.MIN_WORDS:
            raise ValueError(f"Caption field '{style}' is too short: {value}")


def count_words(text: str) -> int:
    """Returns the word count of the given string."""
    return len(text.split())


def contains_technical_terms(text: str) -> bool:
    """Checks if the text contains any tech vocab terms."""
    normalized_text = text.lower()
    return any(term in normalized_text for term in TECHNICAL_DICTIONARY)


def contains_restricted_jargon(text: str) -> bool:
    """Checks if the text contains restricted developer terms."""
    normalized_text = text.lower()
    return any(term in normalized_text for term in RESTRICTED_ENGINEERING_JARGON)


def patch_caption_fields(base_output: GeneratedCaptions, target_output: GeneratedCaptions, target_keys: list[str]) -> GeneratedCaptions:
    """Creates a copy of GeneratedCaptions, replacing specific style keys with values from target_output."""
    merged_data = base_output.model_dump()
    override_data = target_output.model_dump()
    for key in target_keys:
        merged_data[key] = override_data[key]
    return GeneratedCaptions(**merged_data)


def trim_caption_to_word_limit(text: str, max_words: int = AppConfig.MAX_WORDS) -> str:
    """Shortens an overlong caption at the nearest sentence boundary when possible."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    truncated = " ".join(words[:max_words]).rstrip(" ,;:")
    sentence_end = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if sentence_end > len(truncated) // 3:
        return truncated[: sentence_end + 1].strip()
    return truncated.strip()


def apply_local_caption_fixes(captions: GeneratedCaptions) -> GeneratedCaptions:
    """Applies deterministic fixes for common audit failures without an API call."""
    data = captions.model_dump()
    data["formal"] = trim_caption_to_word_limit(data["formal"])
    for style in AppConfig.TARGET_STYLES:
        data[style] = data[style].strip()
    return GeneratedCaptions(**data)


# ── Prompt Assembly ──────────────────────────────────────────────────────────

SCENE_ANALYSIS_EXAMPLE = (
    '{"subject":"a young orange tabby kitten with light fur and alert eyes",'
    '"action":"the kitten sits among plants and occasionally shifts its gaze toward the camera",'
    '"setting":"outdoor garden area with dense green foliage and natural daylight",'
    '"mood":"calm and curious atmosphere",'
    '"details":"warm orange fur against green leaves, soft natural lighting"}'
)

SCENE_ANALYSIS_INSTRUCTION = f"""\
These images are chronological keyframes from one short clip.

Fill every field using only directly visible facts. Each value must be one concrete sentence.

Copy this shape exactly:
{{"subject":"...","action":"...","setting":"...","mood":"...","details":"..."}}

Example:
{SCENE_ANALYSIS_EXAMPLE}

Do not add markdown, commentary, or text outside the JSON object.
"""

SCENE_ANALYSIS_MINIMAL = (
    'Output one JSON object only with full descriptive sentences in every field. '
    'Never use ellipsis, "...", "unknown", or placeholder text. '
    'Shape: {"subject":"...","action":"...","setting":"...","mood":"...","details":"..."}'
)

CAPTION_JSON_EXAMPLE = (
    '{"formal":"A young orange tabby kitten sits among dense green foliage in an outdoor setting, '
    'looking directly at the camera with an alert and curious expression.",'
    '"sarcastic":"A kitten outdoors, clearly plotting something elaborate and fully confident it will succeed.",'
    '"humorous_tech":"A small autonomous agent has entered the garden environment and is scanning for input. '
    'Next action: unknown. Rollback plan: none.",'
    '"humorous_non_tech":"A tiny cat has gone outside and is now judging everything it sees with great authority."}'
)


# ── Few-Shot Reference Examples ──────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
REFERENCE STYLE EXAMPLES (study the tone and structure of each style — do NOT copy the content):

Example A — Autumn trees on a city road with busy traffic:
  formal: "A wide urban boulevard lined with golden ginkgo trees in full autumn foliage, with multiple lanes of traffic flowing through the city below high-rise residential buildings."
  sarcastic: "A city that decided trees were a good idea, which is more than most cities can say."
  humorous_tech: "Nature's annual deployment: all leaf nodes updated to yellow simultaneously, no breaking changes reported."
  humorous_non_tech: "The trees got together and decided to put on a show, and honestly they are the only ones putting in any effort."

Example B — Small orange kitten sitting among green foliage outdoors:
  formal: "A young orange tabby kitten sits among dense green foliage in an outdoor setting, looking directly at the camera with an alert and curious expression."
  sarcastic: "A kitten outdoors, clearly plotting something elaborate and fully confident it will succeed."
  humorous_tech: "A small autonomous agent has entered the garden environment and is scanning for input. Next action: unknown. Rollback plan: none."
  humorous_non_tech: "A tiny cat has gone outside and is now judging everything it sees with great authority."

Example C — Young woman working at a desktop computer in a modern office:
  formal: "A young professional woman is seated at a desktop computer workstation in a bright, modern open-plan office, focused intently on her screen."
  sarcastic: "A person at a computer, apparently working, which is exactly what someone would do if they were not working."
  humorous_tech: "She has been staring at this bug for forty minutes. The bug is a missing comma. The comma is winning."
  humorous_non_tech: "A woman at a computer, visibly handling something extremely important that will be completely forgotten by Thursday."

Example D — A person carefully slicing a green zucchini on a wooden cutting board:
  formal: "A person in a kitchen is slicing a green zucchini into thin, even rounds on a wooden cutting board using a large chef's knife."
  sarcastic: "Someone cutting a vegetable, truly groundbreaking culinary content."
  humorous_tech: "Executing a repetitive slicing loop, outputting identical zucchini data packets with minimal latency."
  humorous_non_tech: "Just chopping up some vegetables, mentally preparing to eat a salad that will definitely leave me hungry."

Example E — A runner navigating a steep, rocky mountain trail:
  formal: "A trail runner navigates a steep, rocky ridge line high in the mountains, carefully stepping over boulders while moving swiftly downhill."
  sarcastic: "A person running down a rocky mountain, because running on flat ground was evidently too easy."
  humorous_tech: "His velocity calculation iterates recursively without hitting any base condition, resulting in an uncontrolled descent."
  humorous_non_tech: "This guy is running down a mountain like he just realized he left the stove on."
"""


def assemble_composition_prompt(visual_report: str, transcript: str = "", priorities: list[str] | None = None) -> str:
    """Constructs the system instructions for generating the styled captions."""
    audio_segment = (
        f'\nSpoken Audio Transcript (for supplementary context):\n"{transcript}"\n'
        if transcript else ""
    )
    priority_flag = (
        f"\n[Attention Required] Focus heavily on refining these styles during this generation: {', '.join(priorities)}\n"
        if priorities else ""
    )

    style_descriptions: list[str] = []
    for p in PERSONA_REGISTRY:
        rules_bulleted = "\n".join(f"      - {rule}" for rule in p.rules)
        style_descriptions.append(
            f"  {p.key}\n"
            f"    Persona Voice: {p.persona}\n"
            f"    Required Rules:\n{rules_bulleted}\n"
            f"    Gold standard example: \"{p.specimen}\"\n"
            f"    Strictly avoid this pattern: \"{p.anti_pattern}\""
        )

    humorous_tech_constraints = (
        "\nHUMOROUS_TECH HARD CONSTRAINTS:\n"
        "- The joke must be anchored to one literal visible cue from the scene summary.\n"
        "- Use exactly one primary tech term or computing metaphor and make it do the punchline work.\n"
        "- Do not use generic hype like fast, smart, powerful, or high performance unless they are part of a concrete tech pun.\n"
        "- Avoid abstract coding talk that is not tied to something visible in the scene.\n"
        "- If the scene has no obvious tech hook, map the most visible object or action to a tech term rather than inventing new context.\n"
    )

    if priorities and "humorous_tech" in priorities:
        humorous_tech_constraints += (
            "- Because humorous_tech is prioritized, keep it especially literal, scene-bound, and pun-first.\n"
        )

    return (
        "You are an expert caption writer generating styled captions for a short video clip.\n"
        "Your captions must stay grounded in the scene analysis below. Do not invent details not present in the scene.\n"
        "If the scene is ambiguous, keep the wording general instead of guessing.\n"
        "Use the transcript only if it directly supports the visible scene.\n\n"
        "Output must be a valid JSON object only. Every field must contain a complete, non-placeholder caption.\n"
        "Do not use ellipses, skeleton values, or repeated schema text.\n\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        "---\n"
        f"NOW WRITE CAPTIONS FOR THIS VIDEO:\n\n"
        f"SCENE SUMMARY:\n{visual_report}\n"
        f"{audio_segment}"
        f"{priority_flag}\n"
        f"{humorous_tech_constraints}\n"
        "Write ONE caption per style. Follow the persona rules and reference examples strictly:\n\n"
        + "\n\n".join(style_descriptions)
        + "\n\nOUTPUT COMPLIANCE RULES:\n"
        "- Return ONLY a valid JSON object. No markdown, no explanation, no preamble.\n"
        '- Required JSON format: {"formal":"...","sarcastic":"...","humorous_tech":"...","humorous_non_tech":"..."}\n'
        f"- Length: each caption must be 1-2 sentences and between {AppConfig.MIN_WORDS} and {AppConfig.MAX_WORDS} words.\n"
        "- Each caption must be a complete sentence or two; do not output fragments, single words, or list items.\n"
        "- CRITICAL: sarcastic must be a short, dry, one-liner — NOT a description. humorous_tech must have clever wordplay tied to something literally visible in the video."
    )


def assemble_composition_prompt_compact(visual_report: str, priorities: list[str] | None = None) -> str:
    """Shorter caption prompt for vision retries when the full prompt truncates."""
    priority_note = (
        f"Prioritize these styles: {', '.join(priorities)}.\n"
        if priorities else ""
    )
    tech_note = (
        "humorous_tech MUST include a tech term (deployment, pipeline, node, API, cache, thread, or packet).\n"
        if priorities and "humorous_tech" in priorities else
        "humorous_tech must include one real tech pun anchored to the scene.\n"
    )
    return (
        f"SCENE:\n{visual_report}\n\n"
        f"{priority_note}"
        f"{tech_note}"
        f"Write formal, sarcastic, humorous_tech, humorous_non_tech captions.\n"
        f"Each caption: {AppConfig.MIN_WORDS}-{AppConfig.MAX_WORDS} words. JSON only.\n"
        f"Example:\n{CAPTION_JSON_EXAMPLE}"
    )


def is_valid_scene_report_text(text: str) -> bool:
    """Returns True when verification output looks like a labeled scene report."""
    if not text.strip():
        return False
    normalized = text.strip().lower()
    if any(normalized.startswith(prefix) for prefix in PROSE_RESPONSE_PREFIXES):
        return False
    if "the user wants" in normalized or "cross-check" in normalized[:80]:
        return False
    return "subject" in normalized and any(
        label in normalized for label in ("action", "setting", "mood", "details")
    )


# ── Model API Calls ──────────────────────────────────────────────────────────

def _select_frame_subset(frames: list[str], max_frames: int | None) -> list[str]:
    """Returns an evenly spaced subset of frames when the full set is too large."""
    if max_frames is None or len(frames) <= max_frames:
        return frames
    if max_frames <= 1:
        return [frames[0]]

    indices: list[int] = []
    for slot in range(max_frames):
        index = round(slot * (len(frames) - 1) / max(max_frames - 1, 1))
        if index not in indices:
            indices.append(index)
    return [frames[index] for index in sorted(indices)]


def _build_vision_payload(
    frames: list[str],
    frame_cache: dict[str, str],
    instruction: str,
    max_frames: int | None = None,
) -> list[dict]:
    """Builds multimodal user content for vision model calls."""
    payload_messages: list[dict] = [{"type": "text", "text": instruction}]
    for path in _select_frame_subset(frames, max_frames):
        payload_messages.append({"type": "image_url", "image_url": {"url": frame_cache[path]}})
    return payload_messages


def _scene_response_format(mode: str) -> dict:
    """Maps retry strategy names to Fireworks response_format payloads."""
    if mode == "strict_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "scene_report",
                "strict": True,
                "schema": FrameDescription.model_json_schema(),
            },
        }
    return {"type": "json_object"}


def _caption_response_format(mode: str) -> dict:
    """Maps retry strategy names to caption response_format payloads."""
    if mode == "strict_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "caption_bundle",
                "strict": True,
                "schema": GeneratedCaptions.model_json_schema(),
            },
        }
    return {"type": "json_object"}


def _parse_scene_response(raw_text: str) -> dict:
    """Parses and validates a scene JSON response, rejecting prose-only outputs."""
    if looks_like_prose_response(raw_text):
        raise ValueError("Model returned prose instead of JSON.")
    try:
        parsed_data = extract_json_object(raw_text)
    except ValueError:
        parsed_data = salvage_scene_fields_from_text(raw_text)
        if parsed_data is None:
            raise
    validate_scene_report(parsed_data)
    return parsed_data


def _merge_scene_partials(existing: dict[str, str] | None, candidate: dict[str, str]) -> dict[str, str]:
    """Keeps the longest valid value for each scene field across attempts."""
    merged = dict(existing or {})
    for key, value in candidate.items():
        cleaned = value.strip()
        if not cleaned or has_placeholder_content(cleaned) or len(cleaned.split()) < 3:
            continue
        if key not in merged or len(cleaned) > len(merged[key]):
            merged[key] = cleaned
    return merged


def _scene_hint_from_partial(partial: dict[str, str]) -> str:
    """Builds a fallback prompt using facts recovered from earlier attempts."""
    if not partial:
        return SCENE_ANALYSIS_MINIMAL
    known_facts = "\n".join(f"- {key}: {value}" for key, value in partial.items())
    return (
        f"{SCENE_ANALYSIS_MINIMAL}\n\n"
        "Use these already-confirmed visible facts and complete every remaining field:\n"
        f"{known_facts}"
    )


def _format_scene_report(parsed_data: dict) -> str:
    """Converts a validated scene dict into the Stage B grounding text block."""
    report_lines = [
        f"Subject : {parsed_data.get('subject', '')}",
        f"Action  : {parsed_data.get('action', '')}",
        f"Setting : {parsed_data.get('setting', '')}",
        f"Mood    : {parsed_data.get('mood', '')}",
        f"Details : {parsed_data.get('details', '')}",
    ]
    return "\n".join(line for line in report_lines if line.split(": ", 1)[-1].strip())


def _invoke_vision_scene_call(
    api_client: OpenAI,
    model_name: str,
    frames: list[str],
    frame_cache: dict[str, str],
    instruction: str,
    response_mode: str,
    max_frames: int | None = None,
    use_json_prefix: bool = False,
) -> str:
    """Issues one vision-model call and returns raw message content."""
    user_messages = [
        {
            "role": "user",
            "content": _build_vision_payload(frames, frame_cache, instruction, max_frames),
        },
    ]
    if use_json_prefix:
        user_messages.append({"role": "assistant", "content": "{"})

    response = api_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Emit one JSON object only. No prose.",
            },
            *user_messages,
        ],
        temperature=0.25,
        response_format=_scene_response_format(response_mode),
    )
    raw_text = (response.choices[0].message.content or "").strip()
    finish_reason = response.choices[0].finish_reason
    if finish_reason == "length":
        print("            Warning: vision response truncated (finish_reason=length).")
    if use_json_prefix and raw_text and not raw_text.startswith("{"):
        raw_text = "{" + raw_text
    return raw_text


def generate_visual_context(
    api_client: OpenAI,
    model_name: str,
    frames: list[str],
    frame_cache: dict[str, str],
) -> str:
    """
    Stage A scene description generation.

    Retries with escalating strategies when the vision model returns prose
    instead of JSON, then falls back to a minimal three-frame request.
    """
    retry_strategies: tuple[dict, ...] = (
        {"response_mode": "json_object", "instruction": SCENE_ANALYSIS_INSTRUCTION, "max_frames": 4, "json_prefix": False},
        {"response_mode": "json_object", "instruction": SCENE_ANALYSIS_INSTRUCTION, "max_frames": 4, "json_prefix": False},
        {"response_mode": "json_object", "instruction": SCENE_ANALYSIS_MINIMAL, "max_frames": 3, "json_prefix": False},
    )
    last_error: Exception | None = None
    best_partial: dict[str, str] = {}

    for attempt_index, strategy in enumerate(retry_strategies[: AppConfig.JSON_RETRY_ATTEMPTS], start=1):
        try:
            raw_text = _invoke_vision_scene_call(
                api_client,
                model_name,
                frames,
                frame_cache,
                strategy["instruction"],
                strategy["response_mode"],
                strategy["max_frames"],
                strategy["json_prefix"],
            )
            best_partial = _merge_scene_partials(best_partial, salvage_json_string_fields(raw_text, SCENE_FIELD_KEYS))
            parsed_data = _parse_scene_response(raw_text)
            if attempt_index > 1:
                print(f"            Stage A succeeded on retry {attempt_index}.")
            return _format_scene_report(parsed_data)
        except Exception as error:
            last_error = error
            print(f"            Stage A attempt {attempt_index} failed: {error}")
            if attempt_index < AppConfig.JSON_RETRY_ATTEMPTS:
                time.sleep(0.75 * attempt_index)

    fallback_strategies: tuple[dict, ...] = (
        {"instruction": _scene_hint_from_partial(best_partial), "max_frames": 3, "json_prefix": True},
        {"instruction": _scene_hint_from_partial(best_partial), "max_frames": 4, "json_prefix": False},
    )
    for fallback_index, strategy in enumerate(fallback_strategies, start=1):
        try:
            print(f"            Stage A fallback {fallback_index} ({strategy['max_frames']} frames)...")
            raw_text = _invoke_vision_scene_call(
                api_client,
                model_name,
                frames,
                frame_cache,
                strategy["instruction"],
                "json_object",
                max_frames=strategy["max_frames"],
                use_json_prefix=strategy["json_prefix"],
            )
            parsed_data = _parse_scene_response(raw_text)
            print(f"            Stage A recovered via fallback {fallback_index}.")
            return _format_scene_report(parsed_data)
        except Exception as fallback_error:
            last_error = fallback_error
            print(f"            Stage A fallback {fallback_index} failed: {fallback_error}")

    if len(best_partial) >= 3:
        try:
            completed_partial = salvage_scene_fields_from_text(
                json.dumps(best_partial, ensure_ascii=False)
            )
            if completed_partial is None:
                completed_partial = best_partial.copy()
                if "mood" not in completed_partial:
                    anchor = completed_partial.get("subject") or completed_partial.get("setting") or "the scene"
                    completed_partial["mood"] = f"an everyday atmosphere visible around {anchor[:100]}"
                if "details" not in completed_partial:
                    completed_partial["details"] = (
                        completed_partial.get("setting")
                        or completed_partial.get("action")
                        or completed_partial.get("subject", "")
                    )
            validate_scene_report(completed_partial)
            print("            Stage A recovered from salvaged partial fields.")
            return _format_scene_report(completed_partial)
        except Exception as salvage_error:
            last_error = salvage_error
            print(f"            Stage A partial salvage failed: {salvage_error}")

    raise RuntimeError(f"Vision grounding failed after retries: {last_error}") from last_error


def ground_scene_verification(
    api_client: OpenAI,
    model_name: str,
    frames: list[str],
    frame_cache: dict[str, str],
    current_draft: str,
) -> str:
    """Stage A.2: cross-checks the scene draft against keyframes and revises hallucinations."""
    if not current_draft:
        return ""
    try:
        verification_instruction = (
            f"Revise only if needed. Return labeled plain text, no commentary.\n\n"
            f"{current_draft}"
        )
        payload_messages: list[dict] = [{"type": "text", "text": verification_instruction}]
        for path in _select_frame_subset(frames, 3):
            payload_messages.append({"type": "image_url", "image_url": {"url": frame_cache[path]}})

        response = api_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only Subject/Action/Setting/Mood/Details lines grounded in the images. "
                        "No preamble."
                    ),
                },
                {"role": "user", "content": payload_messages},
            ],
        )
        verified = (response.choices[0].message.content or "").strip()
        if is_valid_scene_report_text(verified):
            return verified
        print("            Scene verification returned prose; keeping Stage A draft.")
        return current_draft
    except Exception as error:
        print(f"            Scene verification skipped: {error}")
        return current_draft


def _invoke_vision_caption_call(
    api_client: OpenAI,
    model_name: str,
    frames: list[str],
    frame_cache: dict[str, str],
    prompt_text: str,
    temperature: float,
) -> str:
    """Issues one vision-model caption call and returns raw message content."""
    payload_messages: list[dict] = [{"type": "text", "text": prompt_text}]
    for path in _select_frame_subset(frames, 4):
        payload_messages.append({"type": "image_url", "image_url": {"url": frame_cache[path]}})

    response = api_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Emit one JSON object only. No prose."},
            {"role": "user", "content": payload_messages},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw_text = (response.choices[0].message.content or "").strip()
    if response.choices[0].finish_reason == "length":
        print("            Warning: vision caption response truncated (finish_reason=length).")
    return raw_text


def request_vision_caption_inference(
    api_client: OpenAI,
    model_name: str,
    frames: list[str],
    frame_cache: dict[str, str],
    visual_report: str,
    transcript: str = "",
    priorities: list[str] | None = None,
) -> GeneratedCaptions:
    """Drafts captions from keyframes plus scene context using the vision model."""
    full_prompt = assemble_composition_prompt(visual_report, transcript, priorities)
    compact_prompt = assemble_composition_prompt_compact(visual_report, priorities)
    strategies: tuple[dict, ...] = (
        {"prompt": full_prompt, "temperature": 0.85},
        {"prompt": compact_prompt, "temperature": 0.75},
        {"prompt": compact_prompt, "temperature": 0.65},
    )
    last_error: Exception | None = None

    for attempt_index, strategy in enumerate(strategies[: AppConfig.JSON_RETRY_ATTEMPTS], start=1):
        try:
            raw_text = _invoke_vision_caption_call(
                api_client,
                model_name,
                frames,
                frame_cache,
                strategy["prompt"],
                strategy["temperature"],
            )
            captions = _parse_caption_response(raw_text)
            captions = apply_local_caption_fixes(captions)
            if attempt_index > 1:
                print(f"            Vision caption call succeeded on retry {attempt_index}.")
            return captions
        except Exception as error:
            last_error = error
            print(f"            Vision caption attempt {attempt_index} failed: {error}")
            if attempt_index < AppConfig.JSON_RETRY_ATTEMPTS:
                time.sleep(0.75 * attempt_index)

    raise RuntimeError(f"Vision caption generation failed: {last_error}") from last_error


def call_vision_caption_generation(
    api_client: OpenAI,
    frames: list[str],
    frame_cache: dict[str, str],
    visual_report: str,
    transcript: str = "",
    priorities: list[str] | None = None,
) -> GeneratedCaptions:
    """Generates captions via the vision model only (no text-model fallback)."""
    return request_vision_caption_inference(
        api_client,
        AppConfig.VISION_MODEL,
        frames,
        frame_cache,
        visual_report,
        transcript,
        priorities,
    )


def _parse_caption_response(raw_text: str) -> GeneratedCaptions:
    """Parses and validates a caption JSON response, rejecting prose-only outputs."""
    if looks_like_prose_response(raw_text):
        raise ValueError("Model returned prose instead of JSON.")
    try:
        parsed_json = extract_json_object(raw_text)
    except ValueError:
        parsed_json = salvage_caption_fields_from_text(raw_text)
        if parsed_json is None:
            raise
    captions = GeneratedCaptions(**parsed_json)
    validate_caption_structure(captions)
    return captions


# ── Rule Auditing ────────────────────────────────────────────────────────────

def validate_rules_on_captions(caption_set: GeneratedCaptions) -> list[str]:
    """
    Runs quick local heuristics (word count, tech terms, jargon check) to audit style.

    Zero API costs. Returns style keys that violate at least one check.
    """
    flagged_styles: list[str] = []

    # 1. Word count constraint validation
    for style in AppConfig.TARGET_STYLES:
        text_content = getattr(caption_set, style)
        word_count = count_words(text_content)
        if not (AppConfig.MIN_WORDS <= word_count <= AppConfig.MAX_WORDS):
            print(f"  [audit] Flagged '{style}': word count of {word_count} is outside range [{AppConfig.MIN_WORDS}, {AppConfig.MAX_WORDS}]")
            if style not in flagged_styles:
                flagged_styles.append(style)

    # 2. Tech slang requirements in humorous_tech
    if "humorous_tech" not in flagged_styles and not contains_technical_terms(caption_set.humorous_tech):
        print("  [audit] Flagged 'humorous_tech': no programming or hardware terms present.")
        flagged_styles.append("humorous_tech")

    # 3. Exclude tech vocabulary from humorous_non_tech
    if "humorous_non_tech" not in flagged_styles and contains_restricted_jargon(caption_set.humorous_non_tech):
        print("  [audit] Flagged 'humorous_non_tech': restricted technical jargon detected.")
        flagged_styles.append("humorous_non_tech")

    # 4. Objective news compliance in formal
    if "formal" not in flagged_styles and "!" in caption_set.formal:
        print("  [audit] Flagged 'formal': exclamation marks are prohibited.")
        flagged_styles.append("formal")

    return flagged_styles


# ── Unified Pipeline Entrypoint ────────────────────────────────────────────────

def synthesize_narratives(frames: list[str], transcript: str = "") -> GeneratedCaptions:
    """
    Executes the multi-stage narrative generation process for a video.

    Stage A grounds the scene, Stage B drafts captions from frames, Stage C repairs
    audit violations, and Stages D/E refine weak styles using the self-grader.
    """
    api_client = OpenAI(api_key=AppConfig.AIMLAPI_KEY, base_url=AppConfig.AIMLAPI_URL)

    frame_cache = {path: image_to_base64_uri(path) for path in frames}

    # ── Stage A: Extract Scene Description ──
    print("  [Stage A] Generating scene context...")
    initial_draft = generate_visual_context(api_client, AppConfig.VISION_MODEL, frames, frame_cache)
    print("            Verifying scene description correctness...")
    visual_report = ground_scene_verification(
        api_client, AppConfig.VISION_MODEL, frames, frame_cache, initial_draft
    )
    summary_preview = " | ".join(visual_report.splitlines())
    print(f"            Description verified: {summary_preview[:110]}...")

    # ── Stage B: Draft Styled Captions (vision-grounded, like 0.85 pipeline) ──
    print("  [Stage B] Generating captions...")
    generated_output = call_vision_caption_generation(
        api_client, frames, frame_cache, visual_report, transcript
    )
    generated_output = apply_local_caption_fixes(generated_output)

    # ── Rule-Based Validation & Repair ──
    print("  [Stage C] Running static quality audit...")
    audit_violations = validate_rules_on_captions(generated_output)
    if audit_violations:
        print(f"            Violations flagged: {audit_violations} — issuing targeted repair pass...")
        try:
            repaired_captions = call_vision_caption_generation(
                api_client,
                frames,
                frame_cache,
                visual_report,
                transcript,
                priorities=audit_violations,
            )
            generated_output = patch_caption_fields(generated_output, repaired_captions, audit_violations)
            generated_output = apply_local_caption_fixes(generated_output)
            remaining_violations = validate_rules_on_captions(generated_output)
            if remaining_violations:
                print(f"            Remaining issues after repair: {remaining_violations} (keeping best draft)")
        except Exception as repair_error:
            print(f"            Caption repair process failed: {repair_error}")
    else:
        print("            All style captions passed static validation.")

    # Check for Fast Mode (skip LLM Refinement)
    if not AppConfig.REFINEMENT_ENABLED:
        print("            Fast Mode enabled: Skipping LLM Grading & Refinement (Stages D/E).")
        return generated_output

    # ── LLM Quality Assessment ──
    print("  [Stage D] Evaluating captions against rubric criteria...")
    assessment_results = assess_caption_quality(frames, generated_output)
    if assessment_results is None:
        print("            Assessment unavailable: returning current narrative set.")
        return generated_output

    weak_styles: list[str] = []
    for style in AppConfig.TARGET_STYLES:
        metric = getattr(assessment_results, style)
        average_score = (metric.accuracy + metric.style_match) / 2.0
        status_flag = "  <-- below threshold" if average_score < AppConfig.EVAL_THRESHOLD else ""
        print(
            f"            {style:<22} accuracy={metric.accuracy:.2f}  "
            f"style_match={metric.style_match:.2f}  avg={average_score:.2f}{status_flag}"
        )
        if average_score < AppConfig.EVAL_THRESHOLD:
            weak_styles.append(style)

    if not weak_styles:
        print("            All outputs satisfy quality benchmarks.")
        return generated_output

    # ── Multi-Candidate Refinement ──
    print(f"  [Stage E] Refining {weak_styles} ({AppConfig.MAX_REGEN_TRIES} candidates each)...")
    refined_output = generated_output
    historical_best_scores = {
        style: (getattr(assessment_results, style).accuracy + getattr(assessment_results, style).style_match) / 2.0
        for style in weak_styles
    }

    for iteration in range(1, AppConfig.MAX_REGEN_TRIES + 1):
        try:
            candidate_captions = call_vision_caption_generation(
                api_client,
                frames,
                frame_cache,
                visual_report,
                transcript,
                priorities=weak_styles,
            )
            candidate_captions = apply_local_caption_fixes(candidate_captions)
            candidate_evaluation = assess_caption_quality(frames, candidate_captions)
            if candidate_evaluation is None:
                continue

            improved_styles: list[str] = []
            for style in weak_styles:
                style_metrics = getattr(candidate_evaluation, style)
                candidate_average = (style_metrics.accuracy + style_metrics.style_match) / 2.0
                if candidate_average > historical_best_scores.get(style, 0.0):
                    historical_best_scores[style] = candidate_average
                    refined_output = patch_caption_fields(refined_output, candidate_captions, [style])
                    improved_styles.append(style)

            if improved_styles:
                print(f"            [{iteration}/{AppConfig.MAX_REGEN_TRIES}] Improved styles: {improved_styles}")
            else:
                print(f"            [{iteration}/{AppConfig.MAX_REGEN_TRIES}] No improvement found.")
        except Exception as candidate_error:
            print(f"            [{iteration}/{AppConfig.MAX_REGEN_TRIES}] Candidate skipped: {candidate_error}")

    return apply_local_caption_fixes(refined_output)
