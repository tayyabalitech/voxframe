"""
VoxFrame Narrative Composition Engine
====================================
Coordinates Stage A (scene analysis & verification) and Stage B (narrative style composition),
along with constraint-based validation and LLM-assisted self-refinement.
"""
import base64
import json
import re
from dataclasses import dataclass

from openai import OpenAI
from pydantic import ValidationError

from voxframe.config.cfg import AppConfig
from voxframe.config.defs import GeneratedCaptions, EvaluationReport
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
        persona="an agency wire reporter issuing a factual, clear, and objective report",
        rules=(
            "Maintain a professional, objective, and neutral third-person perspective",
            "State only observable facts (what, who, action, setting) without personal opinions",
            "Do not include exclamation points, slang, metaphors, or joking comments",
            "Use full, grammatically correct sentences with precise wording",
        ),
        specimen="A professional cyclist travels at high speed down a steep mountain course.",
        anti_pattern="Unbelievable! This rider is absolutely flying down the mountain trail!",
    ),
    StylePersona(
        key="sarcastic",
        persona="a cynical, unimpressed observer stating the obvious with dry irony",
        rules=(
            "Use subtle, deadpan irony and understated humor — do not state that you are joking",
            "Describe the scene with mock-seriousness, as if noting something boringly obvious",
            "Aim for a dry smirk rather than a direct laugh or silliness",
            "Avoid excessive exclamation marks or loud, childish expressions",
        ),
        specimen="Gravity functions as expected. The physical universe remains largely undisturbed.",
        anti_pattern="OMG that is so funny, I'm literally crying laughing at this guy!",
    ),
    StylePersona(
        key="humorous_tech",
        persona="a systems programmer who views all real-world events through computing metaphors",
        rules=(
            "Must incorporate at least one authentic software, hardware, or engineering concept",
            "The technical analogy should drive or enhance the humor, not just be tagged on",
            "Valid terms include pipeline, kernel, memory leak, cache miss, latency, or API",
            "The joke must resonate with a technical developer audience",
        ),
        specimen="The runner's movement engine has disabled rate limiting, achieving maximum frame rate with zero garbage collection.",
        anti_pattern="This runner is moving fast, which requires high bandwidth calculation.",
    ),
    StylePersona(
        key="humorous_non_tech",
        persona="a witty friend sending a clever voice note to a general group chat",
        rules=(
            "Use accessible, lighthearted humor that any general audience can appreciate",
            "Think of witty everyday remarks, sitcom tropes, or friendly banter",
            "Strictly exclude any technical, developer, or programming jargon",
            "Keep the phrasing conversational, natural, and friendly",
        ),
        specimen="Someone forgot to tell this person that running up vertical walls isn't normally allowed.",
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


def extract_json_object(model_output: str) -> dict:
    """Extracts and parses the first JSON object block from the text response."""
    if not model_output:
        raise ValueError("Received empty model output.")
    
    clean_text = re.sub(r"^```(?:json)?\s*", "", model_output.strip())
    clean_text = re.sub(r"\s*```$", "", clean_text)
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        json_pattern_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_pattern_match:
            return json.loads(json_pattern_match.group(0))
        raise ValueError(f"Failed to find valid JSON in text response: {clean_text[:200]}")


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


# ── Prompt Assembly ──────────────────────────────────────────────────────────

SCENE_ANALYSIS_INSTRUCTION = """\
Analyze the chronologically ordered keyframes extracted from a short video clip.

Provide a detailed, structured scene report in JSON format. Be highly concrete and avoid generalities.

Required JSON Structure:
{
  "subject":  "detailed description of primary subjects/people/objects",
  "action":   "precise movements or events occurring in the timeline",
  "setting":  "environment details, lighting, indoors/outdoors description",
  "mood":     "overall mood, atmosphere, or stylistic aesthetic",
  "details":  "notable visuals: colors, motion speed, actions, background details"
}

Rules:
- Output ONLY the raw JSON object. No explanations, markdown tags, or preambles.
- Avoid vague statements like "a person runs". Write: "a young man wearing red athletic gear sprints along a wet pavement track."
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
            f"    Gold standard: \"{p.specimen}\"\n"
            f"    Avoid pattern: \"{p.anti_pattern}\""
        )

    return (
        "You are writing a set of unique styled captions for a short video clip.\n"
        "Ensure all descriptions are strictly derived from the scene analysis report below. Do not invent details.\n\n"
        f"SCENE SUMMARY:\n{visual_report}\n"
        f"{audio_segment}"
        f"{priority_flag}\n"
        "Create ONE caption for each style:\n\n"
        + "\n\n".join(style_descriptions)
        + "\n\nOUTPUT COMPLIANCE RULES:\n"
        "- Return ONLY a valid JSON object. Do not wrap in markdown syntax or add pre/post explanation text.\n"
        '- Required JSON format: {"formal":"...","sarcastic":"...","humorous_tech":"...","humorous_non_tech":"..."}\n'
        f"- Length limit: each style caption must be 1-2 complete sentences and fall between {AppConfig.MIN_WORDS} and {AppConfig.MAX_WORDS} words.\n"
        "- Ensure sentence structures and verb selections are distinct across styles to maintain a unique persona voice."
    )


# ── Model API Calls ──────────────────────────────────────────────────────────

def generate_visual_context(api_client: OpenAI, frames: list[str], frame_cache: dict[str, str]) -> str:
    """
    Stage A scene description generation.
    
    Requests the vision model to analyze the chronological sequence and return
    a structured JSON report. Returns formatted text representation.
    """
    try:
        payload_messages: list[dict] = [{"type": "text", "text": SCENE_ANALYSIS_INSTRUCTION}]
        for path in frames:
            payload_messages.append({"type": "image_url", "image_url": {"url": frame_cache[path]}})

        response = api_client.chat.completions.create(
            model=AppConfig.FIREWORKS_MODEL,
            messages=[{"role": "user", "content": payload_messages}],
            max_tokens=350,
            temperature=0.25,
            response_format={"type": "json_object"},
        )
        raw_text = (response.choices[0].message.content or "").strip()
        parsed_data = extract_json_object(raw_text)

        report_lines = [
            f"Subject : {parsed_data.get('subject', '')}",
            f"Action  : {parsed_data.get('action', '')}",
            f"Setting : {parsed_data.get('setting', '')}",
            f"Mood    : {parsed_data.get('mood', '')}",
            f"Details : {parsed_data.get('details', '')}",
        ]
        return "\n".join(line for line in report_lines if line.split(": ", 1)[-1].strip())
    except Exception as error:
        print(f"  [author] Vision scene analysis skipped (non-fatal): {error}")
        return ""


def ground_scene_verification(api_client: OpenAI, frames: list[str], frame_cache: dict[str, str], current_draft: str) -> str:
    """
    Stage A.2 grounding check.
    
    Verifies that the scene analysis contains no hallucinations.
    """
    if not current_draft:
        return ""
    try:
        verification_instruction = (
            f"Cross-reference the draft scene description below with the keyframes.\n"
            f"Verify if it describes the settings, subjects, action, and details accurately without hallucinations.\n"
            f"If it is correct, output it exactly unchanged.\n"
            f"If it contains errors or overly vague terms (like 'a person' or 'something'), revise it with details.\n"
            f"Provide only the final verified plain text description.\n\n"
            f"DRAFT DESCRIPTION:\n{current_draft}"
        )
        payload_messages: list[dict] = [{"type": "text", "text": verification_instruction}]
        for path in frames:
            payload_messages.append({"type": "image_url", "image_url": {"url": frame_cache[path]}})

        response = api_client.chat.completions.create(
            model=AppConfig.FIREWORKS_MODEL,
            messages=[{"role": "user", "content": payload_messages}],
            max_tokens=350,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as error:
        print(f"  [author] Scene description verification skipped: {error}")
        return current_draft


def request_model_inference(
    api_client: OpenAI,
    model_name: str,
    frames: list[str],
    frame_cache: dict[str, str],
    visual_report: str,
    transcript: str = "",
    priorities: list[str] | None = None,
    temperature_setting: float = 0.75,
) -> GeneratedCaptions:
    """Performs a single vision LLM call to draft the styled captions in JSON."""
    payload_messages: list[dict] = [
        {"type": "text", "text": assemble_composition_prompt(visual_report, transcript, priorities)}
    ]
    for path in frames:
        payload_messages.append({"type": "image_url", "image_url": {"url": frame_cache[path]}})

    response = api_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": payload_messages}],
        max_tokens=1500,
        temperature=temperature_setting,
        response_format={"type": "json_object"},
    )
    raw_response_content = response.choices[0].message.content
    if raw_response_content is None:
        raise ValueError(f"Inference returned null message (finish_reason: {response.choices[0].finish_reason})")
    
    parsed_json = extract_json_object(raw_response_content)
    return GeneratedCaptions(**parsed_json)


def call_vision_generation(
    api_client: OpenAI,
    frames: list[str],
    frame_cache: dict[str, str],
    visual_report: str,
    transcript: str = "",
    priorities: list[str] | None = None,
) -> GeneratedCaptions:
    """Attempts generation using the primary model twice, then falls back to secondary model."""
    last_encountered_exception = None

    for attempt_index, temperature in enumerate([0.75, 0.90]):
        try:
            return request_model_inference(
                api_client,
                AppConfig.FIREWORKS_MODEL,
                frames,
                frame_cache,
                visual_report,
                transcript,
                priorities,
                temperature,
            )
        except Exception as error:
            print(f"  [author] Attempt {attempt_index + 1} with primary model failed: {error}")
            last_encountered_exception = error

    print("  [author] Primary options exhausted. Invoking fallback model...")
    try:
        return request_model_inference(
            api_client,
            AppConfig.FIREWORKS_FALLBACK,
            frames,
            frame_cache,
            visual_report,
            transcript,
            priorities,
        )
    except Exception as fallback_error:
        raise RuntimeError(
            f"All caption generation channels failed. Primary model error: {last_encountered_exception}. Fallback model error: {fallback_error}."
        )


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
    
    Includes Stage A scene comprehension, verification, Stage B styled narration drafting,
    rule audits, self-evaluation scoring, and dynamic refinement.
    """
    api_client = OpenAI(api_key=AppConfig.FIREWORKS_KEY, base_url=AppConfig.FIREWORKS_URL)

    # Encode images to base64 once to optimize performance
    frame_cache = {path: image_to_base64_uri(path) for path in frames}

    # ── Stage A: Extract Scene Description ──
    print("  [Stage A] Generating scene context...")
    initial_draft = generate_visual_context(api_client, frames, frame_cache)
    if initial_draft:
        print("            Verifying scene description correctness...")
        visual_report = ground_scene_verification(api_client, frames, frame_cache, initial_draft)
        summary_preview = " | ".join(visual_report.splitlines())
        print(f"            Description verified: {summary_preview[:110]}...")
    else:
        visual_report = ""
        print("            Skipped scene description: proceeding without context.")

    # ── Stage B: Draft Styled Captions ──
    print("  [Stage B] Generating captions...")
    generated_output = call_vision_generation(api_client, frames, frame_cache, visual_report, transcript)

    # ── Rule-Based Validation & Repair ──
    print("  [Stage C] Running static quality audit...")
    audit_violations = validate_rules_on_captions(generated_output)
    if audit_violations:
        print(f"            Violations flagged: {audit_violations} — issuing targeted repair pass...")
        try:
            repaired_captions = call_vision_generation(
                api_client, frames, frame_cache, visual_report, transcript, priorities=audit_violations
            )
            generated_output = patch_caption_fields(generated_output, repaired_captions, audit_violations)
            remaining_violations = validate_rules_on_captions(generated_output)
            if remaining_violations:
                print(f"            Remaining issues after repair: {remaining_violations} (keeping current draft)")
        except Exception as repair_error:
            print(f"            Caption repair process failed: {repair_error}")
    else:
        print("            All style captions passed static validation.")

    # ── LLM Quality Assessment ──
    print("  [Stage D] Evaluating captions against rubric criteria...")
    assessment_results = assess_caption_quality(frames, generated_output)
    if assessment_results is None:
        print("            Assessment server offline: returning current narrative set.")
        return generated_output

    weak_styles: list[str] = []
    for style in AppConfig.TARGET_STYLES:
        metric = getattr(assessment_results, style)
        average_score = (metric.accuracy + metric.style_match) / 2.0
        status_flag = "  <-- below threshold" if average_score < AppConfig.EVAL_THRESHOLD else ""
        print(f"            {style:<22} accuracy={metric.accuracy:.2f}  style_match={metric.style_match:.2f}  avg={average_score:.2f}{status_flag}")
        if average_score < AppConfig.EVAL_THRESHOLD:
            weak_styles.append(style)

    if not weak_styles:
        print("            All outputs satisfy quality benchmarks.")
        return generated_output

    # ── Multi-Candidate Refinement ──
    print(f"  [Stage E] Initiating refinement for {weak_styles} ({AppConfig.MAX_REGEN_TRIES} candidates each)...")
    refined_output = generated_output
    historical_best_scores = {
        style: (getattr(assessment_results, style).accuracy + getattr(assessment_results, style).style_match) / 2.0
        for style in weak_styles
    }

    for iteration in range(1, AppConfig.MAX_REGEN_TRIES + 1):
        try:
            candidate_captions = call_vision_generation(
                api_client, frames, frame_cache, visual_report, transcript, priorities=weak_styles
            )
            candidate_evaluation = assess_caption_quality(frames, candidate_captions)
            if candidate_evaluation is None:
                continue

            improved_styles = []
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
            print(f"            [{iteration}/{AppConfig.MAX_REGEN_TRIES}] Candidate iteration skipped due to error: {candidate_error}")

    return refined_output
