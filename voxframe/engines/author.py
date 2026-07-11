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
        persona="a senior engineer who can only process reality through technical metaphors and finds clever wordplay between code concepts and real-world visuals",
        rules=(
            "Find a tech term that DOUBLES as a description of something literally in the video (e.g., autumn leaves → leaf nodes, cat exploring → autonomous agent scanning, office worker → polling loop)",
            "The wordplay must be clever and specific to THIS video — not just a generic tech sentence",
            "1-2 sentences maximum, punchy and witty",
            "The humor comes from the unexpected but perfectly fitting tech analogy",
            "Must include at least one real software/hardware term: deployment, pipeline, kernel, API, agent, loop, thread, cache, etc.",
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

    return (
        "You are an expert caption writer generating styled captions for a short video clip.\n"
        "Your captions must be based STRICTLY on the scene analysis below. Do not invent details not present in the scene.\n\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        "---\n"
        f"NOW WRITE CAPTIONS FOR THIS VIDEO:\n\n"
        f"SCENE SUMMARY:\n{visual_report}\n"
        f"{audio_segment}"
        f"{priority_flag}\n"
        "Write ONE caption per style. Follow the persona rules and reference examples strictly:\n\n"
        + "\n\n".join(style_descriptions)
        + "\n\nOUTPUT COMPLIANCE RULES:\n"
        "- Return ONLY a valid JSON object. No markdown, no explanation, no preamble.\n"
        '- Required JSON format: {"formal":"...","sarcastic":"...","humorous_tech":"...","humorous_non_tech":"..."}\n'
        f"- Length: each caption must be 1-2 sentences and between {AppConfig.MIN_WORDS} and {AppConfig.MAX_WORDS} words.\n"
        "- CRITICAL: sarcastic must be a short, dry, one-liner — NOT a description. humorous_tech must have clever wordplay tied to something literally visible in the video."
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
            max_tokens=600,
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

    for attempt_index, temperature in enumerate([0.95, 1.05]):
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
