"""
VoxFrame Caption Evaluation System
==================================
Uses LLM-based verification to score candidate captions against original video
keyframes on accuracy (grounding) and style compliance.
"""
import base64
import json
import re

from openai import OpenAI
from pydantic import ValidationError

from voxframe.config.cfg import AppConfig
from voxframe.config.defs import GeneratedCaptions, EvaluationReport


# ── Quality Evaluation Helper Functions ───────────────────────────────────────

def convert_to_base64_data_uri(file_path: str) -> str:
    """Reads a local image and builds a base64 encoded data URI."""
    with open(file_path, "rb") as image_file:
        raw_bytes = image_file.read()
        encoded_string = base64.b64encode(raw_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"


def parse_json_response(raw_text: str) -> dict:
    """Safely extracts JSON from an LLM response containing potential fences."""
    if not raw_text:
        raise ValueError("Received empty input text.")
    
    clean_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
    clean_text = re.sub(r"\s*```$", "", clean_text)
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Unable to extract JSON from content: {clean_text[:200]}")


# ── Assessment Prompt Instruction ─────────────────────────────────────────────

EVALUATION_RUBRIC_PROMPT = """\
You are a professional quality assurance critic reviewing video captions against keyframe images.

Assign scores from 0.0 (unacceptable) to 1.0 (outstanding) for the following parameters:

  accuracy     Does the caption describe exactly what is visually present in the keyframes?
               1.0 = highly specific and correct | 0.5 = correct but generic | 0.0 = contains hallucinations.

  style_match  Does the wording align precisely with the requirements of the style persona?
               Style requirements:

    formal          Reuters news style: objective, factual, third-person.
                    Strictly zero jokes, slang, or exclamation marks.

    sarcastic       Dry, ironic, mock-serious tone with understated humor.
                    Keep it clever and subtle, not overly silly or loud.

    humorous_tech   Must feature a valid technology, code, or hardware reference that makes it funny.
                    Score 0.0 if no tech terms are used or if it lacks humor.

    humorous_non_tech  Lighthearted, relatable everyday joke that requires no programming knowledge.
                       Score 0.0 if any developer/technical terminology is used.

Captions to Review:
  formal            : "{formal}"
  sarcastic         : "{sarcastic}"
  humorous_tech     : "{humorous_tech}"
  humorous_non_tech : "{humorous_non_tech}"

Output ONLY a raw JSON block matching this layout, with no extra text or markdown wrappers:
{{"formal":{{"accuracy":0.0,"style_match":0.0}},"sarcastic":{{"accuracy":0.0,"style_match":0.0}},"humorous_tech":{{"accuracy":0.0,"style_match":0.0}},"humorous_non_tech":{{"accuracy":0.0,"style_match":0.0}}}}

Be extremely critical. Scores greater than 0.85 should be reserved for exceptional compliance.\
"""


# ── Quality Evaluation Entrypoint ─────────────────────────────────────────────

def assess_caption_quality(frames: list[str], captions: GeneratedCaptions) -> EvaluationReport | None:
    """
    Evaluates a set of GeneratedCaptions against the corresponding video keyframes.

    Returns an EvaluationReport containing the metrics or None if an error occurs.
    """
    try:
        api_client = OpenAI(api_key=AppConfig.FIREWORKS_KEY, base_url=AppConfig.FIREWORKS_URL)

        rubric_instruction = EVALUATION_RUBRIC_PROMPT.format(
            formal=captions.formal,
            sarcastic=captions.sarcastic,
            humorous_tech=captions.humorous_tech,
            humorous_non_tech=captions.humorous_non_tech,
        )
        
        payload_messages: list[dict] = [{"type": "text", "text": rubric_instruction}]
        for path in frames:
            payload_messages.append({"type": "image_url", "image_url": {"url": convert_to_base64_data_uri(path)}})

        response = api_client.chat.completions.create(
            model=AppConfig.FIREWORKS_MODEL,
            messages=[{"role": "user", "content": payload_messages}],
            max_tokens=600,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        response_content = response.choices[0].message.content
        if response_content is None:
            return None

        parsed_json = parse_json_response(response_content)
        return EvaluationReport(**parsed_json)

    except (ValidationError, ValueError, Exception) as error:
        print(f"  [grader] Quality assessment failed (degrading gracefully): {error}")
        return None
