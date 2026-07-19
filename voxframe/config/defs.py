"""
VoxFrame Domain Models
=====================
Defines structured response schemas for narration components, scene analyses,
and quantitative quality assurance metrics.
"""
from pydantic import BaseModel, Field


class GeneratedCaptions(BaseModel):
    """Holds a set of four stylistically diverse narrations for a single video segment."""

    formal: str = Field(
        description="Objective, factual news-oriented caption. Written in third person without opinion or emotion."
    )
    sarcastic: str = Field(
        description="Ironical, dry, and mockingly serious caption showcasing understated wit."
    )
    humorous_tech: str = Field(
        description="Amusing caption containing programming, code, or computing-related jokes or terminology."
    )
    humorous_non_tech: str = Field(
        description="Lighthearted, relatable everyday joke or caption completely devoid of technical jargon."
    )


class FrameDescription(BaseModel):
    """Encapsulates structured data representing the visual context extracted from keyframes."""

    subject: str = Field(description="Main subject or objects appearing in the video.")
    action: str = Field(description="The primary movement or activity occurring in the clip.")
    setting: str = Field(description="Physical context, environment, or location details.")
    mood: str = Field(description="Atmosphere, emotional tone, or stylistic vibe.")
    details: str = Field(description="Specific characteristics such as pacing, color palettes, and motion.")


class StyleMetrics(BaseModel):
    """Stores quantitative feedback along separate evaluation axes for a single style."""

    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Degree of alignment and factual adherence to actual video content.",
    )
    style_match: float = Field(
        ge=0.0,
        le=1.0,
        description="Compliance of the caption text with the target tone and rules.",
    )


class EvaluationReport(BaseModel):
    """Aggregated grades for all four styles scored by the model critic."""

    formal: StyleMetrics
    sarcastic: StyleMetrics
    humorous_tech: StyleMetrics
    humorous_non_tech: StyleMetrics
