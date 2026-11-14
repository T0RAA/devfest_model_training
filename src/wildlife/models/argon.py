"""Gemini 4 Argon (`gemini-4-argon`) multimodal integration for Google Cloud / Vertex AI & Colab.

Provides:
1. Grounded, non-overclaiming natural-language explanations of photo quality and species
   predictions per SPEC.md §1.6 (e.g., "well-focused bird, partially hidden tail; likely Steller's jay").
2. Optional live multimodal call to `gemini-4-argon` via the Google Gen AI SDK (`google-genai`)
   on Vertex AI or AI Studio when credentials (`GOOGLE_API_KEY` or `USE_VERTEX_AI=true` +
   `GCP_PROJECT_ID`) are configured.
3. Deterministic, zero-hallucination rule-based explanation synthesis when running offline
   or in deterministic unit tests.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

ARGON_MODEL_ID = "gemini-4-argon"


def build_interpretable_explanation(
    status: str,
    predicted_species: Optional[Dict[str, Any]],
    top_3: List[Dict[str, Any]],
    photo_quality: Dict[str, int],
    coarse_group_fallback: Optional[str] = None,
    crop_used: bool = False,
) -> str:
    """Synthesize a concise, grounded explanation without invented biological certainty (§1.6)."""
    sharp = int(photo_quality.get("sharpness", 0))
    vis = int(photo_quality.get("visibility", 0))
    exp = int(photo_quality.get("exposure", 0))
    res = int(photo_quality.get("resolution", 0))

    if sharp >= 75:
        focus_phrase = "well-focused"
    elif sharp >= 50:
        focus_phrase = "moderately sharp"
    else:
        focus_phrase = "motion-blurred or soft-focus"

    if vis >= 75:
        vis_phrase = "clear subject framing"
    elif vis >= 50:
        vis_phrase = "partially framed subject near crop margin"
    else:
        vis_phrase = "heavily cropped or partially occluded subject"

    lighting_note = ""
    if exp < 45:
        lighting_note = ", challenging low-light/clipped exposure"
    elif res < 45:
        lighting_note = ", low effective subject resolution"

    crop_note = " (manual crop applied)" if crop_used else ""

    if status == "identified" and predicted_species is not None:
        common = predicted_species.get("common_name", "supported species")
        group_raw = predicted_species.get("group", "animal").replace("_", " ")
        return (
            f"{focus_phrase.capitalize()} {group_raw} with {vis_phrase}{lighting_note}"
            f"{crop_note}; likely {common}."
        )

    if status == "no_animal":
        return (
            f"No supported wild animal detected ({focus_phrase} frame{lighting_note}); "
            "please upload or crop around a visible mammal or bird."
        )

    if status == "unsupported_animal":
        return (
            f"{focus_phrase.capitalize()} photograph with {vis_phrase}{lighting_note}, "
            "but subject appears outside the 20 supported UCSC campus mammal/bird classes."
        )

    # Uncertain status
    lead = top_3[0].get("common_name", "unknown") if top_3 else "unknown"
    runner = top_3[1].get("common_name", "unknown") if len(top_3) > 1 else None
    group_prefix = f"{coarse_group_fallback}; " if coarse_group_fallback else ""
    cand_text = f"top candidates include {lead}" + (f" and {runner}" if runner else "")
    return (
        f"{group_prefix}{focus_phrase.capitalize()} image with {vis_phrase}{lighting_note}"
        f"{crop_note}; species ID is uncertain ({cand_text})—try a closer crop or clearer angle."
    )


class ArgonMultimodalAssistant:
    """Client wrapper for Gemini 4 Argon (`gemini-4-argon`) on Vertex AI / Google Gen AI SDK."""

    def __init__(
        self,
        model_id: str = ARGON_MODEL_ID,
        project_id: Optional[str] = None,
        location: str = "us-west1",
        enable_live_api: bool = False,
    ) -> None:
        self.model_id = model_id
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "")
        self.location = location or os.environ.get("GCP_REGION", "us-west1")
        self.enable_live_api = bool(enable_live_api)

    def explain_prediction(
        self,
        status: str,
        predicted_species: Optional[Dict[str, Any]],
        top_3: List[Dict[str, Any]],
        photo_quality: Dict[str, int],
        coarse_group_fallback: Optional[str] = None,
        crop_used: bool = False,
        image_bytes: Optional[bytes] = None,
    ) -> str:
        """Generate an interpretable explanation via `gemini-4-argon` if enabled, else deterministic rule synthesis."""
        baseline_text = build_interpretable_explanation(
            status=status,
            predicted_species=predicted_species,
            top_3=top_3,
            photo_quality=photo_quality,
            coarse_group_fallback=coarse_group_fallback,
            crop_used=crop_used,
        )
        if not self.enable_live_api or image_bytes is None:
            return baseline_text

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            if self.project_id:
                client = genai.Client(
                    vertexai=True, project=self.project_id, location=self.location
                )
            else:
                client = genai.Client()

            prompt = (
                "You are auditing a UCSC Campus Natural Reserve wildlife photograph. "
                f"Classifier status: {status}. Top-3 candidates: {top_3}. "
                f"Photo quality breakdown (0-100): {photo_quality}. "
                "Provide a single concise sentence (< 25 words) describing focus, visibility, "
                "and tentative species assessment without inventing biological certainty, sex, age, or abundance."
            )
            response = client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            if response and getattr(response, "text", None):
                return str(response.text).strip()
        except Exception:
            pass
        return baseline_text
