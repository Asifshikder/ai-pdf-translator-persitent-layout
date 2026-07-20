"""Shared Vertex AI (Gemini) client, reused by the text translator and image localizer.

Two service accounts are configured: the tasmiya-ar project is primary, and the
older vertext-ai-project account is kept as a fallback (e.g. for when the
primary project runs out of quota). The switch is one-way and process-wide:
once the primary fails, every subsequent call goes straight to the fallback
instead of re-probing a project that is likely still exhausted.
"""

import logging
import os

from google import genai

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCATION = "global"

PRIMARY_PROJECT_ID = "rapid-strength-477714-f8"
PRIMARY_CREDENTIALS = os.path.join(_BASE_DIR, "vertextkey-tasmiya-ar.json")

FALLBACK_PROJECT_ID = "vertext-ai-project-497411"
FALLBACK_CREDENTIALS = os.path.join(_BASE_DIR, "vertextaiproject2.json")

_primary_client = None
_fallback_client = None
_use_fallback = False


def _build_client(project_id: str, credentials_path: str) -> genai.Client:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    return genai.Client(vertexai=True, project=project_id, location=LOCATION)


def get_client() -> genai.Client:
    """Return the active process-wide Vertex AI client (primary unless it has failed)."""
    global _primary_client, _fallback_client
    if _use_fallback:
        if _fallback_client is None:
            _fallback_client = _build_client(FALLBACK_PROJECT_ID, FALLBACK_CREDENTIALS)
        return _fallback_client
    if _primary_client is None:
        _primary_client = _build_client(PRIMARY_PROJECT_ID, PRIMARY_CREDENTIALS)
    return _primary_client


def generate_content(**kwargs):
    """generate_content on the active client, falling back to the secondary
    project once if the primary raises. After a fallback, later calls skip
    straight to the fallback client."""
    global _use_fallback
    try:
        return get_client().models.generate_content(**kwargs)
    except Exception:
        if _use_fallback:
            raise
        logger.warning("Primary Vertex AI project failed; switching to fallback project")
        _use_fallback = True
        return get_client().models.generate_content(**kwargs)
