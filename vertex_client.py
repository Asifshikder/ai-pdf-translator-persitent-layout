"""Shared Vertex AI (Gemini) client, reused by the text translator and image localizer."""

import os

from google import genai

PROJECT_ID = "vertext-ai-project-497411"
LOCATION = "global"

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vertextaiproject2.json"),
)

_client = None


def get_client() -> genai.Client:
    """Return a lazily-created, process-wide Vertex AI genai client."""
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    return _client
