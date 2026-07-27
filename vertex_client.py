"""Shared Vertex AI (Gemini) client, reused by the text translator and image localizer.

Two service accounts are configured: the tasmiya-ar project is primary, and the
older vertext-ai-project account is kept as a fallback (e.g. for when the primary
project runs out of quota).

Resilience:
  * Transient errors (HTTP 429 / RESOURCE_EXHAUSTED, 503 / UNAVAILABLE, etc.) are
    retried on the same project with exponential backoff + jitter, honoring the
    server's retry hint when it sends one.
  * Only after those retries are exhausted do we try the other project. The switch
    is NOT permanent: a persistent primary failure puts the primary on a short
    cooldown (calls go straight to the fallback during it), then we probe the
    primary again. One 429 no longer poisons the whole process.
"""

import logging
import os
import random
import re
import time

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCATION = "global"

PRIMARY_PROJECT_ID = "rapid-strength-477714-f8"
PRIMARY_CREDENTIALS = os.path.join(_BASE_DIR, "vertextkey-tasmiya-ar.json")

FALLBACK_PROJECT_ID = "vertext-ai-project-497411"
FALLBACK_CREDENTIALS = os.path.join(_BASE_DIR, "vertextaiproject2.json")

# Per-request HTTP timeouts (milliseconds). Without a client-side timeout a stalled
# Vertex request (half-open socket, server that never responds) blocks generate_content
# forever — the retry/fallback logic below only reacts to exceptions, and a silent hang
# never raises one. A timeout turns that hang into a retryable error.
DEFAULT_TIMEOUT_MS = 180_000  # 180s — text / classify calls
IMAGE_TIMEOUT_MS = 300_000    # 300s — image-generation calls (legitimately slower)

# Retry/backoff tuning for transient (rate-limit / unavailable) errors.
MAX_RETRIES = 5
BASE_DELAY = 2.0          # seconds; grows 2, 4, 8, 16, ...
MAX_DELAY = 60.0          # cap per-sleep
PRIMARY_COOLDOWN = 120.0  # skip the primary for this long after it persistently fails

# Substrings that mark an error worth retrying / switching projects for.
_TRANSIENT_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "429",
    "503",
    "500",
    # Client-side stalls once a request timeout is in force (see DEFAULT_TIMEOUT_MS).
    # httpx raises ReadTimeout/ConnectTimeout; Vertex may wrap it as DEADLINE_EXCEEDED.
    "Timeout",
    "timed out",
    "ReadTimeout",
    "ConnectTimeout",
)

_primary_client = None
_fallback_client = None
_primary_cooldown_until = 0.0


def _build_client(project_id: str, credentials_path: str) -> genai.Client:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=LOCATION,
        # Default ceiling for every call, so nothing can hang forever even if a caller
        # forgets to pass timeout_ms. Image calls override this with IMAGE_TIMEOUT_MS.
        http_options=types.HttpOptions(timeout=DEFAULT_TIMEOUT_MS),  # milliseconds
    )


def _get_primary_client() -> genai.Client:
    global _primary_client
    if _primary_client is None:
        _primary_client = _build_client(PRIMARY_PROJECT_ID, PRIMARY_CREDENTIALS)
    return _primary_client


def _get_fallback_client() -> genai.Client:
    global _fallback_client
    if _fallback_client is None:
        _fallback_client = _build_client(FALLBACK_PROJECT_ID, FALLBACK_CREDENTIALS)
    return _fallback_client


def _is_transient(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code in (429, 500, 503):
        return True
    msg = str(e)
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def _retry_hint_seconds(e: Exception) -> float | None:
    """Pull a server-provided retry delay out of the error, if present.

    Vertex returns hints like `retryDelay: '37s'` or `retry_delay { seconds: 37 }`.
    """
    m = re.search(r"retry[_ ]?delay['\"]?\s*[:={]\s*(?:['\"]|seconds:\s*)?(\d+)", str(e), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _call_with_retry(client: genai.Client, kwargs: dict):
    """generate_content on one client, retrying transient errors with backoff + jitter."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:  # noqa: BLE001 — we classify below
            last_exc = e
            if not _is_transient(e) or attempt == MAX_RETRIES:
                raise
            hint = _retry_hint_seconds(e)
            delay = hint if hint is not None else min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)  # jitter to de-sync concurrent workers
            logger.warning(
                "Transient error (%s) on attempt %d/%d; backing off %.1fs: %s",
                type(e).__name__, attempt, MAX_RETRIES, delay, str(e)[:150],
            )
            time.sleep(delay)
    raise last_exc  # pragma: no cover — loop always returns or raises


def _apply_timeout(kwargs: dict, timeout_ms: int) -> None:
    """Set http_options.timeout on the request's config, in place.

    The config may be a GenerateContentConfig, a plain dict, or absent. We keep whatever
    the caller passed and only fill in the timeout, without clobbering an http_options the
    caller may already have set.
    """
    config = kwargs.get("config")
    if config is None:
        kwargs["config"] = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=timeout_ms)
        )
    elif isinstance(config, dict):
        config.setdefault("http_options", types.HttpOptions(timeout=timeout_ms))
    elif getattr(config, "http_options", None) is None:
        config.http_options = types.HttpOptions(timeout=timeout_ms)


def generate_content(*, timeout_ms: int | None = None, **kwargs):
    """generate_content with transient-error backoff and a non-permanent project fallback.

    Tries the preferred project (primary unless it is on cooldown) with retries, then the
    other project with retries. A persistent primary failure starts a short cooldown so the
    process doesn't keep paying the retry cost on a known-exhausted project, but the primary
    is probed again once the cooldown lapses.

    timeout_ms overrides the client's DEFAULT_TIMEOUT_MS for this call (pass IMAGE_TIMEOUT_MS
    for image generation). When omitted, the client-level default applies.
    """
    global _primary_cooldown_until
    if timeout_ms is not None:
        _apply_timeout(kwargs, timeout_ms)
    now = time.time()
    if now < _primary_cooldown_until:
        order = [("fallback", _get_fallback_client), ("primary", _get_primary_client)]
    else:
        order = [("primary", _get_primary_client), ("fallback", _get_fallback_client)]

    last_exc = None
    for i, (name, getter) in enumerate(order):
        try:
            return _call_with_retry(getter(), kwargs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if name == "primary":
                _primary_cooldown_until = time.time() + PRIMARY_COOLDOWN
                logger.warning(
                    "Primary project exhausted (%s); cooling down %.0fs%s",
                    str(e)[:150], PRIMARY_COOLDOWN,
                    ", trying fallback" if i == 0 else "",
                )
            else:
                logger.warning("Fallback project failed (%s)", str(e)[:150])
    raise last_exc
