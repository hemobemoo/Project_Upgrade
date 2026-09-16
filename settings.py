import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Demo / Competition Mode ───────────────────────────────────────
# If True, the app runs with mock predictions when the real model is missing.
DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")

# ── Secrets (loaded from .env) ─────────────────────────────────────
OPENAI_API_KEY: str   = os.getenv("OPENAI_API_KEY", "")
FINE_TUNED_MODEL: str = os.getenv("FINE_TUNED_MODEL", "")

# Only validate secrets when NOT in demo mode
if not DEMO_MODE:
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )
    if not FINE_TUNED_MODEL:
        raise EnvironmentError(
            "FINE_TUNED_MODEL is not set. Add it to your .env file:\n"
            "  FINE_TUNED_MODEL=ft:gpt-4o-mini-..."
        )

# ── Non-secret config (overridable via .env) ───────────────────────
MODEL_PATH: Path = Path(__file__).parent / "utils" / "best_hybrid_v4_transformer.keras"
BINARY_MODEL_DIR: Path = Path(__file__).parent / "utils"

BEST_THRESHOLDS: list[float] = [
    float(x)
    for x in os.getenv("BEST_THRESHOLDS", "0.61,0.49,0.44,0.35,0.42").split(",")
]

ECG_DISPLAY_DURATION: float = float(os.getenv("ECG_DISPLAY_DURATION", "10.0"))
REPORT_TEMPERATURE: float   = float(os.getenv("REPORT_TEMPERATURE", "0.3"))

# ── Streaming config ───────────────────────────────────────────────
STREAM_WINDOW_SEC: float    = 10.0
STREAM_OVERLAP_SEC: float   = 5.0
STREAM_PRE_FILTER: bool     = True