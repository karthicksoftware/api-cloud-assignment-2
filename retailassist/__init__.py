"""RetailAssist: all model execution is local; downloads are explicit CLI actions."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("RETAILASSIST_HOME", Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

