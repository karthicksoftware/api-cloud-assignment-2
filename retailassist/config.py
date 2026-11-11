import json
import os
from dataclasses import dataclass
from pathlib import Path
from . import ROOT

BASE_MODEL = "distilbert/distilbert-base-uncased"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
INTENTS = sorted([
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund",
])

def normalize_ollama_url(value):
    value = value.rstrip("/")
    return value if "://" in value else "http://" + value

@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    ollama_url: str = normalize_ollama_url(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    generator: str = os.environ.get("RETAILASSIST_GENERATOR", "qwen2.5:7b")
    timeout: int = int(os.environ.get("RETAILASSIST_TIMEOUT", "180"))

    @property
    def artifacts(self): return self.root / "artifacts"
    @property
    def checkpoint(self): return self.artifacts / "models" / "intent"
    @property
    def index(self): return self.artifacts / "index"
    @property
    def evaluation(self): return self.artifacts / "evaluation"
    @property
    def thresholds(self):
        path = self.artifacts / "calibration.json"
        return json.loads(path.read_text()) if path.exists() else {"retrieval": 0.35}

SETTINGS = Settings()

def local_model_path(name, settings=SETTINGS):
    """Resolve the downloaded immutable revision, without relying on a mutable main ref."""
    from huggingface_hub import snapshot_download
    manifest = settings.artifacts / "model_downloads.json"
    revision = json.loads(manifest.read_text()).get(name, {}).get("revision", "main") if manifest.exists() else "main"
    return snapshot_download(name, revision=revision, local_files_only=True)

