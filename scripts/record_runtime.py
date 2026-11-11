"""Record tested package/model/runtime provenance without secrets or ticket content."""
import hashlib
import json
import platform
import sys
import requests
import torch
from retailassist.config import SETTINGS
from retailassist.training import dump

def get(path):
    response = requests.get(SETTINGS.ollama_url + path, timeout=5)
    response.raise_for_status()
    return response.json()

tags = get("/api/tags")["models"]
generator = next(model for model in tags if model["name"] == SETTINGS.generator)
manifest = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
            "ollama": get("/api/version"), "generator": generator,
            "requirements_lock_sha256": hashlib.sha256((SETTINGS.root/"requirements.lock.txt").read_bytes()).hexdigest(),
            "huggingface_models": json.loads((SETTINGS.artifacts/"model_downloads.json").read_text())}
dump(SETTINGS.artifacts/"runtime_manifest.json", manifest)
print(json.dumps(manifest,indent=2))
