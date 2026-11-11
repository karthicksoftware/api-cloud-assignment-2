import hashlib
import json
import re
import numpy as np
from .config import SETTINGS, EMBEDDING_MODEL
from .errors import ServiceError

def token_chunks(text, tokenizer, size=180, overlap=30):
    """WordPiece windows, including overlap, below the encoder's 256-token limit."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(ids[start:start + size], skip_special_tokens=True)
            for start in range(0, len(ids), size - overlap) if ids[start:start + size]]

def build_index(models, settings=SETTINGS):
    from pypdf import PdfReader
    embedder = models.embedder()
    chunks, documents = [], []
    for path in sorted((settings.root / "data/policies/pdf").glob("*.pdf")):
        reader = PdfReader(path)
        title = reader.metadata.title or path.stem
        documents.append({"name": path.name, "title": title, "pages": len(reader.pages),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for page_no, page in enumerate(reader.pages, 1):
            # ReportLab's page callback is serialized before the body; exclude its footer
            # so the real section heading, rather than the disclaimer, becomes metadata.
            lines = (page.extract_text() or "").splitlines()
            text = "\n".join(line for line in lines if not line.startswith("FICTIONAL DEMONSTRATION POLICY")
                             and not re.fullmatch(r"Page \d+", line.strip()))
            section = next((line.strip() for line in text.splitlines() if line.strip()), title)
            for part, chunk in enumerate(token_chunks(text, embedder.tokenizer), 1):
                chunks.append({"id": f"{path.stem}-p{page_no}-c{part}", "document": path.name,
                               "title": title, "page": page_no, "section": section, "text": chunk})
    if not chunks: raise ServiceError("index", "No readable policy PDFs. Run: retailassist policies")
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    settings.index.mkdir(parents=True, exist_ok=True)
    # Write metadata last. Runtime validates dimensions and content fingerprint before querying.
    np.savez_compressed(settings.index / "vectors.npz", vectors=np.asarray(vectors, dtype=np.float32))
    (settings.index / "chunks.json").write_text(json.dumps(chunks, indent=2))
    manifest = {"embedding_model": EMBEDDING_MODEL, "chunk_tokens": 180, "overlap_tokens": 30,
                "count": len(chunks), "documents": documents,
                "chunks_sha256": hashlib.sha256((settings.index / "chunks.json").read_bytes()).hexdigest()}
    (settings.index / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

class Retriever:
    def __init__(self, models, settings=SETTINGS):
        self.models, self.settings = models, settings
        self._signature = None

    def _load(self):
        try:
            path = self.settings.index
            signature = tuple((path / x).stat().st_mtime_ns for x in ["vectors.npz", "chunks.json", "manifest.json"])
            if signature == self._signature: return
            self.chunks = json.loads((path / "chunks.json").read_text())
            self.manifest = json.loads((path / "manifest.json").read_text())
            with np.load(path / "vectors.npz", allow_pickle=False) as data:
                self.vectors = data["vectors"].copy()
            if (len(self.vectors) != len(self.chunks) or self.vectors.ndim != 2 or
                not np.isfinite(self.vectors).all() or self.manifest["embedding_model"] != EMBEDDING_MODEL or
                hashlib.sha256((path / "chunks.json").read_bytes()).hexdigest() != self.manifest["chunks_sha256"]):
                raise ValueError("Index metadata mismatch")
            self._signature = signature
        except (OSError, ValueError, KeyError) as exc:
            raise ServiceError("index", "Policy index is missing or inconsistent. Run: retailassist index") from exc

    def search(self, query, k=4):
        self._load()
        query_vector = self.models.embedder().encode([query], normalize_embeddings=True)[0]
        scores = self.vectors @ query_vector
        indexes = np.argsort(-scores, kind="stable")[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in indexes]

    def documents(self):
        self._load()
        return self.manifest

def validate_citations(text, cited_ids, sources):
    allowed = {s["id"] for s in sources}
    if not isinstance(text, str) or not isinstance(cited_ids, list) or not cited_ids: return False
    if any(not isinstance(x, str) or x not in allowed for x in cited_ids): return False
    in_text = re.findall(r"\[([^\[\]]+)\]", text)
    return bool(in_text) and set(in_text) == set(cited_ids) and set(in_text) <= allowed

def render_citations(text, cited_ids, sources):
    """Render validated structured references; never fabricate a missing source ID."""
    allowed = {s["id"] for s in sources}
    if not isinstance(text, str) or not text.strip() or not isinstance(cited_ids, list) or not cited_ids:
        return None
    if any(not isinstance(x, str) or x not in allowed for x in cited_ids): return None
    in_text = re.findall(r"\[([^\[\]]+)\]", text)
    if in_text:
        return text if validate_citations(text, cited_ids, sources) else None
    return text.rstrip() + " " + " ".join(f"[{source_id}]" for source_id in dict.fromkeys(cited_ids))

