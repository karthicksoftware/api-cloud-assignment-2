# RetailAssist

A fully local e-commerce customer-service copilot for AIMLZG 549 Assignment II. The fictional retailer is **DemoMart**. Streamlit calls a FastAPI backend for intent classification, sentiment analysis, ticket summarization and policy-grounded chat. An agent reviews every draft; the application cannot perform transactions or access live customer records.

## Start the prepared application

```sh
zsh scripts/run-local.sh
```

Open <http://127.0.0.1:8501>. Interactive API documentation is at <http://127.0.0.1:8000/docs>. Stop the launcher with Control-C. If an earlier copy is running, stop it before restarting so that the API loads the latest code. All services bind to loopback; Ollama cloud features are disabled. No API key or paid service is used. Internet is needed for initial installation/downloads, not inference.

## Reproduce from source on Apple Silicon macOS

Python 3.12 is required. From this directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .

# Download the local Ollama runtime if it is not already present.
mkdir -p .cache/ollama
curl --fail --location https://github.com/ollama/ollama/releases/download/v0.40.15/ollama-darwin.tgz -o .cache/ollama/runtime.tgz
tar -xzf .cache/ollama/runtime.tgz -C .cache/ollama

.venv/bin/retailassist download
.venv/bin/retailassist prepare
.venv/bin/retailassist policies
.venv/bin/retailassist train
.venv/bin/retailassist index
.venv/bin/retailassist calibrate
```

In a separate terminal, start Ollama and retain that terminal:

```sh
OLLAMA_MODELS="$PWD/artifacts/ollama" OLLAMA_NO_CLOUD=1 .cache/ollama/ollama serve
```

Download Qwen once, then run evaluation:

```sh
.cache/ollama/ollama pull qwen2.5:7b
.venv/bin/retailassist evaluate
```

The main training command performs actual full-model fine-tuning. `train --smoke --epochs 1` creates a separate tiny diagnostic checkpoint and never replaces the real checkpoint. Do not use smoke metrics as assignment evidence. A full training run requires the local base-model download; MPS is selected when available and CPU is used otherwise. The training sequence length is 128; inference accepts up to 512 intent-model tokens and rejects longer tickets rather than silently truncating them.

The installed environment is recorded in `requirements.lock.txt`; model revision SHAs and dataset revision/split hashes are recorded in generated manifests. Downloads currently resolve model/dataset revisions at execution time and record them. Archive these manifests and caches to reproduce exactly the same input revisions rather than rerunning downloads against changed upstream defaults. The lock is for Python 3.12 on the tested Apple Silicon environment.

## Models and data

| Task | Model/data |
|---|---|
| Fine-tuned intent | `distilbert/distilbert-base-uncased`, 27 Bitext intents |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest`, negative/neutral/positive |
| Summarization and answers | `qwen2.5:7b`, Q4 quantization, local Ollama |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, normalized 384-dimensional vectors |
| Training | [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) |

The classifier reads only the dataset's customer instruction and intent. Supplied responses and categories are excluded from features. Exact normalized duplicates and conflicting duplicate labels are removed before stratified 80/10/10 splitting. A bounded character-ngram audit measures remaining near-template leakage. The TF-IDF baseline uses the same split. The best DistilBERT checkpoint is selected by validation macro-F1.

Eight editable synthetic policy documents are in `data/policies/source`; generated PDFs are indexed once. Each PDF has page-level sections. MiniLM-tokenized chunks contain at most 180 wordpieces with 30-token overlap. A persisted NumPy matrix implements exact cosine search over this small corpus. The backend verifies index dimensions, metadata fingerprints and source IDs. A conservative text guard rejects recognized claims of completed/promised business actions, since the application has no transaction tools. This guard is not a complete prompt-injection defense. Semantic faithfulness still requires evaluation; valid source IDs alone do not prove correctness.

The 81 workplace tickets, 30 final RAG cases and 12 development questions are checked into source. Calibration uses development questions only. Final evaluation includes public-test classification, independently authored workplace tickets, sentiment, retrieval, unsupported questions, adversarial instructions and closed-book comparisons. `evaluate --skip-generation` is a retrieval/classification diagnostic and does not establish successful answer generation.

## API examples

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' -d '{"text":"My refund is delayed and I am frustrated."}'
curl http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"How long does a refund take after inspection?"}'
```

`POST /chat` also accepts `ticket_context` and up to six `{role, content}` history turns. The response includes the answer, retrieved passages, cited source IDs, abstention reason and latency. `/knowledge` and `/evaluation` expose the document manifest and saved measurements. `/health` returns HTTP 200 with `ready: false` if setup is incomplete; it does not download or warm every model. Inference failures use structured 503/504 responses. Input failures use 422.

Environment options: `RETAILASSIST_HOME` (repository/artifact root), `OLLAMA_HOST` (loopback URL), `RETAILASSIST_GENERATOR` (default `qwen2.5:7b`), `RETAILASSIST_TIMEOUT` (default 180 seconds), `RETAILASSIST_API` (UI API URL). Keep the defaults for the evaluated configuration.

## Verification and artifacts

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_offline.py
.venv/bin/python scripts/record_runtime.py
```

Runtime verification loads real models and blocks non-loopback connections in the verification process. It requires prepared artifacts and a running local Ollama server. Backend and UI are launched with Hugging Face offline flags by the local launcher. The runtime manifest records actual package/runtime information and the Qwen model digest. Browser evidence is collected by `scripts/capture_evidence.py` against the running application using local Google Chrome and Playwright.

For the submission video, record the live application with your own microphone. Follow the speaking notes and technical checklist in [docs/demo-script.md](docs/demo-script.md). Demonstrate the four application screens, API request/response flow, saved fine-tuning evidence and your individual workplace analysis.

Before recording, restart any earlier API/UI copy so it loads the final service code. If `scripts/run-local.sh` reports ports 8000 or 8501 in use, find the older app processes with `lsof -nP -iTCP:8000 -iTCP:8501 -sTCP:LISTEN`, stop those processes from their original terminal, and rerun the launcher. Keep the local Ollama service available.

Generated artifacts include:

- `artifacts/models/intent`: real checkpoint, tokenizer, label mappings and training manifest.
- `artifacts/evaluation`: actual metrics, predictions, loss plot, confusion matrix and review worksheets.
- `artifacts/index`: normalized embeddings, passages and document fingerprints.
- `artifacts/screenshots`: browser screenshots for the report.
- `artifacts/submissions`: per-member Word/PDF reports and readiness status.

Model weights, environments, downloaded data and generated artifacts are ignored by source control. Preserve the generated evidence separately for submission. This project does not upload source code or reports automatically.

## Reports and remaining factual inputs

Fill `submission/group.json` with the actual group ID and each member's name, BITS ID, contribution and percentage. Contributions must sum to 100%. Each member writes their own four Situated Learning responses; see `docs/situated-learning.md`. Never copy an invented personal reflection into a submission.

```sh
.venv/bin/retailassist report --metadata submission/group.json
```

The same measured project content is exported to Word and PDF, followed by the member's individual Part IV. Each report is named `<groupid>.pdf` / `<groupid>.docx` within its BITS-ID directory. Missing submission facts or technical evidence produce a visibly marked draft and a machine-readable pending list. Internal human-review work is tracked separately and never marked complete by automatic checks.

For the named sharing copies requested by the group, run:

```sh
.venv/bin/python scripts/export_individual_reports.py
```

This creates `Group_28_Individual_Reports` with `28_<name>.pdf` and `28_<name>.docx` for each member. The shared project content is the same; the four individual analyses have different areas of focus. Each member should review their own section and add the group video link before submission. The portal copy should use the assignment's `28.pdf` or `28.docx` filename.

Run `.venv/bin/python scripts/verify_conversation.py` for the six real-model conversation/qualification checks, then `.venv/bin/python scripts/review_outputs.py` for the automated semantic review. The latter uses Qwen2.5 and records its model identity, rationales and input hashes; it is not independent human validation. The report uses the review only while its input hashes match the current predictions.

Use `submission/evidence-review-pack.md` to review the actual policy answers and summaries. The protocol in `docs/evaluation-review.md` explains how a member can record genuine human ratings. These worksheets are supporting QA, not additional mandatory portal uploads. Five paired timing tasks remain optional unless claiming measured human efficiency gains; no such gains are claimed here.

Both report and video are compulsory. Use **12 November 2026** as the conservative working deadline because the announcement conflicts with the portal's 15 November date. Upload the video to Google Drive, verify evaluator access, add its link to the metadata, and regenerate the report. Every member submits the shared project content with their own Part IV.

## Attribution

Bitext dataset: **CDLA-Sharing-1.0**, [dataset card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Preserve its license and attribution when sharing data or derivatives. DistilBERT and MiniLM: Apache-2.0. Qwen2.5 7B: Apache-2.0. CardiffNLP sentiment model: CC-BY-4.0; credit Cardiff NLP and the TimeLMs authors using the [model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest). Downloaded runtime/model documentation retains its upstream notices. The .example support address is fictional and does not receive messages.
