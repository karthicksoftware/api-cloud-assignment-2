# Group 28 — RetailAssist presentation script

Present the live application and record your screen with your own microphone. Aim for 6–7 minutes including clicks and model responses. Read the short “Say” paragraphs; use the technical points as prompts rather than reading every bullet.

## Before recording

1. Close any older RetailAssist API/UI processes using ports 8000 and 8501, then run `zsh scripts/run-local.sh` from the project directory so the presentation uses the final code. The launcher reports a port conflict if an older copy is still running.
2. Open `http://127.0.0.1:8501` and `http://127.0.0.1:8000/docs` in separate tabs.
3. Open your report and `artifacts/models/intent/training_manifest.json` for the technical explanation.
4. On your Mac, press Shift–Command–5, select the recording area and choose your microphone under Options.

## Opening — 20 seconds

**Show:** Ticket workspace, application title and readiness status.

**Say:** “Hello, I am Karthick N A from Group 28. Our project is RetailAssist, an e-commerce customer-service copilot. We combined intent classification, sentiment analysis, summarisation and policy-based question answering in one application. It runs locally, and a support agent reviews the final reply.”

**Technical point:** Streamlit is the client; FastAPI coordinates local models, retrieval and Ollama.

## Page 1: Ticket workspace — 75 seconds

**Do:** Choose **Refund delay**, click **Analyze ticket**, then **Draft reply from policies**. Expand a policy source and edit a few words in the reply.

**Say:** “This customer reports a delayed refund. The fine-tuned DistilBERT model identifies the support intent, the sentiment model estimates the customer's tone, and Qwen summarises the issue. The reply uses retrieved store policies and includes source references. Here I can inspect the policy and edit the response. The application cannot check live refund records or carry out a transaction.”

**Cover technically:**

- DistilBERT: 27 support intents, locally trained checkpoint.
- RoBERTa: negative, neutral and positive sentiment.
- Qwen2.5 7B through local Ollama: summary and reply.
- `POST /analyze` handles analysis; `POST /chat` handles the grounded draft.

## Page 2: Policy chatbot — 60 seconds

**Do:** Ask these questions in order:

1. “How long does a refund take?”
2. “And when does that start?”
3. “And what if it has not arrived by then?”
4. “Does DemoMart provide a five-year refrigerator warranty?”

Expand a cited source after the first or second answer.

**Say:** “The chatbot answers using the policy documents and uses recent questions to support follow-ups. The return window is 30 calendar days, and an approved refund takes 5–7 business days after receipt and inspection. When the documents do not cover a question, the chatbot declines to invent an answer. Source references let the agent verify the evidence.”

**Cover technically:** Preserve the user topic across short follow-ups, retrieve passages, and generate statements with individual source references and exact supporting quotes. The backend checks source IDs, quotes, numbers and selected policy qualifications. These checks support traceability; agent review is still needed to check meaning.

## Page 3: Knowledge library — 40 seconds

**Do:** Show the eight policy documents and open one PDF if convenient.

**Say:** “We prepared eight policy documents covering orders, delivery, refunds, payments, accounts and support procedures. We indexed the PDFs once, retaining document names and page references. MiniLM converts passages into embeddings, and cosine similarity retrieves the four closest passages for a question.”

**Cover technically:** 24 PDF pages; 25 passages; 180-token chunks with 30-token overlap; 384-dimensional embeddings; a local NumPy index. The threshold is 0.35, selected using separate development questions.

## Page 4: Evaluation — 80 seconds

**Do:** Show the baseline comparison, training curve, confusion matrix and independent-ticket score. Expand the intent-error examples.

**Say:** “We fine-tuned DistilBERT on the Bitext dataset using an 80/10/10 split and three training epochs. Public-test macro-F1 is 0.998, compared with 0.988 for the TF-IDF baseline. On 81 independently prepared tickets, intent macro-F1 falls to 0.688. This gap shows why workplace validation matters. Sentiment macro-F1 is 0.975. Retrieval Recall at four is 100% on 20 answerable policy questions, but this measures retrieval coverage rather than perfect answer accuracy.”

**Cover technically:**

- Customer instruction is the input; intent is the target. Supplied responses are excluded.
- Train/validation/test sizes: 19,419 / 2,427 / 2,428 after deduplication.
- Best checkpoint selected by validation macro-F1; test data is held out.
- Learning rate `2e-5`, batch size 16, maximum training length 128, MPS device.
- Only the intent model was fine-tuned. Sentiment, embeddings and Qwen use pretrained weights.
- Related source templates can inflate public scores; the independent-ticket result exposes generalisation limits.
- All five unsupported questions were declined. The 100% fact-keyword check on the 20 known policy questions is an automatic diagnostic, not a human correctness score or evidence of performance on unseen questions.

## API and fine-tuning evidence — 45 seconds

**Do:** Switch to `http://127.0.0.1:8000/docs`. Execute `GET /health`, then show the `POST /analyze` request/response. Briefly open the training manifest.

Example request:

```json
{"text":"My approved refund has not arrived. I am frustrated and want to check its progress."}
```

**Say:** “The interface calls these FastAPI endpoints. The analysis response contains the intent, sentiment, summary and timings. The backend uses cached local models and calls Ollama over HTTP. The saved training manifest records the dataset, hyperparameters and device, and the local checkpoint is the model used by the application.”

**Technical evidence:** `artifacts/models/intent/model.safetensors`, `training_manifest.json`, `trainer_state.json`, and the recorded loss curve.

## Workplace application and closing — 40 seconds

**Show:** Your report's individual Situated Learning section.

**Say:** “For my workplace application, I selected e-commerce support as an industry analogue. RetailAssist assists triage and policy lookup while retaining agent approval. My evaluation shows that high public-test performance does not guarantee the same performance on new customer phrasing. My priority improvement is validation on representative, expert-reviewed workplace data. Authentication, privacy controls and verified CRM integration would also be needed before deployment. Thank you.”

## Finishing the submission

Stop the recording, check the audio and readability, and upload your recording to Google Drive with evaluator access. Add the link to `video_url` in `submission/group.json`, then regenerate the report:

```sh
.venv/bin/python scripts/export_individual_reports.py
```

Keep your explanation tied to the results on screen. Distinguish policy guidance from live order status, and retrieval coverage from answer correctness.
