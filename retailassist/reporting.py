"""Generate matching Word/PDF reports from measured artifacts and factual metadata."""
import json
import re
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape
from .config import SETTINGS
from .training import dump
from .presentation import rag_measurements
from .review import current_review

REFERENCES = [
    ("Assignment requirements", "Questions/Assignment 2.pdf"),
    ("Submission announcement", "Questions/Instructions.png"),
    ("Bitext dataset and CDLA-Sharing-1.0 license", "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"),
    ("DistilBERT fine-tuning", "https://huggingface.co/docs/transformers/tasks/sequence_classification"),
    ("Sentiment model and CC-BY-4.0 attribution", "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest"),
    ("MiniLM embedding model", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"),
    ("Qwen2.5 7B, Apache-2.0", "https://ollama.com/library/qwen2.5:7b"),
    ("Local Ollama chat API", "https://docs.ollama.com/api/chat"),
]

SITUATED_SECTIONS = [
    ("anchor", "Anchor - workplace problem (1 mark)"),
    ("apply", "Apply - adaptation and evidence (2 marks)"),
    ("analyse", "Analyse - workplace relevance (1 mark)"),
    ("reflect", "Reflect - constraints and priority improvement (1 mark)"),
]

def read_json(path, default=None): return json.loads(path.read_text()) if path.exists() else default
def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Group and BITS identifiers must contain only letters, numbers, underscores and hyphens.")
    return value


def evidence_text(value, settings=SETTINGS):
    """Resolve measured RAG values in individual narratives at export time."""
    rag = read_json(settings.evaluation / "rag_metrics.json", {})
    values = {
        "rag_keyword_percent": f"{rag.get('fact_keyword_pass_rate', 0):.0%}",
        "rag_keyword_passes": str(round(rag.get('fact_keyword_pass_rate', 0)*rag.get('answerable_count', 0))),
        "rag_mean_seconds": f"{rag.get('mean_latency_ms', 0)/1000:.2f}",
    }
    for key, replacement in values.items(): value = value.replace("{" + key + "}", replacement)
    return value

def common_blocks(metadata, settings=SETTINGS):
    blocks = []
    def heading(text): blocks.append(("heading", text))
    def paragraph(text): blocks.append(("text", text))
    def table(rows): blocks.append(("table", rows))
    def image(path, caption):
        if path.exists(): blocks.extend([("image", path), ("caption", caption)])
    training = read_json(settings.evaluation / "training_metrics.json", {})
    workplace = read_json(settings.evaluation / "workplace_metrics.json", {})
    rag = read_json(settings.evaluation / "rag_metrics.json", {})
    manifest = read_json(settings.root / "data/processed/manifest.json", {})
    index = read_json(settings.index / "manifest.json", {})
    heading("RetailAssist: API-driven customer-service copilot")
    paragraph("AIMLZG 549 - API-driven Cloud Native Solutions | Assignment II | Group: " + metadata["group_id"])
    paragraph("We selected e-commerce customer support as the project domain and modelled DemoMart as an Indian retail industry analogue. We used public customer-service data and prepared representative store policies and evaluation tickets for the prototype. These materials are synthetic and do not contain real customer records.")
    table([["BITS ID", "Name", "Contribution", "Percentage"]] + [[m["bits_id"], m["name"], m.get("contribution") or "TO BE PROVIDED BY MEMBER", str(m["percentage"]) + "%" if m.get("percentage") is not None else "PENDING"] for m in metadata["members"]])
    paragraph("The contribution table records the group's allocation of work. The common project sections are shared by the group; each member completes an individual Situated Learning section.")
    blocks.append(("pagebreak", None))
    heading("I. API-based AI application (4 marks)")
    paragraph("We developed RetailAssist to assist customer-service agents with ticket triage, reading and policy lookup. Our application combines four cohesive NLP tasks: intent classification, sentiment analysis, ticket summarization and grounded question answering. The agent reviews the suggested response and its evidence before using it.")
    table([["Task", "Model", "Output"], ["Intent classification", "Locally fine-tuned DistilBERT", "27 support intents with top-three scores"],
        ["Sentiment", "CardiffNLP Twitter-RoBERTa sentiment", "Negative, neutral and positive probabilities"],
        ["Summarization", "Qwen2.5 7B, quantized, local Ollama", "Short factual ticket summary"],
        ["Question answering", "Qwen2.5 + MiniLM retrieval", "Policy-based draft with source citations"]])
    paragraph("Architecture: Streamlit client -> FastAPI -> local inference and retrieval -> Ollama generator. The UI contains a ticket workspace, policy chatbot, knowledge library and evaluation dashboard. Initial package/model downloads require internet; runtime inference uses cached models and a loopback-only generator with cloud features disabled. Classifier inference uses CPU; training uses MPS when available.")
    table([["Endpoint", "Contract"], ["GET /health", "Readiness and selected models"], ["POST /analyze", "Ticket text -> intent, sentiment, summary and timings"], ["POST /chat", "Question, optional ticket, six recent turns -> answer, sources, abstention and timing"], ["GET /knowledge", "Document manifest and index metadata"], ["GET /evaluation", "Saved measurements; no invented scores"]])
    for name, caption in [("01-workspace.png", "Ticket analysis and editable grounded draft"), ("06-workspace-source.png", "Expanded source passage for the drafted reply"), ("02-chatbot.png", "Policy conversation with cited evidence"), ("03-knowledge.png", "Indexed fictional policies")]: image(settings.artifacts / "screenshots" / name, caption)
    heading("II. Dataset preparation and fine-tuning (2 marks)")
    paragraph("We used the Bitext customer-support dataset with instruction as the only input and intent as the target. We excluded supplied response text and category to avoid label leakage, normalised whitespace, removed duplicate case-insensitive messages and excluded conflicting duplicate labels. We created stratified train/validation/test splits using an 80/10/10 ratio and seed 42, retaining the dataset revision and raw-data and split hashes for reproducibility.")
    if manifest:
        audit = manifest["near_duplicate_audit"]
        audit_summary = f"{audit['near_duplicate_count']} of {audit['sampled_test_rows']} sampled test messages had a close training match at similarity {audit['similarity_threshold']:.2f} or above. Related templates may inflate public-test performance."
        table([["Measure", "Recorded value"], ["Dataset revision", manifest["revision"]], ["Original / deduplicated rows", f"{manifest['original_rows']} / {manifest['deduplicated_rows']}"], *[[f"{k.title()} rows", str(v["rows"])] for k,v in manifest["split"].items()], ["Near-template audit", audit_summary]])
    else: paragraph("Dataset preparation results: PENDING execution.")
    paragraph("We fine-tuned the DistilBERT encoder and a 27-label classification head. Configuration: 3 epochs, learning rate 2e-5, batch size 16, maximum sequence length 128, weight decay 0.01, warmup ratio 0.1 and seed 42. We selected the checkpoint using validation macro-F1 and compared it with a TF-IDF unigram/bigram logistic-regression baseline on the same held-out test split. We did not use the test set for checkpoint selection.")
    if training:
        table([["Model", "Test accuracy", "Test macro-F1"], *[[name, f"{training[key]['accuracy']:.4f}", f"{training[key]['macro_f1']:.4f}"] for name,key in [("TF-IDF baseline", "baseline"), ("Fine-tuned DistilBERT", "fine_tuned")]]])
        paragraph(f"Recorded device: {training['training']['device']}. End-to-end training/evaluation duration: {training['duration_seconds']:.1f} seconds. The saved checkpoint, tokenizer, label mappings, trainer state and provenance manifest are retained locally.")
        image(settings.evaluation / "training_curve.png", "Recorded training and validation loss")
        image(settings.evaluation / "confusion_matrix.png", "Public held-out test confusion matrix")
    else: paragraph("Training results and checkpoint evidence: PENDING execution. No performance result is claimed.")
    heading("III. Retrieval-augmented policy chatbot (4 marks)")
    paragraph("We prepared eight policy documents covering orders, delivery, refunds, payments, accounts, support, newsletters and agent procedures. We converted the editable sources to PDFs and indexed only the PDFs to avoid duplication. We split page text into 180-WordPiece chunks with 30-token overlap, below MiniLM's 256-token limit. We stored normalised 384-dimensional embeddings in a NumPy index alongside document names, sections, pages and stable source identifiers. Search returns four passages using cosine similarity.")
    paragraph(f"Recorded index: {len(index.get('documents', []))} documents and {index.get('count', 'PENDING')} passages. Retrieval threshold: {settings.thresholds['retrieval']}. Calibration used 12 separate development examples; the threshold remained fixed during the subsequent regression evaluation.")
    paragraph("For most questions, Qwen selects a source and an exact supporting sentence from the retrieved excerpts for each answer statement, then drafts the wording. The backend checks the source identifier, evidence quotation, numeric claims and selected timing qualifications before placing a citation beside the statement. If a paraphrase drops a checked qualification, the original policy sentence is used. For common procedural lookups, the backend quotes the documented invoice-download path, support hours or steps to edit an unshipped order's address. It can also quote an explicit prohibition when generation mistakes 'not permitted' for 'not documented'. Unsupported or invalid answers are declined. These checks catch specific errors but do not prove complete semantic faithfulness.")
    paragraph("Short follow-ups retain the most recent explicit subject from the user's questions; an explicit new topic starts a new retrieval query. Earlier assistant answers are not retrieval evidence. The generator uses an 8,192-token context window. The prototype has no live order or payment records and discloses that limitation for record-specific requests.")
    paragraph("An initial adversarial test elicited a false claim that a refund had been issued. A conservative backend text guard was added to reject recognized claims of completed or promised business actions; the original failure is retained as regression evidence. This guard is not a complete defense against prompt injection, and agent review remains necessary.")
    if rag:
        table([["RAG measure", "Recorded value"], *rag_measurements(rag)])
        paragraph("These 30 known cases form a regression evaluation after fixes to conversation handling and policy grounding. Document Recall@4 measures whether the expected document occurs among the four retrieved passages. The keyword and exact-quote checks are automatic diagnostics. The adversarial result is a retest of known attacks, not evidence of general resistance to unseen attacks. Earlier measurements and failures are retained for comparison.")
    conversation = read_json(settings.evaluation / "conversation_checks.json", {})
    if conversation:
        cases = conversation.get("cases", [])
        paragraph(f"Conversation and policy-qualification checks: {sum(x['passed'] for x in cases)}/{len(cases)} passed. The recorded sequence asks about refund timing, then 'And when does that start?', then 'And what if it has not arrived by then?', followed by an explicit password-recovery topic change. Additional checks cover a verified account-deletion target and support outside operating hours.")
    predictions = read_json(settings.evaluation / "rag_predictions.json", [])
    for case_id in ["qa02", "qa14", "out03", "adv03"]:
        case = next((x for x in predictions if x["id"] == case_id and "response" in x), None)
        if case:
            paragraph("Example " + case_id + " - " + case["question"])
            paragraph(case["response"]["answer"])
            if case_id == "qa02" and case.get("without_retrieval"):
                paragraph("Same question without retrieval: " + case["without_retrieval"])
    review = current_review(settings)
    if review:
        totals = review["totals"]
        heading("Answer and summary quality checks")
        paragraph("We also ran a separate rubric-based review of the saved outputs using local Qwen2.5. It compared answers with policy evidence and summaries with their original tickets, recording a rationale for each judgement. This is an automated assessment using the same model family as the generator, so correlated errors and optimistic ratings remain possible. It does not replace independent human validation.")
        table([["Automated review measure", "Recorded count"],
               ["Answerable responses judged correct", f"{totals['answerable_correct']}/{totals['answerable_count']}"],
               ["Answerable responses judged fully supported", f"{totals['answerable_supported']}/{totals['answerable_count']}"],
               ["Summaries retaining the main facts", f"{totals['summaries_retaining_main_facts']}/{totals['summary_count']}"],
               ["Summaries without invented facts", f"{totals['summaries_without_invented_facts']}/{totals['summary_count']}"]])
    heading("Evaluation and practical limitations")
    if workplace:
        table([["Independent workplace set", "Accuracy", "Macro-F1"], ["Intent, 81 tickets", f"{workplace['intent']['accuracy']:.3f}", f"{workplace['intent']['macro_f1']:.3f}"], ["Sentiment, 81 tickets", f"{workplace['sentiment']['accuracy']:.3f}", f"{workplace['sentiment']['macro_f1']:.3f}"]])
        errors = [x for x in read_json(settings.evaluation / "workplace_predictions.json", []) if x["intent"] != x["predicted_intent"]][:4]
        if errors:
            paragraph("Representative intent errors show why high model confidence should not replace agent review:")
            table([["Ticket excerpt", "Expected", "Predicted", "Confidence"]] + [[textwrap.shorten(x["text"], width=125, placeholder="…"), x["intent"].replace("_", " "), x["predicted_intent"].replace("_", " "), f"{x['confidence']:.2f}"] for x in errors])
    paragraph("The 81 workplace tickets were authored separately from training examples, with three per intent and one intended negative/neutral/positive formulation per intent. They remain synthetic and single-author labeled; ambiguous labels and mixed sentiment can affect scores. Ten generated summaries are exported for fact-retention and invented-fact review. Five timed manual-versus-assisted tasks require a real participant; no efficiency improvement is claimed without those measurements.")
    paragraph("Public test performance may be optimistic because the source contains closely related synthetic templates. Sentiment was trained on social-media text rather than this retailer's tickets. Retrieval similarity is not certainty, and a local language model can misread a policy despite valid citations. The prototype is a single-machine, agent-reviewed demonstration without authentication, a CRM integration or live transaction tools.")
    image(settings.artifacts / "screenshots/04-evaluation.png", "Dashboard of actual recorded measurements")
    image(settings.artifacts / "screenshots/05-policy-evaluation.png", "Recorded policy-answer checks and validation limits")
    heading("Reproduction, privacy and deployment constraints")
    paragraph("Install Python 3.12 dependencies from requirements.lock.txt. Download the selected local models and Bitext data; generate policies; train the classifier; build the index; calibrate; evaluate; then start the API and Streamlit. Exact commands are in README.md. Keep weights, environments and generated caches out of source control. Model revisions and dataset hashes are recorded in artifacts.")
    table([["Purpose", "Command from the project directory"],
           ["Install", "python3.12 -m venv .venv\n.venv/bin/pip install -r requirements.lock.txt\n.venv/bin/pip install --no-deps -e ."],
           ["Prepare and train", ".venv/bin/retailassist download\n.venv/bin/retailassist prepare\n.venv/bin/retailassist policies\n.venv/bin/retailassist train"],
           ["Retrieve and evaluate", ".venv/bin/retailassist index\n.venv/bin/retailassist calibrate\n.venv/bin/retailassist evaluate"],
           ["Run the prepared application", "zsh scripts/run-local.sh"],
           ["Verify and review outputs", ".venv/bin/python -m pytest -q\n.venv/bin/python scripts/verify_conversation.py\n.venv/bin/python scripts/review_outputs.py"],
           ["Export all reports", ".venv/bin/python scripts/export_individual_reports.py"]])
    paragraph("Ollama must be running with qwen2.5:7b downloaded before generation or evaluation. The README gives the tested runtime installation and model-pull commands. The prepared UI is at http://127.0.0.1:8501 and API documentation is at http://127.0.0.1:8000/docs.")
    paragraph("Only synthetic data should be entered into this demonstration. Conversations are kept in Streamlit session memory and can be cleared; backend logs contain request IDs, routes and timings rather than ticket bodies. Production requires authenticated access, retention and deletion policies, access-controlled order integration, security testing, policy ownership and legal review appropriate to the organization. Local execution avoids usage fees but still consumes hardware resources and maintenance time.")
    heading("References and attribution")
    for title, url in REFERENCES: paragraph(title + ": " + url)
    video_status = metadata.get("video_url") or "Our screen-recorded presentation will demonstrate the application workflow, API requests, fine-tuning results, policy-grounded answers and workplace application. The recording link will be provided with the submission."
    paragraph("Video demonstration: " + video_status)
    return blocks

def export_blocks(blocks, target, draft, author="Group 28"):
    from docx import Document
    from docx.shared import Inches, Pt
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    document = Document()
    document.sections[0].page_width = Inches(8.27)
    document.sections[0].page_height = Inches(11.69)
    document.core_properties.author = author
    document.core_properties.last_modified_by = author
    document.core_properties.title = "RetailAssist - Assignment II"
    document.core_properties.comments = ""
    document.styles["Normal"].font.size = Pt(10)
    if draft: document.add_paragraph("DRAFT — factual member details and/or review evidence remain incomplete.")
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 13
    styles["BodyText"].wordWrap = "LTR"
    styles["Heading1"].textColor = colors.HexColor("#214c3d")
    story = []
    if draft: story += [Paragraph("DRAFT - member details and/or review evidence pending", styles["Heading2"])]
    for kind, value in blocks:
        if kind == "pagebreak":
            document.add_page_break()
            story.append(PageBreak())
        elif kind == "heading":
            document.add_heading(value, level=1)
            story += [Spacer(1, 12), Paragraph(escape(value), styles["Heading1"])]
        elif kind in ["text", "caption"]:
            document.add_paragraph(str(value), style="Caption" if kind == "caption" else None)
            story += [Paragraph(escape(str(value)).replace("\n", "<br/>"), styles["BodyText"]), Spacer(1, 8)]
        elif kind == "table":
            table = document.add_table(rows=1, cols=len(value[0])); table.style = "Light Shading Accent 1"
            for cell, text in zip(table.rows[0].cells, value[0]): cell.text = str(text)
            for row in value[1:]:
                for cell, text in zip(table.add_row().cells, row): cell.text = str(text)
            data = [[Paragraph(escape(str(cell)).replace("\n", "<br/>"), styles["BodyText"]) for cell in row] for row in value]
            width = A4[0] - 96
            ratios = ([.20, .18, .49, .13] if value[0][0] == "BITS ID" else
                      [.45, .21, .21, .13] if value[0][0] == "Ticket excerpt" else
                      [.27, .73] if value[0][0] == "Purpose" else [1/len(value[0])]*len(value[0]))
            table.autofit = False
            for row in table.rows:
                for cell, ratio in zip(row.cells, ratios): cell.width = Inches(6.0*ratio)
            pdf_table = Table(data, colWidths=[width*x for x in ratios], repeatRows=1, hAlign="LEFT")
            pdf_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce9df")),
                ("VALIGN", (0,0), (-1,-1), "TOP"), ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#d2ddd3")),
                ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
            story += [pdf_table, Spacer(1, 10)]
        elif kind == "image":
            from PIL import Image as PILImage
            width, height = PILImage.open(value).size
            scale = min((A4[0]-96)/width, 430/height)
            story += [Image(str(value), width=width*scale, height=height*scale)]
            inch_scale = min(6/width, 7.5/height)
            document.add_picture(str(value), width=Inches(width*inch_scale))
    document.save(target.with_suffix(".docx"))
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(48, 28, "RetailAssist | Assignment II | " + ("DRAFT" if draft else "Project report"))
        canvas.drawRightString(A4[0]-48, 28, str(doc.page))
    SimpleDocTemplate(str(target.with_suffix(".pdf")), pagesize=A4, leftMargin=48, rightMargin=48,
        topMargin=42, bottomMargin=48, title="RetailAssist Assignment II", author=author).build(story, onFirstPage=footer, onLaterPages=footer)

def individual_blocks(member, settings=SETTINGS):
    blocks = [("pagebreak", None), ("heading", "IV. Situated Learning – Workplace Application (individual, 5 marks)"),
              ("text", f"Individual author: {member['name']} ({member['bits_id']})")]
    for key, title in SITUATED_SECTIONS:
        value = member.get("situated_learning", {}).get(key)
        if value: value = evidence_text(value, settings)
        if key == "reflect" and value:
            blocks.append(("pagebreak", None))
        blocks += [("heading", title), ("text", value or f"TO BE WRITTEN BY THIS MEMBER. Use the prompts and measured project evidence in docs/situated-learning.md for {key}.")]
    return blocks

def build_reports(metadata_path, settings=SETTINGS):
    path = Path(metadata_path)
    metadata = json.loads(path.read_text())
    group_id = safe_name(metadata["group_id"])
    members = metadata["members"]
    if not members: raise ValueError("At least one member is required.")
    ids = [safe_name(m["bits_id"]) for m in members]
    if len(ids) != len(set(ids)): raise ValueError("BITS IDs must be unique.")
    percentages = [m.get("percentage") for m in members]
    if all(x is not None for x in percentages):
        if any(not isinstance(x, (int, float)) or x < 0 or x > 100 for x in percentages) or abs(sum(percentages)-100) > .001:
            raise ValueError("Contribution percentages must be between 0 and 100 and sum to 100.")
    pending = []
    if "REQUIRED" in group_id: pending.append("group ID")
    for m in members:
        for key in ["name", "bits_id", "contribution", "percentage"]:
            if m.get(key) is None or not str(m.get(key, "")).strip() or "REQUIRED" in str(m.get(key)):
                pending.append(f"{m['bits_id']}: {key}")
    if not metadata.get("video_url"): pending.append("accessible Google Drive video link")
    if not (settings.evaluation / "training_metrics.json").exists(): pending.append("actual fine-tuning results")
    rag = read_json(settings.evaluation / "rag_metrics.json", {})
    if not rag.get("generation_evaluated"): pending.append("recorded RAG generation evaluation")
    quality_reviews = []
    if rag.get("human_correctness_review") != "complete": quality_reviews.append("human RAG answer review")
    summaries = read_json(settings.evaluation / "summary_review.json", {})
    if summaries.get("status") != "complete": quality_reviews.append("human summary review")
    common = common_blocks(metadata, settings)
    outputs = []
    for member in members:
        personal = individual_blocks(member, settings)
        for key, _ in SITUATED_SECTIONS:
            value = member.get("situated_learning", {}).get(key)
            if not value: pending.append(f"{member['bits_id']}: individual {key}")
        directory = settings.artifacts / "submissions" / safe_name(member["bits_id"])
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / group_id
        export_blocks(common + personal, target, draft=bool(pending), author=member["name"])
        outputs.extend([str(target.with_suffix(".docx")), str(target.with_suffix(".pdf"))])
    result = {"submission_ready": not pending, "pending": sorted(set(pending)),
              "quality_reviews_pending": quality_reviews,
              "automated_semantic_review_complete": bool(current_review(settings)), "reports": outputs,
              "note": "Each member uploads their own report with shared Parts I–III and individually written Part IV."}
    dump(settings.artifacts / "submission_status.json", result)
    return result
