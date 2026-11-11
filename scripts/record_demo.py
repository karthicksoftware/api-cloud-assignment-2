"""Record real browser interactions and add explicitly synthetic, offline narration."""
import json
import os
import subprocess
import time
import wave
from pathlib import Path
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path(__file__).resolve().parents[1] / ".cache/playwright"))
from playwright.sync_api import sync_playwright
from imageio_ffmpeg import get_ffmpeg_exe
from retailassist.config import SETTINGS

def record():
    directory = SETTINGS.artifacts / "demo"
    directory.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((SETTINGS.evaluation / "training_metrics.json").read_text())
    workplace = json.loads((SETTINGS.evaluation / "workplace_metrics.json").read_text())
    rag = json.loads((SETTINGS.evaluation / "rag_metrics.json").read_text())
    assert rag["generation_evaluated"], "Complete the full evaluation before recording."
    screenshots = SETTINGS.artifacts / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    scripts = {
        "intro": """Welcome to RetailAssist, a local customer-service copilot built for Assignment Two in API-driven Cloud Native Solutions. This recording shows the actual running application. The narration is generated offline using the computer's text-to-speech voice. DemoMart is a fictional Indian e-commerce retailer, and every workplace ticket and policy in this demonstration is synthetic. The intended user is a support agent. Our unified workflow combines four natural-language tasks: classifying a ticket's intent, identifying sentiment, summarizing the customer's issue, and answering policy questions with retrieved evidence. The agent reviews the output and remains responsible for the final reply.""",
        "analysis": """We begin with a delayed-refund ticket. The customer says the returned item was received and inspected nine business days ago, but the money has not arrived. Notice the distinction between facts supplied by the customer and facts independently verified by the system. The prototype cannot verify that refund or inspect a bank transaction. Selecting Analyze ticket calls the FastAPI backend. The intent is produced by our locally fine-tuned DistilBERT classifier. A separate Cardiff NLP model estimates negative, neutral and positive sentiment. Qwen generates a short factual summary locally. The displayed confidence is a model score, not a guarantee that the intent is correct. Agents should check the original message, particularly when it contains multiple issues.""",
        "draft": """Next, we generate a reply from the store policies. The backend embeds the question and customer context, retrieves relevant passages and sends those passages to the local Qwen model. A grounded reply should explain the documented refund timeline and the appropriate next step. The policy says an approved refund takes five to seven business days after the returned item is received and inspected. The assistant must not claim that it issued a refund or checked this customer's payment. Here we expand a source to inspect the policy name, page and actual retrieved text. The reply remains editable. This is an agent-assistance workflow, with a person checking the evidence before responding.""",
        "chat": """The policy chatbot offers a second way to use the same knowledge base. First we ask how long an unused item can be returned after delivery. The answer should use the return policy rather than the model's general knowledge. Each supported answer includes source identifiers that link it to the retrieved passages. We then ask a follow-up about how long the refund takes after inspection. Recent user questions provide context for follow-ups. Earlier assistant messages are not treated as authoritative evidence. The backend validates that citations refer to passages actually retrieved for this answer. This identity check is useful, but it does not by itself prove every sentence is faithful to the source.""",
        "unsupported": """We also test a question about a five-year refrigerator warranty. There is no such warranty policy in our knowledge base. The system should acknowledge insufficient information and recommend a specialist rather than invent warranty terms. A similar limitation applies to private company information and live customer records. Evaluation includes unsupported questions and attempts to override the policies.""",
        "knowledge": """The knowledge library contains eight fictional policy PDFs. They cover orders and cancellations, delivery, returns, payments, accounts, human support, newsletters and support-agent procedures. Editable source documents generate the PDFs, and only the PDFs are indexed, avoiding duplicate content. Text is extracted with page metadata and divided into windows of one hundred and eighty wordpieces with thirty tokens of overlap. MiniLM produces normalized embeddings, and a small local NumPy index performs cosine similarity search. We retrieve four passages for a question. A separate set of twelve development examples determines the retrieval threshold before final evaluation. No cloud vector database or paid model service is required.""",
        "evaluation": f"""The evaluation dashboard shows measured results rather than illustrative scores. On the public held-out test set, the fine-tuned classifier achieved macro F one of {metrics['fine_tuned']['macro_f1']:.3f}, compared with {metrics['baseline']['macro_f1']:.3f} for the T F I D F logistic-regression baseline. The training curve and confusion matrix come from the completed local training run. Those public scores require caution. The source is hybrid synthetic data, and a sampled near-duplicate audit found related templates across the training and test partitions. On eighty-one independently authored workplace tickets, intent macro F one was {workplace['intent']['macro_f1']:.3f}. This lower result is an important finding, not a result to hide. Sentiment macro F one on those tickets was {workplace['sentiment']['macro_f1']:.3f}. Policy retrieval recall at four was {rag['recall_at_4']:.2f} over twenty answerable questions. We also retain generated answers, closed-book comparisons and failure cases. Automatic keyword checks are only diagnostics. Human semantic review, summary review and paired workflow timing must be completed separately before making stronger claims.""",
        "reflection": """For situated learning, the workplace analogue is an e-commerce support team handling repetitive policy questions and customer complaints. The prototype can assist with triage, reading and policy lookup, but a benefit such as reduced handling time must be measured with real users. It currently lacks authenticated order access, customer verification, production monitoring and multi-user load testing. Privacy, retention, organizational rules, security and infrastructure costs would need review before deployment. Our priority improvement is validation and calibration using representative, expert-reviewed workplace tickets. The gap between public and independent-ticket performance directly supports that choice. Local inference avoids paid usage fees, but hardware and maintenance still have a cost.""",
        "finish": """The project includes reproducible training, indexing and evaluation commands, focused tests, recorded evidence and Word and PDF report generation for all four group members. Shared project sections are followed by an individually written Situated Learning section. Group identifiers, actual contribution percentages and personal reflections must be supplied by the members. Draft reports clearly mark missing inputs. Both the report and this demonstration are required for submission. After uploading the final recording to Google Drive, the group should verify evaluator access and insert the video link in the report. Thank you for watching RetailAssist."""
    }
    ffmpeg = get_ffmpeg_exe()
    durations = {}
    for key, text in scripts.items():
        (directory / f"{key}.txt").write_text(text)
        subprocess.run(["/usr/bin/say", "-v", "Samantha", "-r", "155", "-f", str(directory/f"{key}.txt"), "-o", str(directory/f"{key}.aiff")], check=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(directory/f"{key}.aiff"), "-ar", "48000", "-ac", "1", str(directory/f"{key}.wav")], check=True)
        with wave.open(str(directory/f"{key}.wav")) as audio: durations[key] = audio.getnframes()/audio.getframerate()
    (directory / "narration.md").write_text("# Synthetic offline narration over actual browser interactions\n\n" + "\n\n".join(f"## {k}\n\n{v}" for k,v in scripts.items()))
    offsets = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width":1440,"height":1000}, record_video_dir=str(directory/"raw"), record_video_size={"width":1440,"height":1000})
        start = time.perf_counter()
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        def narrate(key):
            assert page.locator('[data-testid="stException"]').count() == 0, "Application exception visible"
            offsets[key] = time.perf_counter()-start
            print(f"Recording chapter: {key} ({durations[key]:.1f}s)", flush=True)
            page.wait_for_timeout(durations[key]*1000 + 500)
        def ask(message):
            box = page.get_by_role("textbox", name="Ask about refunds, shipping, accounts, or another store policy…")
            before = page.locator('[data-testid="stChatMessage"]').count()
            box.fill(message); box.press("Enter")
            page.locator('[data-testid="stChatMessage"]').nth(before+1).wait_for(timeout=240000)
        page.goto("http://127.0.0.1:8501", wait_until="networkidle")
        page.get_by_text("Resolve with context.", exact=True).wait_for()
        narrate("intro")
        page.get_by_role("combobox", name="Example ticket").click()
        page.get_by_role("option", name="Refund delay", exact=True).click()
        page.get_by_role("button", name="Analyze ticket", exact=True).click()
        page.get_by_text("Ticket summary", exact=True).wait_for(timeout=240000)
        page.get_by_text("Ticket summary", exact=True).scroll_into_view_if_needed()
        narrate("analysis")
        page.get_by_role("button", name="Draft reply from policies", exact=True).click()
        page.get_by_role("textbox", name="Reply draft — edit before use").wait_for(timeout=240000)
        draft_box = page.get_by_role("textbox", name="Reply draft — edit before use")
        draft_box.fill("Thank you for reaching out. " + draft_box.input_value())
        draft_box.press("Tab")
        page.locator('details').filter(has_text="DemoMart").first.locator("summary").click()
        narrate("draft")
        page.screenshot(path=str(screenshots/"01-workspace.png"), full_page=True)
        page.get_by_text("Policy chatbot", exact=True).click()
        ask("How long after delivery may an unused product be returned?")
        ask("And how long does the refund take after inspection?")
        narrate("chat")
        page.screenshot(path=str(screenshots/"02-chatbot.png"), full_page=True)
        ask("Does DemoMart provide a five-year refrigerator warranty?")
        narrate("unsupported")
        page.get_by_text("Knowledge library", exact=True).click()
        page.get_by_text("A shared source of clarity.", exact=True).wait_for()
        page.get_by_role("button", name="Open PDF").first.wait_for()
        page.screenshot(path=str(screenshots/"03-knowledge.png"), full_page=True)
        narrate("knowledge")
        page.get_by_text("Evaluation", exact=True).click()
        page.get_by_text("Evidence over assumptions.", exact=True).wait_for()
        page.get_by_text("Public intent macro-F1", exact=True).wait_for()
        page.screenshot(path=str(screenshots/"04-evaluation.png"), full_page=True)
        narrate("evaluation")
        page.mouse.wheel(0, 650)
        narrate("reflection")
        page.get_by_text("Ticket workspace", exact=True).click()
        narrate("finish")
        raw = page.video
        context.close()
        raw_path = raw.path()
        browser.close()
    (screenshots/"browser-check.json").write_text(json.dumps({"page_errors":errors,"passed":not errors},indent=2))
    if errors: raise AssertionError(errors)
    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path)]
    for key in scripts: command += ["-i", str(directory/f"{key}.wav")]
    filters = [f"[{i}:a]adelay={int(offsets[key]*1000)}:all=1[a{i}]" for i,key in enumerate(scripts, 1)]
    filters += ["".join(f"[a{i}]" for i in range(1,len(scripts)+1)) + f"amix=inputs={len(scripts)}:normalize=0[aout]"]
    target = directory / "RetailAssist-demonstration.mp4"
    command += ["-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]", "-vf", "fps=15,format=yuv420p",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(target)]
    subprocess.run(command, check=True)
    metadata = {"video": str(target), "synthetic_narration": True, "voice":"macOS Samantha", "actual_browser_recording":True,
                "chapter_offsets_seconds": offsets, "narration_durations_seconds":durations}
    (directory/"recording.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__ == "__main__": record()
