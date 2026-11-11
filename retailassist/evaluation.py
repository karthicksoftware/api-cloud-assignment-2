import json
import time
import hashlib
from datetime import datetime, timezone
import numpy as np
from .config import SETTINGS, INTENTS
from .training import dump, score_labels
from .models import LocalModels
from .retrieval import Retriever
from .service import Copilot
from .schemas import ChatRequest
from .errors import ServiceError
from .grounding import checked_claims

def workplace_cases(settings=SETTINGS):
    values = json.loads((settings.root / "data/workplace_cases.json").read_text())["cases"]
    assert sorted(values) == INTENTS
    return [{"id": f"{intent}-{i}", "text": text, "intent": intent,
             "sentiment": ["negative", "neutral", "positive"][i]}
            for intent, messages in values.items() for i, text in enumerate(messages)]

def calibrate(settings=SETTINGS):
    from sklearn.metrics import f1_score
    models = LocalModels(settings)
    retriever = Retriever(models, settings)
    cases = json.loads((settings.root / "data/development_questions.json").read_text())
    scores = [retriever.search(x["question"], k=1)[0]["score"] for x in cases]
    trials = [{"threshold": float(threshold), "f1": float(f1_score(
        [x["supported"] for x in cases], [score >= threshold for score in scores]))}
        for threshold in [round(.20 + .025*i, 3) for i in range(21)]]
    best = max(trials, key=lambda x: (x["f1"], x["threshold"]))
    result = {"retrieval": best["threshold"], "development_f1": best["f1"], "development_rows": len(cases),
              "method": "Threshold maximizing supported/unsupported F1 on development only; ties prefer higher threshold.",
              "trials": trials, "examples": [{**c, "score": s} for c, s in zip(cases, scores)]}
    dump(settings.artifacts / "calibration.json", result)
    return result

def evaluate(settings=SETTINGS, skip_generation=False):
    models = LocalModels(settings)
    service = Copilot(models=models, settings=settings)
    cases = workplace_cases(settings)
    predictions = []
    for case in cases:
        intents = models.classify(case["text"])
        sentiments = models.sentiment(case["text"])
        predictions.append({**case, "predicted_intent": intents[0]["label"], "confidence": intents[0]["score"],
                            "predicted_sentiment": max(sentiments, key=sentiments.get), "sentiment_scores": sentiments})
    metrics = {"count": len(cases), "intent": score_labels([x["intent"] for x in predictions],
        [x["predicted_intent"] for x in predictions], INTENTS), "sentiment": score_labels(
        [x["sentiment"] for x in predictions], [x["predicted_sentiment"] for x in predictions], ["negative", "neutral", "positive"]),
        "note": "Synthetic, independently authored tickets; labels have not been reviewed by a second human annotator."}
    dump(settings.evaluation / "workplace_predictions.json", predictions)
    dump(settings.evaluation / "workplace_metrics.json", metrics)
    rag_cases = json.loads((settings.root / "data/rag_cases.json").read_text())
    outputs = []
    for case in rag_cases:
        sources = service.retriever.search(case["question"])
        row = {**case, "retrieved_documents": [s["document"] for s in sources],
               "retrieval_hit": case.get("document") in [s["document"] for s in sources] if case.get("document") else None}
        if not skip_generation:
            print(f"Evaluating {case['id']}", flush=True)
            try:
                answer = service.chat(ChatRequest(message=case["question"]))
                row["response"] = answer.model_dump()
            except ServiceError as exc:
                row["response"] = {"answer": exc.message, "sources": sources, "cited_source_ids": [],
                    "abstained": True, "reason": "service_error", "latency_ms": 0}
                row["error"] = {"component": exc.component, "message": exc.message}
            normalized = row["response"]["answer"].casefold()
            row["fact_keyword_check"] = all(word.casefold() in normalized for word in case.get("facts", [])) and not row["response"]["abstained"]
            row["forbidden_phrase_present"] = any(word.casefold() in normalized for word in case.get("forbidden", []))
            if case["kind"] == "answerable":
                try:
                    row["without_retrieval"] = models.generate([
                        {"role": "system", "content": "You are a support assistant for the fictional store DemoMart. Answer the question concisely. Do not invent policies you do not know."},
                        {"role": "user", "content": case["question"]}], max_tokens=250)
                except ServiceError as exc: row["without_retrieval_error"] = exc.message
        outputs.append(row)
        dump(settings.evaluation / "rag_predictions.json", outputs)
    answerable = [x for x in outputs if x["kind"] == "answerable"]
    unsupported = [x for x in outputs if x["kind"] == "unsupported"]
    adversarial = [x for x in outputs if x["kind"] == "adversarial"]
    rag_metrics = {"count": len(outputs), "answerable_count": len(answerable),
                   "recall_at_4": sum(x["retrieval_hit"] for x in answerable)/len(answerable),
                   "retrieval_threshold": settings.thresholds["retrieval"], "generation_evaluated": not skip_generation,
                   "human_correctness_review": "pending", "generation_failures": sum("error" in x for x in outputs),
                   "suite_role": "Regression evaluation: known cases were used to investigate and fix failures.",
                   "evaluated_at": datetime.now(timezone.utc).isoformat(),
                   "case_sha256": hashlib.sha256((settings.root / "data/rag_cases.json").read_bytes()).hexdigest(),
                   "code_sha256": {name: hashlib.sha256((settings.root / "retailassist" / name).read_bytes()).hexdigest()
                                   for name in ["service.py", "grounding.py", "models.py", "retrieval.py"]},
                   "recall_definition": "Expected document present among the four retrieved passages; not claim correctness.",
                   "note": "Keyword and exact-evidence checks are automatic diagnostics, not human semantic correctness ratings."}
    if not skip_generation:
        rag_metrics.update({"answerable_nonabstention_rate": sum(not x["response"]["abstained"] for x in answerable)/len(answerable),
            "fact_keyword_pass_rate": sum(x["fact_keyword_check"] for x in answerable)/len(answerable),
            "unsupported_abstention_rate": sum(x["response"]["abstained"] for x in unsupported)/len(unsupported),
            "adversarial_forbidden_phrase_rate": sum(x["forbidden_phrase_present"] for x in adversarial)/len(adversarial),
            "mean_latency_ms": float(np.mean([x["response"]["latency_ms"] for x in outputs]))})
        answered = [x["response"] for x in outputs if not x["response"]["abstained"]]
        rag_metrics["evidence_quote_validation_rate"] = (sum(
            checked_claims(x.get("evidence"), x["sources"])[1] is None for x in answered) / len(answered)) if answered else 0
        rag_metrics["evidence_quote_denominator"] = len(answered)
        summaries = []
        for index in np.linspace(0, len(cases)-1, 10, dtype=int):
            case = cases[index]
            summaries.append({"id": case["id"], "ticket": case["text"], "summary": models.summarize(case["text"]),
                              "facts_retained": None, "invented_facts": None, "reviewer": None})
        dump(settings.evaluation / "summary_review.json", {"status": "pending human review", "cases": summaries})
    dump(settings.evaluation / "rag_metrics.json", rag_metrics)
    return {"workplace_intent_macro_f1": metrics["intent"]["macro_f1"],
            "workplace_sentiment_macro_f1": metrics["sentiment"]["macro_f1"], "rag": rag_metrics}
