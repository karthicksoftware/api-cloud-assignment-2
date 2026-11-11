import json
import logging
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .config import SETTINGS
from .errors import ServiceError
from .schemas import TicketRequest, ChatRequest, AnalysisResponse, ChatResponse
from .service import Copilot
from .review import current_review

def create_app(copilot=None):
    app = FastAPI(title="RetailAssist API", version="1.0.0", description="Local customer-service NLP and policy RAG")
    service = copilot or Copilot()
    app.state.copilot = service

    @app.exception_handler(ServiceError)
    async def unavailable(request, exc):
        return JSONResponse(status_code=exc.status, content={"error": {"component": exc.component, "message": exc.message}})

    @app.middleware("http")
    async def timing(request: Request, call_next):
        request_id, start = uuid.uuid4().hex[:12], time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logging.getLogger("retailassist.api").info("request=%s path=%s status=%s elapsed_ms=%.1f",
            request_id, request.url.path, response.status_code, (time.perf_counter()-start)*1000)
        return response

    @app.get("/health")
    def health(): return service.models.readiness()

    @app.post("/analyze", response_model=AnalysisResponse)
    def analyze(body: TicketRequest): return service.analyze(body.text)

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest): return service.chat(body)

    @app.get("/knowledge")
    def knowledge(): return service.retriever.documents()

    @app.get("/evaluation")
    def evaluation():
        files = ["training_metrics.json", "workplace_metrics.json", "rag_metrics.json", "summary_review.json"]
        result = {}
        for filename in files:
            path = SETTINGS.evaluation / filename
            if path.exists(): result[filename.removesuffix(".json")] = json.loads(path.read_text())
        review = current_review()
        if review: result["semantic_review"] = review
        failures = {}
        path = SETTINGS.evaluation / "workplace_predictions.json"
        if path.exists():
            rows = json.loads(path.read_text())
            failures["intent"] = [{k: row[k] for k in ["text", "intent", "predicted_intent", "confidence"]}
                                  for row in rows if row["intent"] != row["predicted_intent"]][:10]
        path = SETTINGS.evaluation / "rag_predictions.json"
        if path.exists():
            rows = json.loads(path.read_text())
            failures["rag"] = [{"question": row["question"], "response": row["response"]}
                               for row in rows if row.get("kind") == "answerable" and row.get("response", {}).get("abstained")][:5]
        return {"available": bool(result), "results": result,
                "failure_cases": failures,
                "note": "Only recorded measurements are shown. Human review may still be pending."}

    return app

app = create_app()
