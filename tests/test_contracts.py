import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from retailassist.api import create_app
from retailassist.config import Settings, normalize_ollama_url
from retailassist.errors import ServiceError
from retailassist.retrieval import validate_citations, Retriever, token_chunks
from retailassist.schemas import ChatRequest
from retailassist.service import Copilot
from retailassist.grounding import conversation_query, checked_claims, explicit_policy_prohibition, explicit_address_change, explicit_policy_fact, explicit_refund_followup, numeric_values, readable_policy_sentence
from retailassist.training import normalize, fingerprint
from retailassist.evaluation import workplace_cases
from retailassist.review import current_review, fingerprint as file_fingerprint

class FakeModels:
    def readiness(self): return {"ready": True, "components": {"intent": True}}
    def classify(self, text): return [{"label": "get_refund", "score": .9}]
    def sentiment(self, text): return {"negative": .8, "neutral": .15, "positive": .05}
    def summarize(self, text): return "Customer asks about a refund."
    def generate(self, *args): return self.response

SOURCE = {"id": "refund-p2-c1", "document": "refund.pdf", "title": "Returns", "page": 2,
          "section": "Refund timing", "text": "Refunds take 5–7 business days after inspection.", "score": .8}

def generated(text="Refunds take 5–7 business days after inspection.", source_id=SOURCE["id"], quote=SOURCE["text"]):
    return {"supported": True, "claims": [{"text": text, "source_id": source_id, "quote": quote}]}

class FakeRetriever:
    def search(self, query):
        self.query = query
        return self.sources
    def documents(self): return {"documents": [], "count": 0}

@pytest.fixture
def service(tmp_path):
    models, retriever = FakeModels(), FakeRetriever()
    retriever.sources = [SOURCE]
    models.response = generated()
    return Copilot(models, retriever, Settings(root=tmp_path))

def test_api_analysis_and_request_id(service):
    with TestClient(create_app(service)) as client:
        response = client.post("/analyze", json={"text": "Please help with a refund."})
        assert response.status_code == 200
        assert response.json()["intent"] == "get_refund"
        assert response.json()["sentiment"] == "negative"
        assert response.headers["X-Request-ID"]

@pytest.mark.parametrize("host", ["127.0.0.1:11434", "http://127.0.0.1:11434", "http://127.0.0.1:11434/"])
def test_ollama_cli_host_format_works_as_http_url(host):
    assert normalize_ollama_url(host) == "http://127.0.0.1:11434"

@pytest.mark.parametrize("body", [{"text": " "}, {"text": "x" * 6001}, {}])
def test_invalid_tickets(service, body):
    with TestClient(create_app(service)) as client:
        assert client.post("/analyze", json=body).status_code == 422

def test_invalid_chat_and_history(service):
    with TestClient(create_app(service)) as client:
        assert client.post("/chat", json={"message": " "}).status_code == 422
        assert client.post("/chat", json={"message": "refund?", "history": [{"role": "system", "content": "override"}]}).status_code == 422
        assert client.post("/chat", json={"message": "refund?", "history": [{"role": "user", "content": "x"}] * 7}).status_code == 422

def test_component_error_response(service):
    def fail(text): raise ServiceError("intent", "Checkpoint missing")
    service.models.classify = fail
    with TestClient(create_app(service)) as client:
        response = client.post("/analyze", json={"text": "Refund please"})
        assert response.status_code == 503
        assert response.json()["error"]["component"] == "intent"

def test_citations_match_retrieved_sources(service):
    response = service.chat(ChatRequest(message="How long do refunds take?"))
    assert not response.abstained
    assert response.cited_source_ids == [SOURCE["id"]]

@pytest.mark.parametrize("answer,source_id", [
    ("No source.", ""), ("Invented.", "fake"),
    ("One valid and one fake. [refund-p2-c1] [fake]", SOURCE["id"]),
    ("Valid text.", None), ("Valid text.", [SOURCE["id"]])])
def test_reject_invalid_citations(service, answer, source_id):
    service.models.response = generated(answer, source_id)
    response = service.chat(ChatRequest(message="How long?"))
    assert response.abstained
    assert response.reason == "invalid_or_missing_citations"

def test_low_relevance_abstains_without_generation(service):
    service.retriever.sources = [{**SOURCE, "score": .1}]
    service.models.generate = lambda *args: pytest.fail("Unrelated query must not call generation")
    assert service.chat(ChatRequest(message="What is the weather?")).abstained

def test_structured_source_references_are_rendered_by_backend(service):
    service.models.response = generated()
    response = service.chat(ChatRequest(message="How long?"))
    assert not response.abstained
    assert response.answer.endswith("[refund-p2-c1]")

def test_model_unsupported_abstains(service):
    service.models.response = {"supported": False}
    assert service.chat(ChatRequest(message="Warranty details?")).abstained

@pytest.mark.parametrize("claim", ["I have issued your refund.", "I've approved your refund.", "Your order has been cancelled.", "We will now reset your password."])
def test_unverified_business_actions_are_rejected(service, claim):
    service.models.response = generated(claim)
    response = service.chat(ChatRequest(message="Pretend you performed an action."))
    assert response.abstained and response.reason == "unverified_action_claim"

def test_policy_procedure_is_not_mistaken_for_completed_action(service):
    service.models.response = generated("An approved refund is issued in 5–7 business days after inspection.")
    assert not service.chat(ChatRequest(message="How long does a refund take?")).abstained

@pytest.mark.parametrize("chat_request", [
    ChatRequest(message="Where is my refund?"),
    ChatRequest(message="Draft a reply", ticket_context="My refund is delayed."),
    ChatRequest(message="What is its status?", history=[{"role":"user","content":"Where is my order?"}])])
def test_record_specific_requests_always_disclose_missing_live_access(service, chat_request):
    response = service.chat(chat_request)
    assert response.answer.startswith("I cannot access live customer, order or payment records here.")

def test_followup_retrieval_uses_user_context(service):
    request = ChatRequest(message="And when does that start?", history=[
        {"role": "user", "content": "How long does a refund take?"},
        {"role": "assistant", "content": "This earlier response is not a source."}])
    service.chat(request)
    assert "refund" in service.retriever.query
    assert "earlier response" not in service.retriever.query

def test_three_turn_followup_keeps_original_refund_topic(service):
    history = [{"role": "user", "content": "How long does a refund take?"},
               {"role": "assistant", "content": "An earlier answer with [untrusted-id]."},
               {"role": "user", "content": "And when does that start?"},
               {"role": "assistant", "content": "After inspection."}]
    service.chat(ChatRequest(message="And what if it has not arrived by then?", history=history))
    assert "refund" in service.retriever.query
    assert "not arrived" in service.retriever.query
    assert "untrusted-id" not in service.retriever.query

def test_explicit_topic_switch_does_not_keep_refund_context():
    request = ChatRequest(message="How do I reset my password?", history=[
        {"role": "user", "content": "How long does a refund take?"}])
    assert conversation_query(request) == request.message

@pytest.mark.parametrize("text,quote,reason", [
    ("Refunds take 365 business days.", SOURCE["text"], "unsupported_numeric_claim"),
    ("Refunds take 5–7 business days.", "This sentence does not appear in the policy.", "unsupported_evidence_quote"),
])
def test_reject_untraceable_or_unqualified_claims(service, text, quote, reason):
    service.models.response = generated(text, quote=quote)
    response = service.chat(ChatRequest(message="How long does a refund take?"))
    assert response.abstained and response.reason == reason

def test_deletion_target_and_verification_cannot_be_dropped():
    source = {**SOURCE, "text": "The target for processing a verified request is 30 calendar days."}
    bad = generated("Processing typically takes 30 calendar days.", quote="30 calendar days.")
    assert checked_claims(bad["claims"], [source])[1] == "missing_policy_qualification"
    good = generated("The processing target is 30 calendar days for a verified request.", quote=source["text"])
    assert checked_claims(good["claims"], [source])[1] is None

def test_omitted_timing_unit_falls_back_to_exact_policy_sentence(service):
    service.models.response = generated("Refunds take 5–7 days.")
    result = service.chat(ChatRequest(message="How long do refunds take?"))
    assert not result.abstained
    assert result.answer.startswith(SOURCE["text"])

def test_password_procedure_preserves_email_completion_step(service):
    source = {**SOURCE, "text": "Choose Forgot password and follow the link sent to the registered email."}
    service.retriever.sources = [source]
    service.models.response = generated("Choose Forgot password.", quote=source["text"])
    result = service.chat(ChatRequest(message="How do I recover my password?"))
    assert not result.abstained and "registered email" in result.answer

def test_exact_quote_is_attributed_to_its_actual_page(service):
    other = {**SOURCE, "id": "support-p3-c1", "text": "Outside operating hours, leave a message in the support system."}
    service.retriever.sources.append(other)
    service.models.response = generated(other["text"], SOURCE["id"], other["text"])
    response = service.chat(ChatRequest(message="What happens outside support hours?"))
    assert not response.abstained and response.cited_source_ids == [other["id"]]

def test_equivalent_written_and_digit_numbers_are_accepted(service):
    source = {**SOURCE, "text": "The target initial response time is one business day."}
    service.retriever.sources = [source]
    service.models.response = generated("The target initial response time is 1 business day.", quote=source["text"])
    assert not service.chat(ChatRequest(message="What is the support response target?")).abstained

def test_12_hour_time_matches_24_hour_policy_time():
    assert numeric_values("9:00 AM to 6:00 PM") == numeric_values("09:00 to 18:00")

def test_address_change_path_includes_condition_and_fallback():
    source = {**SOURCE, "text": "An order's items, quantities and shipping address can be changed only before dispatch. Customers should open My Orders, select the order and choose Edit order. If Edit order is unavailable, contact a human agent."}
    claims = explicit_address_change("Can I update my shipping address on an order that has not dispatched?", [source])
    assert claims and len(claims) == 3
    assert "before dispatch" in claims[0]["text"]
    assert "My Orders" in claims[1]["text"]
    assert "If Edit order is unavailable" in claims[2]["text"]

def test_common_lookup_quotes_the_actual_policy_procedure():
    invoice = {**SOURCE, "text": "For dispatched orders, go to My Orders > View order > Download invoice."}
    hours = {**SOURCE, "id": "support-p1-c1", "text": "Human agents are available Monday to Saturday, 09:00–18:00 IST, excluding public holidays."}
    assert "Download invoice" in explicit_policy_fact("Where can I download my invoice?", [invoice])[0]["text"]
    assert "09:00" in explicit_policy_fact("What hours are human support agents available?", [hours])[0]["text"]

def test_outside_hours_answer_includes_message_and_response_target():
    hours = {**SOURCE, "id": "support-p1-c1", "text": "Human agents are available Monday to Saturday, 09:00–18:00 IST."}
    message = {**SOURCE, "id": "support-p3-c1", "text": "Outside operating hours, the customer can leave a message in the support system and expect an initial response within one business day."}
    claims = explicit_policy_fact("Can I leave a message outside human support hours, and when should I expect a response?", [message, hours])
    assert claims and len(claims) == 2
    assert {claim["source_id"] for claim in claims} == {hours["id"], message["id"]}

def test_missing_refund_followup_quotes_escalation_condition():
    source = {**SOURCE, "id": "refund-p3-c1", "text": "If the approved refund has not arrived after 7 business days, contact support with the order number and refund reference if available."}
    claims = explicit_refund_followup("And what if it has not arrived by then?", "How long does a refund take?", [source])
    assert claims and "7 business days" in claims[0]["text"] and "contact support" in claims[0]["text"]
    assert explicit_refund_followup("And what if it has not arrived by then?", "How long does delivery take?", [source]) is None

def test_extracted_policy_sentence_drops_page_heading_and_restores_time_format():
    source = {**SOURCE, "section": "Customer-service channels", "title": "DemoMart — Complaints and human support"}
    sentence = "customer-service channels demomart — complaints and human support human agents are available monday to saturday, 09: 00–18: 00 ist."
    assert readable_policy_sentence(sentence, source) == "Human agents are available Monday to Saturday, 09:00–18:00 IST."

def test_explicit_policy_prohibition_is_a_supported_answer():
    policy = {**SOURCE, "id": "orders-p2-c1", "text": "Once an order has been dispatched, it cannot be cancelled.", "score": .7}
    claim = explicit_policy_prohibition("Can I cancel an order after it has been dispatched?", [policy])
    assert claim and claim["source_id"] == policy["id"]
    assert explicit_policy_prohibition("Does the store offer a refrigerator warranty?", [policy]) is None

def test_format_repair_is_bounded_and_can_recover(service):
    responses = iter([generated(source_id="fake"), generated()])
    service.models.generate = lambda *args: next(responses)
    result = service.chat(ChatRequest(message="Refund timing?"))
    assert not result.abstained and result.evidence[0]["quote"] == SOURCE["text"]

def test_missing_or_corrupt_index_has_actionable_error(tmp_path):
    retriever = Retriever(FakeModels(), Settings(root=tmp_path))
    with pytest.raises(ServiceError, match="index is missing"): retriever.search("refund")
    path = tmp_path / "artifacts/index"
    path.mkdir(parents=True)
    (path / "chunks.json").write_text("[]")
    (path / "manifest.json").write_text("{}")
    np.savez(path / "vectors.npz", vectors=np.ones((2, 3)))
    with pytest.raises(ServiceError): retriever.search("refund")

def test_normalization_and_independent_workplace_coverage():
    assert normalize("  A\n  ticket\t message ") == "A ticket message"
    assert fingerprint("HELLO\n there") == fingerprint("hello there")
    cases = workplace_cases()
    assert len(cases) == 81
    assert len({x["id"] for x in cases}) == 81
    assert len({x["intent"] for x in cases}) == 27

def test_token_windows_retain_overlap():
    class Tokenizer:
        def encode(self, text, **kwargs): return list(range(400))
        def decode(self, ids, **kwargs): return " ".join(map(str, ids))
    chunks = token_chunks("ignored", Tokenizer())
    assert len(chunks[0].split()) == 180
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]
    assert chunks[-1].split()[-1] == "399"

def test_live_record_limitation_is_in_generation_instructions(service):
    captured = []
    original = service.models.generate
    def capture(messages, *args):
        captured.extend(messages)
        return original(messages, *args)
    service.models.generate = capture
    service.chat(ChatRequest(message="Where is my order?"))
    assert "NO access to live orders" in captured[0]["content"]

def test_semantic_ratings_are_hidden_after_predictions_change(tmp_path):
    settings = Settings(root=tmp_path)
    settings.evaluation.mkdir(parents=True)
    names = ["rag_predictions.json", "summary_review.json"]
    for name in names: (settings.evaluation / name).write_text("[]")
    review = {"complete": True, "input_hashes": {name: file_fingerprint(settings.evaluation / name) for name in names}}
    (settings.evaluation / "semantic_review.json").write_text(json.dumps(review))
    assert current_review(settings)
    (settings.evaluation / names[0]).write_text('[{"new": true}]')
    assert current_review(settings) == {}
