"""Real-model conversation and policy-qualification regression checks."""
import json
from retailassist.service import Copilot
from retailassist.schemas import ChatRequest
from retailassist.config import SETTINGS
from retailassist.training import dump


def main():
    service = Copilot()
    history, rows = [], []
    for question, required in [
        ("How long does a refund take?", ["5", "7", "business", "receiv", "inspect"]),
        ("And when does that start?", ["receiv", "inspect"]),
        ("And what if it has not arrived by then?", ["support", "7", "business"]),
        ("How do I reset my password?", ["forgot password", "email"]),
    ]:
        response = service.chat(ChatRequest(message=question, history=history[-6:]))
        answer = response.answer.casefold()
        expected_document = "05-accounts.pdf" if "password" in question else "03-refunds.pdf"
        cited_documents = {s.document for s in response.sources if s.id in response.cited_source_ids}
        passed = not response.abstained and all(word in answer for word in required) and expected_document in cited_documents
        rows.append({"question": question, "history": list(history[-6:]), "response": response.model_dump(),
                     "required_fragments": required, "expected_document": expected_document, "passed": passed})
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": response.answer}])
        print(json.dumps({"question": question, "passed": passed, "answer": response.answer}), flush=True)
    for question, required in [
        ("How do I request account deletion and how long does processing take?", ["target", "verif", "30", "calendar"]),
        ("Can I leave a message outside human support hours, and when should I expect a response?", ["message", "business day"]),
    ]:
        response = service.chat(ChatRequest(message=question))
        passed = not response.abstained and all(word in response.answer.casefold() for word in required)
        rows.append({"question": question, "response": response.model_dump(), "required_fragments": required, "passed": passed})
        print(json.dumps({"question": question, "passed": passed, "answer": response.answer}), flush=True)
    result = {"suite": "Known conversation and qualification regressions; automatic checks, not human ratings",
              "passed": all(row["passed"] for row in rows), "cases": rows}
    dump(SETTINGS.evaluation / "conversation_checks.json", result)
    if not result["passed"]: raise SystemExit("A conversation regression failed; inspect saved evidence.")


if __name__ == "__main__": main()
