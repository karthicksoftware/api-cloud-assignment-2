"""Collect actual outputs for member review without inventing human sign-off."""
import json
from retailassist.config import SETTINGS
from retailassist.review import current_review


def main():
    evaluation = SETTINGS.evaluation
    cases = json.loads((evaluation / "rag_predictions.json").read_text())
    summaries = json.loads((evaluation / "summary_review.json").read_text())["cases"]
    review = current_review()
    ratings = {row["id"]: row for row in review.get("answers", [])}
    text = ["# Group 28 — Evidence review pack", "",
            "This pack contains actual outputs and their supporting passages. A member should check the answer against the question and cited policy, then record their own judgement. No human approval is pre-filled.", "",
            "The automated review is a diagnostic using Qwen2.5, not a substitute for your judgement. The submission report and video are the mandatory deliverables; this pack is supporting QA.", ""]
    for case in cases:
        response = case["response"]
        text += [f"## {case['id']} — {case['kind']}", "", "**Question:** " + case["question"], "",
                 "**Actual answer:** " + response["answer"], ""]
        for source in response["sources"]:
            if source["id"] in response["cited_source_ids"]:
                text += [f"**{source['document']}, page {source['page']} — {source['id']}**", "", source["text"], ""]
        if case["kind"] == "answerable" and (response["abstained"] or not case["fact_keyword_check"]):
            text += ["**Review priority:** The answer was declined or the automatic keyword check did not pass. A correct paraphrase can still fail the keyword check.", ""]
        if case["id"] in ratings and ratings[case["id"]].get("correct") is not None:
            rating = ratings[case["id"]]
            text += [f"Automated assessment: correct={rating.get('correct')}; fully supported={rating.get('fully_supported')}. {rating.get('rationale', '')}", ""]
        text += ["Member judgement: correctness ___ / citation support ___ / reviewer ___ / notes ___", ""]
    for case in summaries:
        text += ["## Summary — " + case["id"], "", "**Original ticket:** " + case["ticket"], "",
                 "**Actual summary:** " + case["summary"], "",
                 "Member judgement: necessary facts retained ___ / invented facts ___ / reviewer ___ / notes ___", ""]
    target = SETTINGS.root / "submission/evidence-review-pack.md"
    target.write_text("\n".join(text))
    print(target)


if __name__ == "__main__": main()
