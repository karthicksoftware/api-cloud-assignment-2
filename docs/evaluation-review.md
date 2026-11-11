# Human review and timing protocol

The application exports measured predictions; it does not invent human ratings.

Start with `submission/evidence-review-pack.md`, which places each answer beside its actual source passages. `semantic_review.json` contains a separate automated Qwen review with rationales and input hashes. Its ratings must not be copied into the human-review fields without a member actually inspecting the evidence. The report identifies those ratings as automated and explains the shared-model limitation.

1. Open `artifacts/evaluation/rag_predictions.json`. For each of the 20 answerable questions, read the question, answer, cited passage and complete source PDF. Record correctness (0/1), citation support (0/1), reviewer and a short rationale in a separate `rag_human_review.json` file. Unsupported/adversarial cases should also be reviewed for appropriate refusal or policy adherence. Only after all cases are reviewed, add `human_correctness_review: "complete"` and actual reviewed scores to `rag_metrics.json`. Keep the review file as supporting evidence.
2. Open `summary_review.json`. For each of its 10 cases, compare every summary statement with the original ticket. Fill `facts_retained` (0/1), `invented_facts` (0/1) and `reviewer`. Mark status `complete` only after all cases are reviewed. Mixed or omitted facts should have notes.
3. Time five questions with a real participant. Use questions about return eligibility, refund timing, shipping cost, invoice download and support hours. Record manual lookup time from opening the policy library to a correct answer, then assistant time including source verification. Counterbalance order across questions to reduce practice effects. Record participant, question, condition order, both times, correctness and notes in `efficiency_timings.csv`. Do not use model-only latency as a substitute for human workflow time.
4. Report medians, individual times and limitations. Five synthetic tasks are a pilot, not proof of an organization-wide productivity increase.

The automatic fact-keyword pass rate is a surface diagnostic. Paraphrases can fail it and incorrect answers can pass it. Source-ID validation checks identity, not the support of every claim. These limits must remain explicit in the report.
