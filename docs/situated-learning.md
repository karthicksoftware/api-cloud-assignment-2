# Individual Situated Learning – Workplace Application

Each group member writes this section independently. Use a real workplace context you can discuss, or explicitly identify DemoMart as a permitted industry analogue. An analogue is not employment history. Use only anonymized or synthetic inputs and real recorded prototype outputs.

See [the worked industry analogue](workplace-example.md) for a common reference case based on the actual recorded project results.

## Editing the report

Karthick's four sections are filled in `submission/group.json` and exported in his `28.docx` and `28.pdf`. A readable copy is in `submission/karthick-situated-learning.md`.

For a one-off revision, open the Word file, edit Section IV and use Save As or Export to PDF. For reproducible revisions, edit the four fields in your member's `situated_learning` entry in `submission/group.json`, then run:

```sh
.venv/bin/retailassist report --metadata submission/group.json
```

Regeneration rebuilds the Word and PDF files from that JSON, so update the JSON too if you want to preserve edits made directly in Word.

## Anchor — 1 mark

Describe the support operation, intended agents, channels and current handling of tickets. Identify where manual triage, reading and policy search create delays or inconsistency. State the expected benefit as a hypothesis, not a measured outcome. Explain how the fictional scenario relates to your workplace or industry.

## Apply — 2 marks

Identify the public training data and synthetic policy documents. Walk through three representative tickets (for example delayed refund, delivery question and account access). Include the input, classifier prediction, sentiment, generated summary, retrieved policy/page and draft. Explain how the locally fine-tuned classifier contributes. Include a follow-up query and a missing-live-data example. Reference screenshots and prediction artifacts; do not invent successful outputs.

## Analyse — 1 mark

Interpret the actual held-out and independent-ticket results. Discuss classification errors, mixed sentiment, retrieval misses, citation support and hallucination risks. Distinguish synthetic evaluation from workplace validation. Use the paired timing worksheet if completed and state the small sample. Explain why the prototype lacks real order access, authentication and production concurrency testing.

## Reflect — 1 mark

Discuss privacy, access controls, record retention, organizational/legal requirements, local hardware, operating cost and integration. Choose and justify one most important improvement. Recommended starting point: validation and calibration on representative, expert-reviewed workplace data before deployment, because synthetic templates and social-media sentiment may not transfer to real tickets. Explain why that improvement takes priority in your own context.

Enter the four final paragraphs into your member entry in `submission/group.json`, then regenerate the report. Names, contributions, workplace claims and reflections must be factual and individually supplied.
