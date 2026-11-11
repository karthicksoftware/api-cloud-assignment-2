# Worked industry analogue: DemoMart customer support

This is a shared reference case using the permitted fictional industry analogue. It is not a statement about any member's employment or an individually authored submission. Each member can use the actual evidence below to develop their own analysis and reflection.

## Anchor

DemoMart represents an Indian e-commerce support operation handling order, delivery, refund, payment and account questions. In the modeled workflow, an agent reads the incoming message, decides its topic, condenses the issue and looks through policy documents before writing a response. The limitations are repeated reading, inconsistent topic assignment and the risk of quoting the wrong policy. RetailAssist is intended to assist these steps while retaining agent approval. Faster handling is an expected benefit; the prototype has not yet established an actual human time saving.

## Apply

The training source is Bitext's public customer-support dataset. Store procedures are represented by eight synthetic PDF policies. The public instruction field is used to fine-tune a 27-class DistilBERT model, and the running API loads that saved checkpoint. A separate sentiment model identifies tone; Qwen summarizes messages and drafts answers using MiniLM-retrieved policy evidence.

In the recorded refund scenario, the customer says that the returned item was received and inspected nine business days ago, but no refund has arrived. The workspace identifies a refund-tracking intent and negative sentiment, then summarizes the issue. The relevant policy states that an approved refund takes 5–7 business days after receipt and inspection, and recommends contacting support when the approved refund has not arrived after seven business days. The draft includes retrieved source references. The backend explicitly discloses that live customer/order/payment records are unavailable, so the agent must verify the real status separately.

Two further recorded policy examples explain standard delivery in 3–5 business days after dispatch and password recovery through the Forgot password email flow. These are procedural answers, not claims that the assistant tracked a parcel or reset a password. The chatbot also handles follow-ups and declines an unsupported warranty question. Application screenshots, prediction JSON files and the video provide the representative inputs and outputs.

## Analyse

The fine-tuned classifier's public-test macro-F1 is approximately 0.998, compared with 0.988 for the TF-IDF baseline. On 81 separately authored synthetic workplace tickets, intent macro-F1 falls to approximately 0.688. The difference is central to assessing workplace relevance: high performance on synthetic source templates does not demonstrate reliable handling of different customer phrasing. The near-template audit found 138 close training matches in 500 sampled public test messages, reinforcing that caution. Some wrong intent predictions have high confidence, so confidence cannot replace agent judgment.

Sentiment macro-F1 on the independent tickets is approximately 0.975, but those examples deliberately contain relatively clear sentiment cues and have not been labeled by a second reviewer. Retrieval finds the expected document in the top four passages for all 20 answerable policy questions. All five unsupported questions are declined in the final run. Automatic answer checks and valid source IDs still do not prove semantic correctness; the human answer and summary reviews remain outstanding. Five paired human timing tasks must be completed before claiming efficiency gains. Production scalability and concurrent-agent workloads have not been tested.

## Reflect

The prototype uses synthetic data and local inference, but a workplace version would need authenticated access, restrictions on order data, retention/deletion controls, policy ownership, monitoring and organizational/legal review. It lacks a verified CRM or payment integration. Local execution avoids paid model usage fees while still requiring hardware and maintenance. An observed adversarial refund prompt originally elicited a false claim of action; the final backend rejects recognized action claims, but this text guard is not a complete security defense.

The single priority improvement is validation and calibration on representative, expert-reviewed workplace tickets and policy answers. The large gap between public and independent-ticket intent performance provides a concrete reason to prioritize that work before expanding integrations. A representative anonymized dataset, reviewed labels and a stable held-out evaluation would reveal the actual error distribution and support informed decisions about model adaptation and human escalation.

Use the final generated metrics rather than these rounded values if the project is retrained. This case does not supply personal experiences, contribution claims or human review outcomes.
