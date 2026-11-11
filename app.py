"""Streamlit client: business/model operations go through the FastAPI service."""
import json
import os
from pathlib import Path
import requests
import streamlit as st
import pandas as pd
from retailassist.config import SETTINGS
from retailassist.presentation import rag_measurements

st.set_page_config(page_title="RetailAssist · Agent console", page_icon="✳", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp {background:#f7f8f4;color:#193b34}
[data-testid="stSidebar"] {background:#e9efe8;border-right:1px solid #d5dfd4}
[data-testid="stSidebar"] h1 {font-size:1.65rem!important;letter-spacing:-.06em}
h1,h2,h3 {font-family:ui-sans-serif,system-ui,sans-serif;letter-spacing:-.04em;color:#163d34}
h1 {font-size:3.2rem!important;font-weight:650!important}
.eyebrow {font-size:.72rem;font-weight:700;letter-spacing:.18em;color:#688174;margin-bottom:12px}
.intro {font-size:1.05rem;color:#6b7c70;max-width:680px;margin-bottom:28px;line-height:1.7}
.panel {padding:20px 24px;background:white;border:1px solid #dce5d8;border-radius:12px;margin:12px 0}
.tag {display:inline-block;padding:5px 12px;border-radius:20px;background:#e8efdd;color:#45613c;font-size:.75rem;font-weight:600;margin-bottom:18px}
[data-testid="stMetric"] {padding:20px;background:white;border:1px solid #dce5d8;border-radius:12px}
[data-testid="stMetricValue"] {font-size:1.45rem}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {background:#214c3d;border:0;border-radius:8px}
textarea {font-size:1rem!important;line-height:1.7!important}
footer {visibility:hidden}
</style>""", unsafe_allow_html=True)
API = os.environ.get("RETAILASSIST_API", "http://127.0.0.1:8000")

def api(method, route, **kwargs):
    try:
        response = requests.request(method, API + route, timeout=240, **kwargs)
        if response.ok: return response.json()
        payload = response.json()
        error = payload.get("error", {}).get("message")
        st.error(error or "Please check the input and try again.")
    except requests.Timeout:
        st.error("The local model is taking longer than expected. Please retry with a shorter message.")
    except (requests.RequestException, ValueError):
        st.error("The application service is unavailable. Start the local API and try again.")
    return None

def source_cards(sources, cited_ids=None):
    for source in sources:
        cited = cited_ids is None or source["id"] in cited_ids
        label = f"{'●' if cited else '○'} {source['title']} · page {source['page']}"
        with st.expander(label):
            st.caption(source["section"])
            st.write(source["text"])
            st.caption(f"Source: {source['id']} · relevance {source['score']:.2f}")

with st.sidebar:
    st.markdown("# ✳ RetailAssist")
    st.caption("A little clarity. Better support.")
    st.divider()
    nav = st.radio("Workspace", ["Ticket workspace", "Policy chatbot", "Knowledge library", "Evaluation"], label_visibility="collapsed")
    st.divider()
    try:
        health = requests.get(API + "/health", timeout=4).json()
        st.success("All systems ready" if health["ready"] else "Setup in progress", icon=":material/check_circle:")
        with st.expander("Service status"):
            for name, ready in health["components"].items(): st.caption(f"{'✓' if ready else '○'} {name.title()}")
    except (requests.RequestException, ValueError, KeyError): st.warning("Service offline")
    st.caption("DEMOMART · FICTIONAL DEMO")
    st.caption("Local models · Agent-reviewed replies")
    if st.button("Clear current session", use_container_width=True):
        st.session_state.clear(); st.rerun()

if nav == "Ticket workspace":
    st.markdown('<div class="eyebrow">AGENT WORKSPACE / TICKET ASSISTANCE</div>', unsafe_allow_html=True)
    st.title("Resolve with context.")
    st.markdown('<div class="intro">Understand the customer, find the right policy, and shape a thoughtful reply. You make the final call.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("01 / Understand the ticket")
        samples = {"Refund delay": "My returned item was received and inspected nine business days ago, but I still have no refund. I am really frustrated. What should I do?",
                   "Delivery question": "Thanks for the great service! How much does express shipping cost and when can I expect delivery?",
                   "Account access": "I forgot my password and cannot get into my account. How can I regain access?"}
        sample = st.selectbox("Example ticket", ["Write your own", *samples])
        if st.session_state.get("last_sample") != sample:
            st.session_state["ticket_text"] = samples.get(sample, "")
            st.session_state["last_sample"] = sample
        text = st.text_area("Customer message", height=210, key="ticket_text", placeholder="Paste a customer ticket or choose an example…", max_chars=6000)
        if st.button("Analyze ticket", type="primary", disabled=not text.strip(), use_container_width=True):
            with st.spinner("Reading the ticket with local models…"):
                result = api("POST", "/analyze", json={"text": text})
            if result:
                st.session_state["analysis"] = result
                st.session_state["analyzed_text"] = text
                st.session_state.pop("draft_result", None)
        analysis = st.session_state.get("analysis")
        if analysis and st.session_state.get("analyzed_text") == text:
            a, b = st.columns(2)
            a.metric("Intent", analysis["intent"].replace("_", " ").title())
            b.metric("Customer sentiment", analysis["sentiment"].title())
            st.caption(f"Intent confidence: {analysis['confidence']:.1%} · Review predictions in context")
            st.markdown("**Ticket summary**")
            st.write(analysis["summary"])
            with st.expander("Analysis details"):
                st.dataframe(pd.DataFrame(analysis["top_intents"]), hide_index=True, use_container_width=True)
                st.json({"sentiment_scores": analysis["sentiment_scores"], "latency_ms": analysis["latency_ms"]})
    with right:
        st.subheader("02 / Prepare a grounded reply")
        st.caption("Drafts use store policies. Live customer records are unavailable. Check sources and edit before responding.")
        if st.button("Draft reply from policies", type="primary", disabled=not text.strip(), use_container_width=True):
            with st.spinner("Finding policy evidence and drafting a reply…"):
                result = api("POST", "/chat", json={"message": "Draft a concise and helpful reply to this customer's issue using the applicable policies.", "ticket_context": text})
            if result:
                st.session_state["draft_result"] = result
                st.session_state["draft_text"] = text
                st.session_state["reply_draft"] = result["answer"]
        draft = st.session_state.get("draft_result")
        if draft and st.session_state.get("draft_text") == text:
            if draft["abstained"]: st.info("A specialist should review this question.")
            st.text_area("Reply draft — edit before use", key="reply_draft", height=260)
            st.download_button("Download reply", st.session_state["reply_draft"], "reply-draft.txt", "text/plain")
            st.caption("Policy evidence")
            source_cards(draft["sources"], draft["cited_source_ids"])
        else:
            st.markdown('<div class="panel"><b>Your next reply starts here.</b><p style="color:#718170;margin-top:12px">Add a ticket, then draft a response with visible policy evidence.</p></div>', unsafe_allow_html=True)

elif nav == "Policy chatbot":
    st.markdown('<div class="eyebrow">POLICY DESK / CONVERSATION</div>', unsafe_allow_html=True)
    st.title("Ask. Check. Respond.")
    st.markdown('<div class="intro">Explore store policies through a conversation. Every supported answer includes its evidence.</div>', unsafe_allow_html=True)
    use_ticket = st.checkbox("Use the ticket from my workspace as context", value=False)
    history = st.session_state.setdefault("chat", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn.get("sources"): source_cards(turn["sources"], turn.get("cited_source_ids"))
    if message := st.chat_input("Ask about refunds, shipping, accounts, or another store policy…", max_chars=3000):
        recent = [{"role": x["role"], "content": x["content"][:6000]} for x in history[-6:]]
        with st.chat_message("user"): st.write(message)
        with st.spinner("Checking the policy library…"):
            result = api("POST", "/chat", json={"message": message, "history": recent,
                "ticket_context": st.session_state.get("ticket_text", "") if use_ticket else ""})
        if result:
            history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": result["answer"],
                           "sources": result["sources"], "cited_source_ids": result["cited_source_ids"]}])
            st.rerun()

elif nav == "Knowledge library":
    st.markdown('<div class="eyebrow">KNOWLEDGE / STORE POLICIES</div>', unsafe_allow_html=True)
    st.title("A shared source of clarity.")
    st.markdown('<div class="intro">Eight fictional policy documents support consistent answers. Open a policy to review the complete source.</div>', unsafe_allow_html=True)
    manifest = api("GET", "/knowledge")
    if manifest:
        a, b, c = st.columns(3)
        a.metric("Policy documents", len(manifest["documents"])); b.metric("Searchable passages", manifest["count"]); c.metric("Policy effective date", "01 Nov 2026")
        for doc in manifest["documents"]:
            with st.container(border=True):
                left, right = st.columns([4, 1])
                left.markdown(f"**{doc['title']}**")
                left.caption(f"{doc['pages']} pages · Fictional educational policy")
                path = SETTINGS.root / "data/policies/pdf" / Path(doc["name"]).name
                if path.exists(): right.download_button("Open PDF", path.read_bytes(), doc["name"], "application/pdf", key=doc["name"])

else:
    st.markdown('<div class="eyebrow">EVALUATION / RECORDED RESULTS</div>', unsafe_allow_html=True)
    st.title("Evidence over assumptions.")
    st.markdown('<div class="intro">Measured results from held-out public examples, independent workplace tickets, and policy questions.</div>', unsafe_allow_html=True)
    data = api("GET", "/evaluation")
    if data and data["available"]:
        results = data["results"]
        training, workplace, rag = (results.get(k, {}) for k in ["training_metrics", "workplace_metrics", "rag_metrics"])
        cols = st.columns(3)
        cols[0].metric("Public intent macro-F1", f"{training['fine_tuned']['macro_f1']:.3f}" if training else "Pending")
        cols[1].metric("Workplace intent macro-F1", f"{workplace['intent']['macro_f1']:.3f}" if workplace else "Pending")
        cols[2].metric("Document retrieval Recall@4", f"{rag['recall_at_4']:.1%}" if rag else "Pending")
        if training:
            st.subheader("Fine-tuning evidence")
            st.dataframe(pd.DataFrame([{"Model": name, "Accuracy": training[key]["accuracy"], "Macro-F1": training[key]["macro_f1"]}
                                      for name, key in [("TF-IDF baseline", "baseline"), ("Fine-tuned DistilBERT", "fine_tuned")]]), hide_index=True, use_container_width=True)
            path = SETTINGS.evaluation / "training_curve.png"
            if path.exists(): st.image(str(path), width="stretch")
            with st.expander("Per-intent confusion matrix"):
                path = SETTINGS.evaluation / "confusion_matrix.png"
                if path.exists(): st.image(str(path), width="stretch")
        if rag:
            st.subheader("Policy-answer evaluation")
            st.dataframe(pd.DataFrame(rag_measurements(rag), columns=["Measure", "Recorded value"]),
                         hide_index=True, use_container_width=True)
            st.caption("Known regression cases. Document retrieval, keywords and evidence-quote checks do not establish full semantic correctness. Agents should verify each answer against its cited policy.")
        review = results.get("semantic_review")
        if review:
            totals = review["totals"]
            with st.expander("Answer and summary quality checks"):
                st.write(f"Automated review judged {totals['answerable_correct']}/{totals['answerable_count']} policy answers correct and {totals['answerable_supported']}/{totals['answerable_count']} fully supported.")
                st.write(f"For summaries, {totals['summaries_retaining_main_facts']}/{totals['summary_count']} retained the main facts and {totals['summaries_without_invented_facts']}/{totals['summary_count']} contained no invented facts.")
                st.caption("The review uses the same local model family as generation. These are automated ratings, not independent human validation.")
        failures = data.get("failure_cases", {})
        if failures.get("intent"):
            with st.expander("Review intent-classification errors"):
                st.dataframe(pd.DataFrame(failures["intent"]), hide_index=True, use_container_width=True)
        if failures.get("rag"):
            with st.expander("Review unanswered policy questions"): st.json(failures["rag"])
        st.download_button("Download evaluation data", json.dumps(results, indent=2),
                           "evaluation-results.json", "application/json")
    elif data: st.info("No evaluation has been recorded yet. Results will appear after the evaluation pipeline runs.")
