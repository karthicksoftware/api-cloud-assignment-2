"""Conversation context and deterministic evidence checks (not semantic proof)."""
import re
import unicodedata


TOPIC = re.compile(r"\b(?:refund\w*|return\w*|ship\w*|deliver\w*|parcel\w*|invoice\w*|"
                   r"payment\w*|order\w*|account\w*|password\w*|newsletter\w*|"
                   r"promot\w*|discount\w*|cancel\w*|support|warranty|complaint\w*)\b", re.I)
REFERENCE = re.compile(r"\b(?:it|its|that|those|this|these|they|them|then|same)\b|^(?:and|what about|how about)\b", re.I)


def conversation_query(request):
    """Keep the last explicit user topic through successive anaphoric questions.

    Explicit new topics start a new retrieval query. Assistant responses are never
    retrieval evidence, so a wrong earlier answer cannot become the topic anchor.
    """
    question = request.message.strip()
    previous = [turn.content for turn in request.history if turn.role == "user"]
    if previous and REFERENCE.search(question) and not TOPIC.search(question):
        anchor = next((text for text in reversed(previous) if TOPIC.search(text)), previous[-1])
        question = anchor[:500] + "\nFollow-up question: " + question
    if request.ticket_context:
        question += "\nCustomer issue: " + request.ticket_context[:1000]
    return question


def normalise_evidence(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", re.sub(r"\s*([^\w\s])\s*", r"\1", text)).strip()


def readable_policy_sentence(sentence, source=None):
    """Restore ordinary punctuation in a source sentence decoded from WordPieces."""
    text = sentence.strip()
    if source:
        prefix = source["section"] + " " + source["title"] + " "
        if text.casefold().startswith(prefix.casefold()): text = text[len(prefix):]
    text = re.sub(r"\s+'\s+s\b", "'s", text)
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    text = re.sub(r"\s*[-–]\s*", lambda match: "–" if "–" in match[0] else "-", text)
    text = re.sub(r"(\d{1,2}):\s+(\d{2})", r"\1:\2", text)
    for term in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "India", "DemoMart"]:
        text = re.sub(r"\b" + term + r"\b", term, text, flags=re.I)
    text = re.sub(r"\bist\b", "IST", text, flags=re.I)
    for term in ["My Orders", "Edit order", "Download invoice"]:
        text = re.sub(r"\b" + re.escape(term) + r"\b", term, text, flags=re.I)
    return text[:1].upper() + text[1:]


def evidence_sentences(sources):
    # The demonstration email domain is tokenised as 'demomart. example'. Keep
    # it inside its sentence; the strings still come directly from the index.
    return list(dict.fromkeys(sentence.strip() for source in sources
                for sentence in re.split(r"(?<=[.!?])\s+(?!example\b)", source["text"])
                if len(sentence.strip()) >= 12))


def numeric_values(text):
    def hour_24(match):
        hour = int(match[1]) % 12 + (12 if match[3].casefold() == "pm" else 0)
        return f"{hour}:{match[2]}"
    text = re.sub(r"\b(1[0-2]|0?[1-9]):(\d\d)\s*(am|pm)\b", hour_24, text, flags=re.I)
    words = {word: str(i) for i, word in enumerate(
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])}
    text = re.sub(r"\b(?:" + "|".join(words) + r")\b", lambda m: words[m[0].lower()], text, flags=re.I)
    return {int(n) for n in re.findall(r"\b\d+\b", text)}


def explicit_policy_prohibition(question, sources):
    """Quote a directly matching prohibition when generation misreads 'no' as unknown.

    Requires a permission question, a negative policy sentence, and three shared
    substantive word roots. This is a conservative extractive fallback.
    """
    if not re.search(r"\bcan\s+i\b|\bis\s+it\s+(?:possible|allowed)\b", question, re.I):
        return None
    stop = {"can", "the", "that", "this", "have", "been", "after", "before", "with", "from", "does", "what", "would", "when", "your", "their"}
    def roots(value):
        return {word[:5] for word in re.findall(r"[a-z]{4,}", value.casefold()) if word not in stop}
    query_roots = roots(question)
    candidates = []
    for source in sources:
        for sentence in evidence_sentences([source]):
            if not re.search(r"\b(?:cannot|may not|not permitted|not supported|no longer)\b", sentence, re.I):
                continue
            overlap = len(query_roots & roots(sentence))
            if overlap >= 3: candidates.append((overlap, source["score"], source, sentence))
    if not candidates: return None
    _, _, source, sentence = max(candidates, key=lambda item: item[:2])
    claim = {"text": readable_policy_sentence(sentence, source), "source_id": source["id"], "quote": sentence}
    return claim if checked_claims([claim], sources)[1] is None else None


def explicit_address_change(question, sources):
    """Use the complete documented self-service path for an unshipped order."""
    if not (re.search(r"\b(?:address|destination)\b", question, re.I) and
            re.search(r"\b(?:order|purchase)\b", question, re.I) and
            re.search(r"\b(?:chang\w*|updat\w*|replac\w*)\b", question, re.I) and
            re.search(r"\b(?:dispatch\w*|ship\w*)\b", question, re.I)):
        return None
    for source in sources:
        sentences = evidence_sentences([source])
        condition = next((s for s in sentences if "shipping address" in s.casefold() and "only before dispatch" in s.casefold()), None)
        procedure = next((s for s in sentences if "my orders" in s.casefold() and "edit order" in s.casefold()), None)
        fallback = next((s for s in sentences if "if edit order is unavailable" in s.casefold()), None)
        if condition and procedure and fallback:
            claims = [{"text": readable_policy_sentence(s, source), "source_id": source["id"], "quote": s}
                      for s in [condition, procedure, fallback]]
            if checked_claims(claims, sources)[1] is None: return claims
    return None


def explicit_policy_fact(question, sources):
    """Use directly quoted policy sentences for common procedural lookups."""
    question = question.casefold()
    invoice = "invoice" in question and any(word in question for word in ["download", "obtain", "where", "how"])
    hours = (any(word in question for word in ["support", "agent", "human"]) and
             any(word in question for word in ["hours", "available", "open"]))
    if not (invoice or hours): return None
    hours_claim = None
    outside_claim = None
    for source in sources:
        for sentence in evidence_sentences([source]):
            lower = sentence.casefold()
            if invoice and not ("my orders" in lower and "download invoice" in lower): continue
            if hours and not (("human agents are available" in lower and "09" in lower and "18" in lower) or
                              ("outside operating hours" in lower and "initial response" in lower)): continue
            claim = {"text": readable_policy_sentence(sentence, source), "source_id": source["id"], "quote": sentence}
            if checked_claims([claim], sources)[1] is not None: continue
            if invoice: return [claim]
            if "human agents are available" in lower: hours_claim = claim
            if "outside operating hours" in lower: outside_claim = claim
    if hours:
        wants_outside_response = "outside" in question or "leave a message" in question
        if wants_outside_response and outside_claim:
            claims = [c for c in [hours_claim, outside_claim] if c]
            return claims if checked_claims(claims, sources)[1] is None else None
        if not wants_outside_response and hours_claim: return [hours_claim]
    return None


def explicit_refund_followup(question, resolved_query, sources):
    """Preserve the documented escalation step for a missing approved refund."""
    if not ("refund" in resolved_query.casefold() and
            re.search(r"\b(?:not arrived|hasn['’]?t arrived|not received|still missing)\b", question, re.I)):
        return None
    for source in sources:
        for sentence in evidence_sentences([source]):
            lower = sentence.casefold()
            if not ("approved refund has not arrived after 7 business days" in lower and "contact support" in lower):
                continue
            claim = {"text": readable_policy_sentence(sentence, source), "source_id": source["id"], "quote": sentence}
            if checked_claims([claim], sources)[1] is None: return [claim]
    return None


def checked_claims(claims, sources):
    """Validate quoted evidence, numeric claims and selected policy qualifiers.

    Exact evidence and IDs establish traceability. They do not establish that a
    paraphrase is entailed; recorded semantic review is a separate evaluation.
    """
    if not isinstance(claims, list) or not 1 <= len(claims) <= 4:
        return None, "invalid_or_missing_citations"
    allowed = {source["id"]: source for source in sources}
    checked = []
    for claim in claims:
        if not isinstance(claim, dict): return None, "invalid_or_missing_citations"
        text, source_id, quote = (claim.get(k) for k in ["text", "source_id", "quote"])
        if (not isinstance(text, str) or not text.strip() or
                not isinstance(source_id, str) or source_id not in allowed or
                not isinstance(quote, str) or len(quote.strip()) < 12):
            return None, "invalid_or_missing_citations"
        # The backend, not model prose or previous answers, supplies citation tags.
        if "[" in text or "]" in text: return None, "invalid_or_missing_citations"
        evidence = normalise_evidence(allowed[source_id]["text"])
        excerpt = normalise_evidence(quote)
        if excerpt not in evidence: return None, "unsupported_evidence_quote"
        numbers = numeric_values(text)
        if not numbers <= numeric_values(quote):
            return None, "unsupported_numeric_claim"
        # Include the surrounding sentence so quoting only '30 calendar days'
        # cannot remove 'target' and 'verified request' from the same policy rule.
        start = evidence.index(excerpt)
        sentence_start = evidence.rfind(".", 0, start) + 1
        sentence_end = start + len(excerpt) if excerpt.endswith(".") else evidence.find(".", start + len(excerpt))
        context = evidence[sentence_start:sentence_end if sentence_end >= 0 else len(evidence)]
        if re.search(r"\bfollow\b.*\bemail\b", excerpt) and not re.search(r"\bemail\b", text, re.I):
            return None, "missing_policy_qualification"
        if "initial response" in excerpt and not re.search(r"\b(?:initial|first)\b", text, re.I):
            return None, "missing_policy_qualification"
        if "if edit order is unavailable" in excerpt and not (
                "if" in text.casefold().split() and "unavailable" in text.casefold()):
            return None, "missing_policy_qualification"
        if "address" in text.casefold() and "address" not in excerpt and "edit order" not in excerpt:
            return None, "unsupported_evidence_quote"
        if "newsletter" in excerpt and re.search(r"\b(?:unsubscribe|disabl\w*)\b", text, re.I) and "newsletter" not in text.casefold():
            return None, "missing_policy_qualification"
        if numbers:
            requirements = [(r"\btarget\b", r"\btarget\b"),
                            (r"\bverified\b", r"\bverif\w*\b"),
                            (r"\bbusiness\s*-?\s*days?\b", r"\bbusiness\s*-?\s*days?\b"),
                            (r"\bcalendar\s+days?\b", r"\bcalendar\s+days?\b"),
                            (r"\bafter dispatch\b", r"\bafter dispatch\b"),
                            (r"\bof delivery\b", r"\b(?:of|after|from) delivery\b"),
                            (r"\bapproved refund\b", r"\bapprov\w*\b")]
            for policy_pattern, answer_pattern in requirements:
                if re.search(policy_pattern, context) and not re.search(answer_pattern, text, re.I):
                    return None, "missing_policy_qualification"
        checked.append({"text": text.strip(), "source_id": source_id, "quote": quote.strip()})
    return checked, None
