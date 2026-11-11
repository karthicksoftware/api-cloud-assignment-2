"""Reader-facing evaluation measures, separate from internal review tracking."""

def rag_measurements(metrics):
    rows = []
    for key, label in [
        ("count", "Total evaluation questions"),
        ("answerable_count", "Answerable policy questions"),
    ]:
        if key in metrics: rows.append([label, str(metrics[key])])
    for key, label in [
        ("recall_at_4", "Document retrieval Recall@4"),
        ("answerable_nonabstention_rate", "Answerable questions receiving cited responses"),
        ("fact_keyword_pass_rate", "Fact-keyword check pass rate (automatic)"),
        ("unsupported_abstention_rate", "Unsupported questions declined"),
        ("adversarial_forbidden_phrase_rate", "Forbidden-phrase occurrence in attack regressions"),
        ("evidence_quote_validation_rate", "Answered responses passing evidence-quote checks"),
    ]:
        if key in metrics: rows.append([label, f"{metrics[key]:.1%}"])
    if "generation_failures" in metrics and metrics.get("generation_evaluated"):
        rows.append(["Response-generation failures", str(metrics["generation_failures"])])
    if "mean_latency_ms" in metrics:
        rows.append(["Mean response time", f"{metrics['mean_latency_ms']/1000:.2f} seconds"])
    return rows
