from .report import RAGAutopsyReport


def format_rag_autopsy_report(
    report: RAGAutopsyReport,
) -> str:
    """Format a structured RAG autopsy report for humans."""

    lines = [
        "RAG AUTOPSY REPORT",
        "=" * 60,
        "",
        "Question:",
        report.question,
        "",
        (
            "PRIMARY DIAGNOSIS: "
            f"{report.verdict.diagnosis.value}"
        ),
        report.verdict.explanation,
        "",
        (
            "Retrieval: "
            f"{report.retrieval.diagnosis.value}"
        ),
        report.retrieval.explanation,
        "",
        "Generation:",
        report.generation.answer,
        "",
        (
            "Citation validity: "
            f"{report.citations.diagnosis.value}"
        ),
        report.citations.explanation,
        "",
        (
            "Citation support: "
            f"{report.citation_support.diagnosis.value}"
        ),
        report.citation_support.explanation,
        "",
        "Retrieved chunks:",
    ]

    if not report.retrieval_results:
        lines.append("None")
    else:
        for rank, result in enumerate(
            report.retrieval_results,
            start=1,
        ):
            lines.append(
                f"{rank}. {result.chunk.chunk_id} "
                f"| score={result.score:.4f}"
            )

    return "\n".join(lines)
