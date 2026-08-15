import pytest

from rag_autopsy.cli import build_parser, main


def test_parser_uses_project_name() -> None:
    parser = build_parser()

    assert parser.prog == "rag-autopsy"


def test_help_displays_project_description(
    capsys,
) -> None:
    with pytest.raises(
        SystemExit
    ) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0

    output = capsys.readouterr().out

    assert "RAG Autopsy" in output
    assert "diagnostic" in output.lower()


def test_no_command_displays_help(
    capsys,
) -> None:
    exit_code = main([])

    assert exit_code == 0

    output = capsys.readouterr().out

    assert "usage:" in output
    assert "rag-autopsy" in output


def test_autopsy_command_accepts_question(
    capsys,
) -> None:
    exit_code = main(
        [
            "autopsy",
            "--question",
            "What happened?",
        ]
    )

    assert exit_code == 0

    output = capsys.readouterr().out

    assert "What happened?" in output


def test_autopsy_command_requires_question() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["autopsy"])

    assert exc_info.value.code == 2


def test_autopsy_help_lists_question_option(
    capsys,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "autopsy",
                "--help",
            ]
        )

    assert exc_info.value.code == 0

    output = capsys.readouterr().out

    assert "--question" in output
    assert "--top-k" in output


def test_autopsy_command_accepts_question_id(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "rag_autopsy.cli.run_benchmark_retrieval",
        lambda question_id, top_k: (
            f"Question ID: {question_id}"
        ),
    )

    exit_code = main(
        [
            "autopsy",
            "--question-id",
            "q031",
        ]
    )

    assert exit_code == 0

    output = capsys.readouterr().out

    assert "q031" in output


def test_autopsy_requires_question_or_question_id() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["autopsy"])

    assert exc_info.value.code == 2


def test_question_id_dispatches_to_benchmark_retrieval(
    monkeypatch,
    capsys,
) -> None:
    calls = []

    def fake_run_benchmark_retrieval(
        question_id,
        top_k,
    ):
        calls.append(
            (
                question_id,
                top_k,
            )
        )
        return "BENCHMARK RETRIEVAL COMPLETE"

    monkeypatch.setattr(
        "rag_autopsy.cli.run_benchmark_retrieval",
        fake_run_benchmark_retrieval,
        raising=False,
    )

    exit_code = main(
        [
            "autopsy",
            "--question-id",
            "q031",
            "--top-k",
            "5",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            "q031",
            5,
        )
    ]

    output = capsys.readouterr().out

    assert "BENCHMARK RETRIEVAL COMPLETE" in output
