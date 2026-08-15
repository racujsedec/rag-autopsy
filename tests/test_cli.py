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
