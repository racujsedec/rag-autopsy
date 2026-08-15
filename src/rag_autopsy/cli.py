import argparse


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="rag-autopsy",
        description=(
            "RAG Autopsy diagnostic and evaluation CLI."
        ),
    )


def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_parser()

    if argv == []:
        parser.print_help()
        return 0

    parser.parse_args(argv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
