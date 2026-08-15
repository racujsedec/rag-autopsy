import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-autopsy",
        description=(
            "RAG Autopsy diagnostic and evaluation CLI."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    autopsy_parser = subparsers.add_parser(
        "autopsy",
        help="Run a RAG autopsy for a question.",
    )

    autopsy_parser.add_argument(
        "--question",
        required=True,
        help="Question to analyze.",
    )

    autopsy_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to inspect.",
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_parser()

    if argv == []:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.command == "autopsy":
        print(
            f"Question: {args.question}"
        )
        print(
            f"Top-k: {args.top_k}"
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
