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

    question_group = (
        autopsy_parser.add_mutually_exclusive_group(
            required=True
        )
    )

    question_group.add_argument(
        "--question",
        help="Question to analyze.",
    )

    question_group.add_argument(
        "--question-id",
        help="Benchmark question ID to analyze.",
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
        if args.question_id:
            print(
                f"Question ID: {args.question_id}"
            )
        else:
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
