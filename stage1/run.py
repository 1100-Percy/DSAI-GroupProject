"""Command-line entry point. Set deterministic environment before Python starts."""
import argparse
import os

from .data_io import check_inputs, record_environment
from .embeddings import extract_all
from .probe import run_probe, verify_reproducibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "extract", "probe", "figures", "build-notebook"])
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--verify-reproducibility", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Start Python with PYTHONHASHSEED=0")
    import torch
    torch.manual_seed(42)
    record_environment(args.command)
    if args.command == "check":
        check_inputs()
    elif args.command == "extract":
        extract_all()
    elif args.command == "probe":
        verify_reproducibility() if args.verify_reproducibility else run_probe(args.recompute)
    elif args.command == "figures":
        from .figures import create_figures, write_reports
        create_figures(recompute=args.recompute)
        write_reports()
    else:
        from .build_notebook import build_notebook
        build_notebook()


if __name__ == "__main__":
    main()
