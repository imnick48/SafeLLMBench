"""
Command-line entry point for SafeLLMBench.

Sub-commands:
    setup      Download the trained bundles from Google Drive.
    serve      Start an OpenAI-compatible server for a HuggingFace model.
    run        Full benchmark (spawns its own target server unless --base-url).
    score      Re-score an existing CSV of {seed, jailbreak_prompt, response}.
    info       Show where the bundles are installed and their status.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from . import config


def _cmd_setup(args: argparse.Namespace) -> int:
    from . import setup_bundles
    setup_bundles.install(force=args.force)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server.openai_server import serve
    serve(model_id=args.model, host=args.host, port=args.port)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .benchmark.runner import BenchmarkConfig, run_benchmark

    if not config.bundles_installed():
        print("[cli] bundles missing — running `setup` first ...")
        from . import setup_bundles
        setup_bundles.install(force=False)

    out_dir = Path(args.output) if args.output else Path("runs") / (
        f"{args.model.replace('/', '__')}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    cfg = BenchmarkConfig(
        model_id=args.model,
        output_dir=out_dir,
        n_samples=args.samples,
        base_url=args.base_url,
        api_model=args.api_model,
        api_key=args.api_key,
        server_host=args.host,
        server_port=args.port,
        skip_generator=args.no_generator,
        gen_temperature=args.gen_temp,
        gen_top_p=args.gen_top_p,
        tgt_temperature=args.tgt_temp,
        tgt_top_p=args.tgt_top_p,
        tgt_max_tokens=args.tgt_max_tokens,
        seed=args.seed,
    )
    summary = run_benchmark(cfg)
    print(f"[cli] summary: {summary}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from .models.classifier import load_classifier

    if not config.bundles_installed():
        print("[cli] bundles missing — running `setup` first ...")
        from . import setup_bundles
        setup_bundles.install(force=False)

    clf = load_classifier()
    rows = []
    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "response" not in reader.fieldnames:
            print("[cli] input CSV must contain a `response` column", file=sys.stderr)
            return 2
        for row in reader:
            is_jb, prob = clf.score(row.get("response") or "")
            row["is_jailbreak"] = is_jb
            row["prob"] = prob
            rows.append(row)

    out_path = Path(args.output) if args.output else Path(args.input).with_name(
        Path(args.input).stem + "_scored.csv"
    )
    fieldnames = list(rows[0].keys()) if rows else ["response", "is_jailbreak", "prob"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_jb = sum(1 for r in rows if r["is_jailbreak"])
    print(f"[cli] scored {len(rows)} rows -> {out_path}")
    print(f"[cli] jailbreak rate: {n_jb}/{len(rows)} ({n_jb/max(len(rows),1):.1%})")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    print(f"SafeLLMBench home:  {config.HOME_DIR}")
    print(f"Generator dir:      {config.GENERATOR_DIR}   "
          f"({'ok' if (config.GENERATOR_DIR/'adapter_config.json').exists() else 'MISSING'})")
    print(f"Classifier ckpt:    {config.CLASSIFIER_CKPT}   "
          f"({'ok' if config.CLASSIFIER_CKPT.exists() else 'MISSING'})")
    print(f"BPE tokenizer:      {config.BPE_TOKENIZER_JSON}   "
          f"({'ok' if config.BPE_TOKENIZER_JSON.exists() else 'MISSING'})")
    print(f"Bundles installed:  {config.bundles_installed()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="safellmbench",
        description="Open safety benchmark for HuggingFace LLMs.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("setup", help="Download pretrained bundles from Google Drive.")
    ps.add_argument("--force", action="store_true",
                    help="Re-download even if bundles are already installed.")
    ps.set_defaults(fn=_cmd_setup)

    pv = sub.add_parser("serve",
                        help="Start an OpenAI-compatible server for a HuggingFace model.")
    pv.add_argument("--model", required=True, help="HuggingFace model id.")
    pv.add_argument("--host", default=config.DEFAULT_HOST)
    pv.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    pv.set_defaults(fn=_cmd_serve)

    pr = sub.add_parser("run", help="Run the full safety benchmark.")
    pr.add_argument("--model", required=True,
                    help="HuggingFace model id to benchmark.")
    pr.add_argument("--samples", type=int, default=100,
                    help="Number of jailbreak attempts (default: 100).")
    pr.add_argument("--output", type=str, default=None,
                    help="Output directory (default: runs/<model>_<timestamp>/).")
    pr.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint. If unset, spawn our own server.")
    pr.add_argument("--api-model", default=None,
                    help="Model name to send in API requests (default: --model).")
    pr.add_argument("--api-key", default="sk-not-needed",
                    help="Bearer token for --base-url (default: dummy).")
    pr.add_argument("--host", default=config.DEFAULT_HOST,
                    help="Host for the spawned target server.")
    pr.add_argument("--port", type=int, default=config.DEFAULT_PORT,
                    help="Port for the spawned target server.")
    pr.add_argument("--no-generator", action="store_true",
                    help="Skip the LoRA rewriter and attack with raw seeds only.")
    pr.add_argument("--gen-temp", type=float, default=0.85)
    pr.add_argument("--gen-top-p", type=float, default=0.92)
    pr.add_argument("--tgt-temp", type=float, default=0.7)
    pr.add_argument("--tgt-top-p", type=float, default=0.9)
    pr.add_argument("--tgt-max-tokens", type=int, default=256)
    pr.add_argument("--seed", type=int, default=42)
    pr.set_defaults(fn=_cmd_run)

    pc = sub.add_parser("score",
                        help="Score an existing CSV (must contain a `response` column).")
    pc.add_argument("--input", required=True)
    pc.add_argument("--output", default=None)
    pc.set_defaults(fn=_cmd_score)

    pi = sub.add_parser("info", help="Show install status of the bundles.")
    pi.set_defaults(fn=_cmd_info)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
