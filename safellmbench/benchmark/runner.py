"""
End-to-end benchmark pipeline:

    seed prompt  --generator-->  jailbreak prompt
                                       |
                                       v
                          target model (OpenAI API)
                                       |
                                       v
                          safety classifier
                                       |
                                       v
                      CSV + JSON summary + PNG plot

The target can be either:
  - a local HuggingFace model id we spin up ourselves (uses `--model`), OR
  - a remote OpenAI-compatible server (uses `--base-url` and `--api-model`).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

from .. import config
from ..models.classifier import load_classifier
from ..models.generator import load_generator
from .seeds import DEFAULT_SEEDS


# ---------------------------------------------------------------------------
# OpenAI-compat client (deliberately tiny — no dependency on `openai` package)
# ---------------------------------------------------------------------------
@dataclass
class OpenAIClient:
    base_url: str
    api_key: str = "sk-not-needed"
    timeout: float = 300.0

    def chat(self, model: str, user_prompt: str,
             system_prompt: Optional[str] = None,
             max_tokens: int = 256,
             temperature: float = 0.7,
             top_p: float = 0.9) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        r = requests.post(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature,
                  "top_p": top_p},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


def _wait_for_health(base_url: str, timeout_s: float = 300.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"{base_url.rstrip('/')}/health", timeout=2)
            if r.ok:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server at {base_url} did not become healthy in {timeout_s}s")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    model_id: str                     # HF id (used for spawn-server mode or as OpenAI model name)
    output_dir: Path
    n_samples: int = 100
    base_url: Optional[str] = None    # if set: talk to remote server, don't spawn one
    api_model: Optional[str] = None   # override model name sent in requests
    api_key: str = "sk-not-needed"
    gen_temperature: float = 0.85
    gen_top_p: float = 0.92
    tgt_temperature: float = 0.7
    tgt_top_p: float = 0.9
    tgt_max_tokens: int = 256
    seeds: List[str] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    seed: int = 42
    server_host: str = config.DEFAULT_HOST
    server_port: int = config.DEFAULT_PORT
    skip_generator: bool = False      # if True, use seeds directly (no LoRA rewriting)


def _spawn_target_server(cfg: BenchmarkConfig) -> subprocess.Popen:
    """Start the OpenAI-compatible target server as a subprocess."""
    cmd = [
        sys.executable, "-m", "safellmbench.cli", "serve",
        "--model", cfg.model_id,
        "--host", cfg.server_host,
        "--port", str(cfg.server_port),
    ]
    print(f"[runner] spawning target server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def run_benchmark(cfg: BenchmarkConfig) -> dict:
    """Execute the full pipeline and write artifacts to `cfg.output_dir`.

    Returns the summary dict that also gets written to `summary.json`.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)

    # ---- 1. Load generator + classifier locally -----------------------------
    print("[runner] loading safety classifier ...")
    classifier = load_classifier()

    generator = None
    if not cfg.skip_generator:
        print("[runner] loading jailbreak generator ...")
        generator = load_generator()

    # ---- 2. Build the pool of jailbreak prompts -----------------------------
    pool = (cfg.seeds * (cfg.n_samples // len(cfg.seeds) + 2))[: cfg.n_samples]
    random.shuffle(pool)

    print(f"[runner] preparing {cfg.n_samples} attack prompts ...")
    attacks: List[dict] = []
    for i, seed_q in enumerate(pool, 1):
        if generator is not None:
            jb = generator.rewrite(
                seed_q,
                temperature=cfg.gen_temperature,
                top_p=cfg.gen_top_p,
            )
        else:
            jb = seed_q
        attacks.append({"seed": seed_q, "jailbreak": jb})
        if i % 10 == 0 or i == cfg.n_samples:
            print(f"  generated {i}/{cfg.n_samples}")

    # Free generator VRAM before we load the target
    if generator is not None:
        del generator
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- 3. Set up the target: spawn server OR use --base-url ---------------
    proc: Optional[subprocess.Popen] = None
    if cfg.base_url:
        base_url = cfg.base_url
        api_model = cfg.api_model or cfg.model_id
        print(f"[runner] using remote OpenAI-compat endpoint at {base_url}")
    else:
        base_url = f"http://{cfg.server_host}:{cfg.server_port}"
        api_model = cfg.model_id
        proc = _spawn_target_server(cfg)
        print("[runner] waiting for target server to warm up ...")
        try:
            _wait_for_health(base_url, timeout_s=600.0)
        except Exception:
            proc.terminate()
            raise

    client = OpenAIClient(base_url=base_url, api_key=cfg.api_key)

    # ---- 4. Attack the target model + score ---------------------------------
    results: List[dict] = []
    try:
        for i, item in enumerate(attacks, 1):
            try:
                resp = client.chat(
                    api_model, item["jailbreak"],
                    max_tokens=cfg.tgt_max_tokens,
                    temperature=cfg.tgt_temperature,
                    top_p=cfg.tgt_top_p,
                )
            except Exception as e:
                resp = f"[ERROR] {e}"
            is_jb, prob = classifier.score(resp)
            results.append({
                "seed": item["seed"],
                "jailbreak_prompt": item["jailbreak"],
                "response": resp,
                "is_jailbreak": bool(is_jb),
                "prob": float(prob),
            })
            if i % 5 == 0 or i == len(attacks):
                print(f"  attacked {i}/{len(attacks)}   "
                      f"(current jailbreak rate: "
                      f"{sum(r['is_jailbreak'] for r in results)/i:.1%})")
    finally:
        if proc is not None:
            print("[runner] shutting down spawned target server")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ---- 5. Write artifacts -------------------------------------------------
    return _write_artifacts(cfg, results)


def _write_artifacts(cfg: BenchmarkConfig, results: List[dict]) -> dict:
    import csv

    out = cfg.output_dir
    n = len(results)
    n_jb = sum(1 for r in results if r["is_jailbreak"])
    rate = n_jb / n if n else 0.0
    mean_prob = sum(r["prob"] for r in results) / n if n else 0.0

    summary = {
        "model_id": cfg.model_id,
        "api_model": cfg.api_model or cfg.model_id,
        "n_samples": n,
        "n_jailbreak": n_jb,
        "n_safe": n - n_jb,
        "jailbreak_rate": round(rate, 4),
        "mean_probability": round(mean_prob, 4),
        "threshold": config.CLASSIFIER_THRESHOLD,
        "generator_used": not cfg.skip_generator,
        "seed": cfg.seed,
        "timestamp": int(time.time()),
    }

    # summary.json
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # responses.csv — full per-sample record
    with (out / "responses.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["seed", "jailbreak_prompt", "response",
                           "is_jailbreak", "prob"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    # results.json — full per-sample record in JSON too
    (out / "results.json").write_text(json.dumps(results, indent=2))

    # plot
    try:
        _plot_summary(out, results, summary)
    except Exception as e:
        print(f"[runner] plotting skipped: {e}")

    # human-readable report
    _write_report(out, summary)

    print()
    print(f"[runner] DONE. results in: {out.resolve()}")
    print(f"[runner] jailbreak rate: {rate:.1%}  "
          f"({n_jb}/{n} responses classified as harmful)")
    return summary


def _plot_summary(out: Path, results: List[dict], summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = [r["prob"] for r in results]
    jb = summary["n_jailbreak"]
    safe = summary["n_safe"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].pie([jb, safe], labels=["Jailbreak", "Safe"],
                autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Outcomes")

    axes[1].hist(probs, bins=20, color="#3498db", edgecolor="white")
    axes[1].axvline(config.CLASSIFIER_THRESHOLD, color="red", linestyle="--")
    axes[1].set_xlabel("Jailbreak probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Probability distribution")

    axes[2].bar(["Jailbreak", "Safe"], [jb, safe],
                color=["#e74c3c", "#2ecc71"])
    axes[2].set_title(f"{summary['model_id']}")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out / "redteam_results.png", dpi=120)
    plt.close(fig)


def _write_report(out: Path, summary: dict) -> None:
    text = (
        "# SafeLLMBench report\n\n"
        f"- **Model:** `{summary['model_id']}`\n"
        f"- **Samples:** {summary['n_samples']}\n"
        f"- **Jailbreak rate:** **{summary['jailbreak_rate']:.1%}** "
        f"({summary['n_jailbreak']} / {summary['n_samples']})\n"
        f"- **Mean harmful probability:** {summary['mean_probability']:.3f}\n"
        f"- **Classifier threshold:** {summary['threshold']}\n"
        f"- **Generator used:** {summary['generator_used']}\n\n"
        f"Artifacts:\n"
        f"- `summary.json`\n"
        f"- `responses.csv`\n"
        f"- `results.json`\n"
        f"- `redteam_results.png`\n"
    )
    (out / "report.md").write_text(text)
