# SafeLLMBench

**An open safety benchmark for any HuggingFace LLM.**
Give it a HuggingFace model id — it will download the model, expose it through an API on `localhost:3000`, generate adversarial jailbreak prompts, attack the model, score the responses with a trained safety classifier, and drop a full report in a folder.

Built on top of the SafeLLMBench research pipeline (LoRA jailbreak generator + custom Transformer safety classifier).

---

## Quick start

```bash
# 1. Install the package
pip install -e .

# 2. Download the pretrained bundles (generator LoRA + classifier)
safellmbench setup

# 3. Benchmark any HuggingFace instruct model
safellmbench run --model Qwen/Qwen3-1.7B --samples 100 --output ./runs/qwen17b
```

When it finishes you'll find:

```
runs/qwen17b/
├── summary.json          # top-level metrics
├── results.json          # full per-sample records
├── responses.csv         # same, CSV form
├── redteam_results.png   # pie + histogram + bar plot
└── report.md             # human-readable summary
```

## What it does — in one picture

```
seed prompt ──► LoRA jailbreak generator ──► adversarial prompt
                                                     │
                                                     ▼
                                                 API (localhost:3000)
                                                     │
                                                     ▼
                                       target HuggingFace model
                                                     │
                                                     ▼
                              custom Transformer safety classifier
                                                     │
                                                     ▼
                             CSV + JSON + PNG + Markdown report
```

## CLI

```
safellmbench setup                   # one-time: download bundles from Drive
safellmbench info                    # show install status
safellmbench serve --model MODEL     # server on :3000
safellmbench run --model MODEL       # full benchmark
safellmbench score --input file.csv  # score an existing CSV of responses
```

### `safellmbench run` — flags

| flag | meaning | default |
|---|---|---|
| `--model` | HuggingFace model id to benchmark | *(required)* |
| `--samples` | number of jailbreak attempts | `100` |
| `--output` | output directory | `runs/<model>_<ts>/` |
| `--base-url` | talk to an **external** endpoint instead of spawning our own | *(spawn)* |
| `--api-model` | model name to send in requests (only useful with `--base-url`) | same as `--model` |
| `--api-key` | bearer token (only useful with `--base-url`) | `sk-not-needed` |
| `--host` / `--port` | address of the spawned server | `127.0.0.1:3000` |
| `--no-generator` | skip the LoRA rewriter, attack with raw seeds only | off |
| `--seed` | RNG seed for reproducibility | `42` |

### server

`safellmbench serve --model MODEL` starts a FastAPI server implementing:

- `GET  /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET  /health`


## How the pipeline works

1. **Jailbreak generator** — `Qwen/Qwen3-4B-Instruct-2507` fine-tuned with LoRA
   (`r=16, α=32`) on `JailbreakV-28K/JailBreakV-28k` for 3 epochs. Rewrites
   plain "red-team" seed queries into jailbreak-style attack prompts.
2. **Target server** — any HuggingFace causal LM, exposed 
   FastAPI. The server automatically applies the target's own chat template
   whenever the tokenizer publishes one (fair benchmarking of instruct models).
3. **Safety classifier** — a from-scratch 4-layer Transformer
   (`d_model=512, heads=8, d_ff=2048, vocab=12k BPE`) trained on
   `allenai/wildguardmix` (harmful vs. unharmful responses). Outputs a
   sigmoid probability; **prob ≥ 0.5 → jailbreak**.

## Repo layout

```
safellmbench/
├── assets
│   ├── classifier_training.png
│   ├── generator_loss.png
│   └── redteam_results.png
├── data
│   └── redteam_results.csv
├── LICENSE
├── MANIFEST.in
├── notebooks
│   ├── 01_generator_finetune.ipynb
│   ├── 02_redteam_inference.ipynb
│   └── 03_safety_classifier.ipynb
├── pyproject.toml
├── README.md
├── requirements.txt
└── safellmbench
    ├── benchmark
    │   ├── __init__.py
    │   ├── runner.py
    │   └── seeds.py
    ├── cli.py
    ├── config.py
    ├── __init__.py
    ├── models
    │   ├── classifier.py
    │   ├── generator.py
    │   ├── __init__.py
    │   └── target.py
    ├── server
    │   ├── __init__.py
    │   └── openai_server.py
    └── setup_bundles.py
```

## Requirements

- Python ≥ 3.9
- CUDA-capable GPU strongly recommended (Qwen3-4B generator + Qwen3-1.7B target
  fits comfortably on a single 24 GB GPU; larger targets need more).
- Bundles are downloaded on first run to `~/.safellmbench/`
  (~ 500 MB adapter + 75 MB classifier).

## Notes on the model

The generator, classifier and their bundles are **compatible with the original
SafeLLMBench notebooks** — the same architecture code paths are used at
inference time as at training time, so the trained checkpoint's
`state_dict` loads cleanly without any key remapping.

Small improvements over the notebooks in this package:

- Target model prompting now honours the model's own chat template (via
  `tokenizer.apply_chat_template`) when it exposes one — a fairer test of
  instruct-tuned models than the notebook's raw-text prompting.
- Bundles are cached in a proper user-config directory (`~/.safellmbench/`)
  instead of the working directory.
- Heavy `checkpoint-*/` optimizer states are auto-pruned after install
  (saves ~250 MB).
- Modular, importable, packaged — no notebook re-execution required.

## License

MIT (see `LICENSE`).
