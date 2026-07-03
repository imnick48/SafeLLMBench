<div align="center">

# SafeLLMBench: Stress Testing the Safety Boundaries of Large Language Models

**An open safety benchmark for any HuggingFace LLM.**

Give it a HuggingFace model id — it fine-tunes and applies a LoRA jailbreak generator, attacks the model over an OpenAI-compatible API, scores every response with a custom-trained safety classifier, and drops a full report in a folder.

</div>

---

## Why this project stands out

- **LoRA-based fine-tuning** for efficient jailbreak prompt generation, not hand-written attack templates
- **A safety classifier trained from scratch** — custom BPE tokenizer + Transformer encoder, not a wrapped third-party moderation API
- **A real CLI, not just notebooks** — `pip install`, `setup`, `run`, done. The original research notebooks are still included for anyone who wants to retrain the pipeline from zero
- **Bring-your-own-target** — benchmark a model SafeLLMBench downloads and serves for you, *or* point it at any OpenAI-compatible endpoint you already have running
- **Confidence-aware reporting** — probability histograms and outcome summaries, not just a pass/fail count

> **Intended use.** This is a research/evaluation tool for probing the jailbreak-resistance of models you own or are authorized to test. Treat generated jailbreak prompts and target responses as sensitive research artifacts — the same way you'd treat any other red-team output.

---

## What it does — in one picture

```
seed prompt ──► LoRA jailbreak generator ──► adversarial prompt
                                                     │
                                                     ▼
                                        OpenAI-compatible API (localhost:3000,
                                           or any endpoint you already run)
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

## Quick start

```bash
# 1. Install the package
pip install -e .

# 2. Download the pretrained bundles (generator LoRA + classifier, ~575 MB total)
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

Already have a model running locally (vLLM, Ollama, llama.cpp, LM Studio, …)? Point `run` at it instead of letting SafeLLMBench spawn its own:

```bash
safellmbench run --model my-local-model --base-url http://localhost:8000 --api-model llama3.1:8b
```

---

## Results snapshot

### Generator training
The LoRA loss drops sharply and then stabilizes, which is exactly what you want from a clean fine-tuning run.

![Generator loss](assets/generator_loss.png)

### Classifier training
The classifier learns quickly. Validation metrics improve early, then start to wobble later, which signals mild overfitting near the end of training.

![Classifier training](assets/classifier_training.png)

### Red-team evaluation
On **100 generated attack samples** against the target model(Qwen3-1.7B), the classifier marked:

- **48 jailbreak**
- **52 safe**

The probability distribution is strongly bimodal, which means the classifier isn't drifting in the middle — it's making hard calls.

![Red-team results](assets/redteam_results.png)

---

## CLI

```
safellmbench setup                   # one-time: download bundles from Drive
safellmbench info                    # show install status
safellmbench serve --model MODEL     # OpenAI-compatible server on :3000
safellmbench run --model MODEL       # full benchmark
safellmbench score --input file.csv  # score an existing CSV of responses
```

### `safellmbench run` — flags

| flag | meaning | default |
|---|---|---|
| `--model` | HuggingFace model id to benchmark (or a label, if used with `--base-url`) | *(required)* |
| `--samples` | number of jailbreak attempts | `100` |
| `--output` | output directory | `runs/<model>_<ts>/` |
| `--base-url` | talk to an **external** OpenAI-compatible endpoint instead of spawning our own | *(spawn)* |
| `--api-model` | model name to send in requests (only used with `--base-url`) | same as `--model` |
| `--api-key` | bearer token (only used with `--base-url`) | `sk-not-needed` |
| `--host` / `--port` | address of the spawned server (ignored with `--base-url`) | `127.0.0.1:3000` |
| `--no-generator` | skip the LoRA rewriter, attack with raw seeds only | off |
| `--seed` | RNG seed for reproducibility | `42` |

> **Note:** `--base-url` should be the bare origin (e.g. `http://localhost:8000`) — SafeLLMBench appends `/v1/chat/completions` itself, so a trailing `/v1` will double up and 404.

### server

`safellmbench serve --model MODEL` starts a FastAPI server implementing:

- `GET  /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET  /health`

---

## How the pipeline works

### 1) Jailbreak generator
`Qwen/Qwen3-4B-Instruct-2507` fine-tuned with LoRA to rewrite plain "red-team" seed queries into jailbreak-style attack prompts.

| setting | value |
|---|---|
| Dataset | `JailbreakV-28K/JailBreakV-28k` |
| Samples used | 28,000 |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05` |
| Learning rate | `1e-4` |
| Epochs | `3` |
| Batch size | `1` (gradient accumulation `8`) |
| Max input length | `512` |

### 2) Target server
Any HuggingFace causal LM, exposed over an OpenAI-compatible FastAPI server (or your own already-running endpoint — see [Quick start](#quick-start)). The server applies the target's own chat template whenever the tokenizer publishes one, for a fairer benchmark of instruct-tuned models than raw-text prompting.

### 3) Safety classifier
A from-scratch Transformer trained on `allenai/wildguardmix` to classify responses as harmful or unharmful. Outputs a sigmoid probability; **prob ≥ 0.5 → jailbreak**.

| setting | value |
|---|---|
| Tokenizer | custom BPE, vocab size `12,000` |
| Max sequence length | `320` |
| Transformer | `d_model=512`, `num_heads=8`, `num_layers=4`, `d_ff=2048`, `dropout=0.1` |
| Epochs | `8` |
| Batch size | `64` |
| Learning rate | `2e-4` |

**Training notes:** the generator notebook runs against a hard time budget so training exits cleanly instead of running indefinitely; the classifier notebook tracks validation loss/F1 and uses early stopping; the inference path checks the adapter checkpoint for NaNs before use.

---

## Repo layout

```
safellmbench/
├── assets
│   ├── classifier_training.png
│   ├── generator_loss.png
│   └── redteam_results.png
├── data
│   └── redteam_results.csv
├── LICENSE
├── MANIFEST.in
├── notebooks
│   ├── 01_generator_finetune.ipynb   # trains the LoRA jailbreak generator
│   ├── 02_redteam_inference.ipynb    # original research-path attack loop
│   └── 03_safety_classifier.ipynb    # trains the safety classifier
├── pyproject.toml
├── README.md
├── requirements.txt
└── safellmbench
    ├── benchmark
    │   ├── __init__.py
    │   ├── runner.py                 # packaged version of the attack loop
    │   └── seeds.py                  # default red-team seed prompts
    ├── cli.py
    ├── config.py
    ├── __init__.py
    ├── models
    │   ├── classifier.py
    │   ├── generator.py
    │   ├── __init__.py
    │   └── target.py
    ├── server
    │   ├── __init__.py
    │   └── openai_server.py
    └── setup_bundles.py
```

---

## Requirements

- Python ≥ 3.9
- CUDA-capable GPU strongly recommended (Qwen3-4B generator + Qwen3-1.7B target fits comfortably on a single 24 GB GPU; larger targets need more)
- Bundles are downloaded on first run to `~/.safellmbench/` (~500 MB adapter + 75 MB classifier)

**Runtime / CLI** (`requirements.txt`): `torch`, `transformers`, `tokenizers`, `peft`, `accelerate`, `safetensors`, `sentencepiece`, `fastapi`, `uvicorn`, `pydantic`, `requests`, `gdown`, `matplotlib`, `pandas`, `tqdm`, `huggingface_hub`.

**Additional, training-only** (only needed if you retrain via the notebooks): `datasets`, `bitsandbytes`, `scikit-learn`.

---

## Getting the model weights

`safellmbench setup` downloads the pretrained bundles automatically — no manual steps required:

```bash
safellmbench setup      # fetches trained_models.zip + classifier_bundle.zip via gdown
safellmbench info       # confirms install paths and status
```

Bundles land in `~/.safellmbench/` (override with the `SAFELLMBENCH_HOME` environment variable). Heavy `checkpoint-*/` optimizer states are pruned automatically after install to save space. `run` and `score` both call `setup` automatically on first use if bundles aren't found yet, so this step is optional but recommended to run once up front so you can watch it succeed before kicking off a full benchmark.

---

## Reproducing / retraining from scratch

**Fastest path — use the pretrained bundles:** just run the [Quick start](#quick-start) above.

**Full retrain:**

1. Run `notebooks/01_generator_finetune.ipynb` to fine-tune the LoRA jailbreak generator.
2. Run `notebooks/03_safety_classifier.ipynb` to train the safety classifier and BPE tokenizer.
3. Drop your retrained artifacts into the same `~/.safellmbench/` layout that `setup` produces (run `safellmbench info` to see the exact expected paths).
4. Attack a target with `safellmbench run`, or use `notebooks/02_redteam_inference.ipynb` directly if you want the original, un-packaged research loop.
5. Inspect `redteam_results.csv`/`.json`, `redteam_results.png`, `generator_loss.png`, and `classifier_training.png`.

---

## Output artifacts

- **`trained_models.zip`** — LoRA adapter bundle for the jailbreak generator
- **`classifier_bundle.zip`** — Transformer classifier checkpoint and tokenizer bundle
- **`redteam_results.csv` / `.json`** — per-sample seed prompt, generated jailbreak prompt, target response, label, and probability
- **`redteam_results.png`** — outcome summary for the red-team run
- **`generator_loss.png`** — fine-tuning loss curve for the jailbreak generator
- **`classifier_training.png`** — training and validation curves for the safety classifier
- **`report.md`** — auto-generated human-readable run summary

---

## Notes on the model

The generator, classifier, and their bundles are **compatible with the original SafeLLMBench notebooks** — the same architecture code paths run at inference time as at training time, so a checkpoint trained in the notebooks loads cleanly via `state_dict`, no key remapping needed.

Improvements the packaged CLI makes over the raw notebooks:

- Target model prompting honours the model's own chat template (via `tokenizer.apply_chat_template`) when it exposes one — a fairer test of instruct-tuned models than the notebooks' raw-text prompting.
- Bundles are cached in a proper user-config directory (`~/.safellmbench/`) instead of the working directory, and can be relocated via `SAFELLMBENCH_HOME`.
- Heavy `checkpoint-*/` optimizer states are auto-pruned after install (saves ~250 MB).
- `--base-url` support to benchmark any already-running OpenAI-compatible endpoint, not just models SafeLLMBench downloads and serves itself.
- Modular, importable, packaged — no notebook re-execution required for day-to-day benchmarking.

---

## License

MIT — see [LICENSE](LICENSE).
