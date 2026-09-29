# Agentic Utils

A collection of standalone Google Colab notebooks: coding-agent studios, model-serving
servers, GGUF conversion tooling, and LLM evaluation experiments. Each notebook is
self-contained — open it in Colab, fill in the settings cell(s), and run.

## Notebooks

### `OpenSpec_Studio.ipynb`
**What:** Spins up [OpenSpec](https://github.com/Fission-AI/OpenSpec) as a small web studio inside Colab, with a coding agent (Codex, Claude, Gemini, opencode, Cursor, GitHub Copilot, ...) wired in to work the spec.
**How:** Installs OpenSpec and the chosen agent CLI, writes a minimal `server.py` + static UI (`index.html`/`app.css`/`app.js`), starts the server, and exposes it publicly (e.g. via a tunnel) so you can create/open projects from the browser.
**Why:** Lets you drive an OpenSpec-based coding-agent workflow from a disposable Colab runtime instead of a local machine.
**Output:** A running studio UI at a public URL where projects are created and OpenSpec commands are executed through the selected agent.

### `archify_studio_colab.ipynb`
**What:** Runs [Archify](https://github.com/tt-a1i/archify) (a project-architecture visualizer/studio) on Colab and exposes it through a Cloudflare quick tunnel.
**How:** Step-by-step cells: clone & install Archify, install `cloudflared`, write the studio server and index page, write a sample project-architecture JSON, wire in an OpenAI key + private access key, start the server on port 8000, open the tunnel, verify the public link responds, then stop everything.
**Why:** Gives a shareable, temporary link to inspect/browse a project's architecture without any local setup.
**Output:** A public Cloudflare URL serving the Archify studio UI, plus a small JSON describing the sample project architecture.

### `FreeToken_colab_server_L4.ipynb`
**What:** Serves an LLM (default: `nvidia/Qwen3.6-35B-A3B-NVFP4`) via [FreeToken](https://github.com/) on a Colab GPU (needs sm_86+, i.e. L4/A100/RTX 30-40-50 class) and exposes an OpenAI-compatible endpoint.
**How:** Checks GPU compute capability, installs FreeToken + CUDA 13 libs + cloudflared, launches `ft serve` in the background with MoE slot sizing derived from available VRAM, waits for a real 1-token completion to confirm readiness, then opens a Cloudflare tunnel to the local server.
**Why:** Turns a free/cheap Colab GPU runtime into a temporary, internet-reachable OpenAI-compatible inference server for quick testing.
**Output:** A public URL (`PUBLIC_URL`) serving `/v1/chat/completions`, testable directly with the OpenAI Python client or `curl`.

### `omniroute_colab_server.ipynb`
**What:** Minimal launcher for the `omniroute` CLI server on Colab, exposed via a Cloudflare tunnel.
**How:** `npm install -g omniroute`, run it in the background with `nohup`, install `cloudflared`, then tunnel `localhost:20128` to a public URL.
**Why:** Quick way to get an Omniroute instance reachable from outside Colab with almost no setup.
**Output:** Log output confirming the server started, plus a public tunnel URL in `cloudflared.log`.

### `Ollama_PublicURL.ipynb`
**What:** Runs [Ollama](https://ollama.com) on a Colab GPU and exposes it through a public URL (Cloudflare tunnel or ngrok).
**How:** Installs Ollama and `cloudflared`, starts `ollama serve` in the background, pulls a chosen model (several presets commented out, e.g. `llama3.2:3b-instruct-fp16`, `qwen3:8b`, `gpt-oss:20b`), then opens either a Cloudflare quick tunnel or an ngrok tunnel (using an `NGROK_AUTHTOKEN` secret) to `localhost:11434`, and includes a small `query_ollama` helper plus a keep-alive loop.
**Why:** Gets a disposable, internet-reachable Ollama server running on free/cheap Colab GPU hardware for quick model testing.
**Output:** A public URL serving Ollama's `/api/chat` endpoint, plus a sample response printed from `query_ollama`.

### `VLLM_Quantized_PublicURL.ipynb`
**What:** Serves a quantized model (default: `IFM/K2-Horizon-7B`) with [vLLM](https://github.com/vllm-project/vllm) on Colab and exposes it via a Cloudflare tunnel.
**How:** Picks a dtype from the GPU's compute capability, launches `vllm serve` with tool-calling/reasoning parsers enabled, waits on `/health`, opens a Cloudflare tunnel, then drives the endpoint both with an async OpenAI client (streaming, bounded by a semaphore) and with a small `curl`+`jq` shell helper for concurrent prompts.
**Why:** Quick way to stand up and load-test an OpenAI-compatible vLLM server for a quantized model without local infrastructure.
**Output:** A public `/v1` endpoint, streamed completions for a batch of sample queries (printed and saved to `out.*.txt`), and `vllm.log`/`cf.log` for debugging.

### `gguf_convert.ipynb`
**What:** End-to-end GGUF conversion pipeline for diffusion/LLM checkpoints, built on `city96/ComfyUI-GGUF` and `llama.cpp`.
**How:** Fetches a checkpoint, detects its architecture (read live from ComfyUI-GGUF so new upstream families work without edits), figures out which tensors must stay F32 (including single-row parameters upstream's `n_dims == 1` rule misses), works out which quant targets are actually achievable given the source, converts, quantizes, and verifies the result against the original. Fill in settings 1a-1e and "Runtime → Run all"; a "survey a folder" mode can header-scan many checkpoints without downloading.
**Why:** Automates a fiddly, error-prone conversion process and avoids silently producing a broken or mis-typed GGUF.
**Output:** Converted/quantized `.gguf` file(s), a Markdown conversion report (architecture, source hash, F32-kept tensors, build settings), and optional upload of the result.

### `llm_sglang_tpu.ipynb`
**What:** Runs an LLM behind [SGLang](https://github.com/sgl-project/sglang)'s TPU backend on a Colab TPU runtime.
**How:** Detects the assigned TPU, installs SGLang's TPU backend, launches the server in the background, polls its health endpoint, talks to it with the OpenAI client, and optionally exposes it via a public URL — with cells to peek at server logs and cleanly stop everything.
**Why:** Validates that a model serves correctly on TPU hardware through an OpenAI-compatible API, without managing infrastructure by hand.
**Output:** A running local (and optionally public) OpenAI-compatible endpoint, plus a sample completion printed from the SGLang server.

### `jev_vs_openai_medical_anomaly.ipynb`
**What:** Benchmarks three methods for flagging anomalies in medical discharge summaries: **Jev** (via OpenRouter's System One API), **GPT structured output**, and **GPT logprobs** (all OpenAI/OpenRouter-backed — no local model).
**How:** Loads discharge summaries from a CSV, asks the same shared question set of all three methods with matched concurrency and per-call timing, then compares them on a scoreboard (accuracy, ROC-AUC, Brier score, precision/recall/F1), plots (latency, calibration, agreement), a disagreement breakdown, and a stability check (repeated calls on a few cases).
**Why:** Answers whether a cheaper/self-reported probability method (structured JSON output) is as reliable as a measured one (logprobs) for this task, and how both compare to Jev.
**Output:** In-notebook scoreboard table, latency/calibration/agreement plots, and a list of cases where the methods disagree.

### `laya_vs_jev_discharge_anomaly.ipynb`
**What:** Benchmarks **Laya** (a local GPU classifier) against **Jev** (via OpenRouter) for detecting anomalies in discharge summaries.
**How:** Loads a discharge-summary CSV, chunks text for Laya (with configurable chunk size/overlap and checkpoint choice), runs both methods on every case with per-call timing, then produces a scoreboard, latency/calibration/agreement plots, a disagreement analysis, a confidence-gating pass (route low-confidence cases differently), and a stability check.
**Why:** Evaluates whether a local, self-hosted model (Laya) is a viable/cheaper substitute for a hosted model (Jev) on this classification task.
**Output:** `laya_vs_jev_*_results.csv`, `*_scoreboard.csv` saved to disk (and offered as a Colab download), plus in-notebook plots and disagreement/confidence-gating tables.

### `laya_vs_jev_vs_gpt_discharge_anomaly.ipynb`
**What:** Three-way extension of the above — **Laya** (local GPU) vs **Jev** (OpenRouter) vs **GPT-4o-mini**, the latter evaluated both via structured JSON output and via logprobs.
**How:** Same pipeline as `laya_vs_jev_discharge_anomaly.ipynb` (shared question design, CSV loading, shared answer parsing) extended with two GPT methods, run across all methods per case, then scored, plotted (latency, calibration, pairwise agreement), diffed for disagreements, confidence-gated, and stability-tested.
**Why:** A fuller comparison to see how a local model, a hosted specialist model, and a general-purpose frontier model trade off on accuracy, calibration, latency, and cost for medical anomaly detection.
**Output:** `laya_vs_jev_vs_gpt_results.csv`, `_scoreboard.csv`, and `_side_by_side.csv` saved to disk (and downloadable), plus in-notebook scoreboard, plots, and disagreement/confidence-gating analysis.

### `zeek_vllm_qwen_langgraph_nsm.ipynb`
**What:** A local AI network-security-monitoring pipeline: captures traffic, parses it with [Zeek](https://zeek.org), and has a [LangGraph](https://github.com/langchain-ai/langgraph) agent (backed by a self-hosted quantized Qwen via vLLM) triage and report on findings.
**How:** Installs Zeek + vLLM (Qwen2.5-7B-Instruct, AWQ/GPTQ/bnb/fp16 selectable), captures traffic (live interface, upload, or URL) into a pcap, runs Zeek offline to produce JSON logs, does deterministic pre-analysis/stats on those logs, defines a Pydantic schema for findings (severity/category/etc.), queries vLLM for structured output against that schema, and wires it all into a LangGraph pipeline with retries and checkpointing; a final cell repeats capture→analyze in rounds for continuous monitoring.
**Why:** Demonstrates an end-to-end, fully local/self-hosted alternative to cloud security-analysis tools — no traffic or logs leave the Colab runtime.
**Output:** `network_report.json` (structured, schema-validated findings per run) plus per-round reports from the continuous-monitoring loop.

### `squash_commit_history.ipynb`
**What:** Squashes a Hugging Face Hub model repo's entire commit history into a single commit.
**How:** Authenticates with `HF_TOKEN` from Colab secrets, calls `HfApi.super_squash_history` on a given `REPO_ID`/branch, then lists the repo's commits to confirm the result.
**Why:** Cleans up bloated repo history (e.g. after many incremental pushes) without needing to clone/rewrite history locally.
**Output:** Confirmation message and the post-squash commit list for the target Hugging Face repo.
