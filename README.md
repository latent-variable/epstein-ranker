# Epstein Ranker

LLM-powered tooling for triaging Epstein-related document corpora. Processes PDFs, office documents, and audio/video files through vision-language models, scoring each for investigative significance. Ships a dashboard for filtering, charting, and inspecting scored documents.

**Default model:** [`qwen/qwen3-vl-30b-a3b-thinking`](https://openrouter.ai/qwen/qwen3-vl-30b-a3b-thinking) via OpenRouter (free, hosted by Alibaba). No API costs, no local GPU required.

---

## Quick Start

1. **Install deps:**

```bash
pip install -r requirements.txt  # just `requests`
```

2. **Set up your OpenRouter key** (free account at [openrouter.ai](https://openrouter.ai)):

```bash
cp .env.template.openrouter .env.openrouter
# Edit .env.openrouter with your key:
#   OPENROUTER_API_KEY='sk-or-...'
```

3. **Run:**

```bash
./run_ranker.sh --volumes 1           # process volume 1
./run_ranker.sh --volumes 1,2,6-8     # multiple volumes
./run_ranker.sh --volumes all         # all available volumes
./run_ranker.sh --volumes 1 --dry-run # preview without processing
```

That's it. The script auto-loads `.env.openrouter`, connects to OpenRouter, and writes Git-trackable results to `contrib/fta/`.

---

## Processing Pipelines

### PDF Ranker (`run_ranker.sh` / `run_hybrid_volume.sh`)

The primary pipeline. Renders PDFs as images, sends them to the VLM for analysis.

```bash
# Cloud only (default)
./run_ranker.sh --volumes 11

# Hybrid: cloud + local fallback
./run_hybrid_volume.sh --volume 11

# Local only
./run_ranker.sh --volumes 11 --provider local --endpoint http://localhost:5555/v1
```

Key options: `--parallel N`, `--max-rows N` (smoke test), `--start-pdf / --end-pdf` (split work), `--retry-failed-local`.

### Office Ranker (`run_office_ranker.sh`)

Processes Excel, Word, PowerPoint, and other office docs via LibreOffice PDF conversion + the same VLM pipeline.

```bash
./run_office_ranker.sh --volume 11
```

### AV Ranker (`run_av_ranker.sh`)

Processes video and audio files. Extracts frames into 2×2 grid composites (4× token savings), transcribes audio locally via Whisper (`small` model).

```bash
./run_av_ranker.sh --volume 11                                    # cloud
./run_av_ranker.sh --volume 11 --endpoint http://localhost:5555/v1  # local
./run_av_ranker.sh --volume 11 --max-files 3                      # smoke test
./run_av_ranker.sh --volume 11 --no-transcription                 # frames only
```

Key options: `--fps N`, `--max-frames N`, `--grid-cols/rows N`, `--no-grid`, `--whisper-model SIZE`.

---

## Local Processing

All pipelines support local inference via LM Studio or any OpenAI-compatible server:

```bash
# Serve qwen/qwen3-vl-30b-a3b-thinking in LM Studio on port 5555, then:
./run_ranker.sh --volumes 1 --provider local --endpoint http://localhost:5555/v1
```

---

## Configuration

### OpenRouter Environment (`.env.openrouter`)

```bash
OPENROUTER_API_KEY='sk-or-...'
OPENROUTER_REFERER='https://epsteingate.org'
OPENROUTER_TITLE='Epstein File Ranker'
OPENROUTER_PROVIDER='alibaba'
OPENROUTER_NO_FALLBACKS='1'
```

### TOML Config (`ranker_config.toml`)

Copy `ranker_config.example.toml` and customize. CLI flags override TOML values.

### Custom Prompts

```bash
python gpt_ranker.py --prompt-file prompts/my_custom_prompt.txt
```

See `prompts/README.md` for details.

---

## Data Sources

- **DOJ FTA corpus (primary):** [Epstein-Files GitHub](https://github.com/yung-megafone/Epstein-Files) — raw PDFs under `data/new_data/VOL00001...`
- **StandardWorks index:** [standardworks.ai/epstein-files](https://standardworks.ai/epstein-files)
- **Legacy OCR dataset:** [tensonaut/EPSTEIN_FILES_20K](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K)

---

## Viewer

```bash
./viewer.sh 9000  # or: cd viewer && python -m http.server 9000
```

Open http://localhost:9000 — AG Grid table with score filtering, charts, power mention analysis, and full document text inspection.

---

## Code Layout

| File | Purpose |
|---|---|
| `gpt_ranker.py` | Main orchestration pipeline |
| `av_ranker.py` | Audio/video processing pipeline |
| `office_ranker.py` | Office document processing pipeline |
| `ranker/cli.py` | CLI parsing + config resolution |
| `ranker/model_client.py` | API client, retries, vision request building |
| `ranker/constants.py` | Canonical maps and shared constants |
| `run_ranker.sh` | PDF processing wrapper (cloud by default) |
| `run_hybrid_volume.sh` | Hybrid cloud+local processing wrapper |
| `run_av_ranker.sh` | AV processing wrapper |
| `run_office_ranker.sh` | Office document processing wrapper |

---

## Scoring

Documents are scored 0–100 based on investigative significance:

| Range | Meaning |
|---|---|
| 0–10 | Noise, duplicates, no actionable info |
| 10–30 | Weak leads, speculative |
| 30–50 | Moderate leads, partial details |
| 50–70 | Strong leads, actionable info |
| 70–85 | High-impact revelations |
| 85–100 | Blockbuster evidence |

---

## Ethics & Intended Use

The corpus contains sensitive content (abuse, trafficking, violence, unverified allegations). Scores prioritize leads for human review — this project does not assert the veracity of any individual document.

---

## License

This project is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (Creative Commons Attribution-ShareAlike 4.0 International). You must give appropriate credit and distribute derivative works under the same license.
