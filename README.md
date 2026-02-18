# Epstein Ranker

LLM-powered tooling for triaging Epstein-related document corpora.
This project:

1. Streams document corpora through locally hosted open-source models (default vision run: `qwen/qwen3-vl-30b` via **LM Studio**) to produce ranked, structured leads.
2. Ships a dashboard (`viewer/`) so investigators can filter, chart, and inspect every scored document (including the full source text) offline.

The entire workflow operates on a single MacBook Pro (M3 Max, 128 GB RAM). With an average draw of ~100 W, a 60-hour pass consumes ≈6 kWh (~$1.50 at SoCal off-peak rates) with zero cloud/API spend.

---

## Current Progress

Status as of **February 16, 2026**:

- **Primary active pipeline:** DOJ File Transparency Act corpus (`data/new_data/VOL00001...`) in image/PDF mode.
- **Local volumes currently available:** `VOL00001` to `VOL00008`, `VOL00010`, `VOL00011`, and `VOL00012`.
- **Still pending locally:** `VOL00009`.
- **Isolation policy:** FTA runs are written to `data/workspaces/standardworks_epstein_files_volXXXXX/` so they do not mix with oversight outputs.
- **Legacy oversight data:** existing House Oversight chunks remain in `contrib/` + `data/chunks.json`.

### Volume verification snapshot (raw package integrity)

Using the bundled load-file indexes (`DATA/VOLxxxxx.DAT` and `DATA/VOLxxxxx.OPT`) as the source of truth:

- `VOL00003`: `67` documents, `1,847` pages, `67` PDF files present (matches local package metadata).
- `VOL00004`: `152` documents, `2,704` pages, `152` PDF files present (matches local package metadata).
- `VOL00005`: `120` documents, `120` pages, `120` PDF files present (matches local package metadata).

Notes:
- The StandardWorks profile currently lists +1 document/page for volumes 3-5 versus the local package indexes.
- Processing status snapshot (chunk outputs): `VOL00003` complete (`67/67`), `VOL00004` has one row pending (`151/152`, missing `IMAGES/0001/EFTA00005932.pdf`), `VOL00005` is in progress.

---

## Screenshots

| Table View | Insights & Charts |
| ---------- | ----------------- |
| ![Table view](imgs/table.png) | ![Insights + charts](imgs/graphs.png) |

| Methodology Explainer |
| --------------------- |
| ![Methodology explainer](imgs/info.png) |

---

## Data source & provenance

Primary corpus (active):

- Raw PDF source repo (used for local volume downloads): <https://github.com/yung-megafone/Epstein-Files>
- StandardWorks File Transparency Act index (text/index reference): <https://standardworks.ai/epstein-files>
- Local data root: `data/new_data/VOL00001...`
- Dataset profile metadata: `data/dataset_profiles/standardworks_epstein_files.json`
- DOJ source links are generated using `--justice-files-base-url` and stored per record (`source_pdf_url`).

Legacy corpus (still supported):

- Hugging Face OCR dataset by **tensonaut**: <https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K>
- House Oversight release reference (Nov 12, 2025): [Oversight Committee Releases Additional Epstein Estate Documents](https://oversight.house.gov/release/oversight-committee-releases-additional-epstein-estate-documents/)

Both corpora include sensitive material (abuse, trafficking, violence, unverified allegations). Use accordingly and validate claims before publication.

---

## Requirements

- Python 3.9+
- `requests`
- LM Studio (or another local gateway) serving your selected model locally (for vision runs: `qwen/qwen3-vl-30b` via OpenAI-compatible `http://localhost:5555/v1`)
- Optional hosted provider: OpenRouter (`https://openrouter.ai/api/v1`) with API key
- For the active FTA workflow: downloaded volume folders under `data/new_data/` (for example `VOL00001`, `VOL00002`, ...) from the repo above.
- Optional legacy text workflow: `data/EPS_FILES_20K_NOV2026.csv` from the Hugging Face link above.

Install Python deps (only `requests` is needed):

```bash
python -m pip install -r requirements.txt  # or just: python -m pip install requests
```

---

## Running the ranker

Recommended (FTA image/PDF workflow): use the helper script.

```bash
./run_ranker.sh --volumes 1
./run_ranker.sh --provider openrouter --openrouter-api-key sk-or-... --volumes 1 --parallel 2
./run_ranker.sh --volumes 1,2,6-7 --parallel 4 --parallel-scheduling batch
./run_ranker.sh --provider openrouter --volumes 3 --image-max-pages 8 --pdf-pages-per-image 4
./run_ranker.sh --volumes 1-12 --dry-run
./run_ranker.sh --volumes all -- --reasoning-effort low
```

OpenRouter key file (recommended):

1. Create `/Users/linovaldovinos/Documents/LatentPlayground/EpstineFileRanker/EpsteinFileRanker-deploy/source/.env.openrouter`
2. Add:

```bash
OPENROUTER_API_KEY='sk-or-...'
OPENROUTER_REFERER='https://epsteingate.org'
OPENROUTER_TITLE='Epstein File Ranker'
```

`run_ranker.sh` auto-loads `.env.openrouter` (or `OPENROUTER_ENV_FILE=/custom/path`), and this file is git-ignored by default.

For `--provider openrouter`, `run_ranker.sh` now auto-applies token pricing defaults for `qwen/qwen3-vl-30b-a3b-instruct`:
- input: `$0.13 / 1M`
- output: `$0.52 / 1M`

Override at runtime with:
`--input-price-per-1m`, `--output-price-per-1m`, `--cache-read-price-per-1m`, `--cache-write-price-per-1m`.

`run_ranker.sh` automatically:

- Resolves selected volumes (`1`, `1,2,6-7`, or `all`).
- Uses workspace isolation per volume (`standardworks_epstein_files_volXXXXX`).
- Writes Git-trackable chunk outputs to `contrib/fta/VOLXXXXX/` by default.
- Rebuilds a global FTA manifest at `contrib/fta/chunks.json` after each run (used by the DOJ dataset view).
- Uses `--resume` by default.
- Skips missing volumes by default (use `--strict-missing` to fail instead).

Direct CLI example (single volume, no wrapper):

```bash
python gpt_ranker.py \
  --input data/new_data/VOL00001 \
  --input-glob "*.pdf" \
  --processing-mode image \
  --dataset-workspace-root data/workspaces \
  --dataset-tag standardworks_epstein_files_vol00001 \
  --endpoint http://localhost:5555/v1 \
  --api-format openai \
  --model qwen/qwen3-vl-30b \
  --max-parallel-requests 4 \
  --image-max-pages 1 \
  --image-render-dpi 180 \
  --resume
```

## Code layout

- `gpt_ranker.py`: orchestration pipeline / entrypoint.
- `ranker/cli.py`: CLI parsing + config/workspace default resolution.
- `ranker/model_client.py`: endpoint fallback, retries, text/vision request building.
- `ranker/constants.py`: canonical maps and shared constants.

By default, direct CLI runs write **1,000-row chunks** to `contrib/` and update `data/chunks.json`.  
When using `run_ranker.sh`, checkpoints/metadata stay in `data/workspaces/<dataset-tag>/` while chunk outputs are written to `contrib/fta/VOLXXXXX/` for Git tracking.
Set `--chunk-size 0` if you really want a single CSV/JSONL output (not recommended for sharing).

Notable flags:

- `--prompt-file`: specify a custom system prompt file (defaults to `prompts/default_system_prompt.txt`). See `prompts/README.md` for details on creating custom prompts.
- `--system-prompt`: provide an inline system prompt string (overrides `--prompt-file`).
- `--input`: accepts the legacy CSV (`filename` + `text`) or a directory tree of `.txt`, `.pdf`, and image files.
- `--input-glob`: glob pattern used when `--input` is a directory (default `*.txt`, recursive).
- `--processing-mode`: `auto`, `text`, or `image`. Use `image` for PDF/image-first workflows.
- `--dataset-workspace-root` + `--dataset-tag`: isolate this corpus into its own workspace (`results/`, `chunks/`, `state/`, `metadata/`) so outputs do not mix with the oversight dataset.
- `--dataset-source-label`, `--dataset-source-url`, `--dataset-metadata-file`: attach provenance details; if `--run-metadata-file` is set (or auto-set in workspace mode), a run metadata JSON is written.
- `--resume`: skips rows already present in the JSONL/checkpoint so you can stop/restart long runs.
- `--checkpoint data/.epstein_checkpoint`: stores processed filenames to guard against duplication.
- `--reasoning-effort low/high`: trade accuracy for speed if your model exposes the reasoning control knob.
- `--max-parallel-requests`: number of concurrent requests to LM Studio (default `4`).
- `--parallel-scheduling`: `batch` (submit N, wait all N, then next N) or `window` (continuous refill). `run_ranker.sh` defaults to `batch`.
- `--image-prefetch`: queue extra image render/prep tasks beyond active requests (used with `window` scheduling).
- `--max-output-tokens`: hard cap for completion length per request (useful to stop runaway outputs).
- `--api-key`, `--http-referer`, `--x-title`: auth and optional headers for hosted OpenAI-compatible endpoints (including OpenRouter).
- `--api-format`: `auto` (default), `openai`, or `chat`. Vision/image mode requires `openai` (or `auto`, which resolves to OpenAI format).
- `--image-max-pages`, `--image-render-dpi`, `--image-detail`: configure PDF rendering + vision detail level for multimodal inference.
- `--pdf-pages-per-image`: pack multiple PDF pages into one tiled image block before sending to the model (for example `4` = 2x2 collage).
- `--pdf-part-pages`: split long PDFs into independent part records (`source_id`, `document_part`, `part_index`, `part_total`, page range fields) so each page window can be processed and reviewed separately.
- `--max-retries`, `--retry-backoff`: retry transient endpoint failures with exponential backoff.
- `--skip-low-quality` / `--no-skip-low-quality`: enable/disable pre-LLM filtering for empty/short/noisy OCR rows.
- `--min-text-chars`, `--min-text-words`, `--min-alpha-ratio`, `--min-unique-word-ratio`, `--max-short-token-ratio`, `--min-avg-word-length`, `--min-long-word-count`, `--max-repeated-char-run`: tune skip thresholds (helps skip OCR gibberish with mostly short tokens).
- `--justice-files-base-url`: base URL used to derive DOJ PDF links (stored as `source_pdf_url` and shown in the viewer).
- `--source-files-base-url`: optional hosted base URL for local files when DOJ URL derivation is unavailable.
- `--include-action-items`: opt-in if you want the LLM to list action items (off by default for brevity).
- `--timeout`: HTTP request timeout in seconds (default: 600 = 10 minutes). Increase for very large documents (100K+ tokens).
- `--max-rows N`: smoke-test on a small subset.
- `--list-models`: query your endpoint for available model IDs.
- `--rebuild-manifest`: scan `contrib/` for chunk files and rebuild `data/chunks.json` (useful if the manifest gets out of sync).
- `--start-row`, `--end-row`: process only a slice of the dataset (ideal for collaborative chunking).
- `--chunk-size`, `--chunk-dir`, `--chunk-manifest`: control chunk splitting, where chunk files live, and where the manifest is written.
- `--overwrite-output`: explicitly allow truncating existing files (default is to refuse unless `--resume` or unique paths are used).
- `--power-watts`, `--electric-rate`, `--run-hours`: plug in your local power draw/cost to estimate total electricity usage (also configurable via the TOML file).
- `--input-price-per-1m`, `--output-price-per-1m`, `--cache-read-price-per-1m`, `--cache-write-price-per-1m`: track hosted API model costs from reported token usage.
- `run_ranker.sh --workspace-chunks`: keep chunks inside ignored workspace paths instead of `contrib/fta/` if you do not want Git-tracked outputs.

Pause/resume behavior:

- Press `Ctrl+C` to pause gracefully.
- The ranker flushes completed rows, preserves the checkpoint, and exits with resume instructions.
- Restart with `--resume` to continue from where it left off.

Outputs:

- `contrib/fta/VOLXXXXX/epstein_ranked_<start>_<end>.jsonl` + `contrib/fta/VOLXXXXX/chunks.json` – Default per-volume, Git-trackable outputs when using `run_ranker.sh`.
- `contrib/fta/chunks.json` – Global manifest auto-built by `run_ranker.sh` so the DOJ viewer can load all tracked volume chunks.
- `contrib/epstein_ranked_<start>_<end>.jsonl` + `data/chunks.json` – Default chunk outputs/manifest for direct CLI runs without workspace mode.
- `data/epstein_ranked.csv/jsonl` – Only produced if you disable chunking via `--chunk-size 0`.
- `data/workspaces/<dataset-tag>/metadata/run_metadata.json` – Optional run provenance sidecar (enabled automatically in workspace mode).
- Output rows now include API usage/cost fields (`api_prompt_tokens`, `api_completion_tokens`, `api_total_tokens`, cache token fields, `api_cost_usd`) when the provider returns usage data.

### Independent Corpus Workspace

For large non-oversight corpora (like `data/new_data`), run in isolated workspace mode:

```bash
python gpt_ranker.py \
  --input data/new_data/VOL00001 \
  --input-glob "*.pdf" \
  --processing-mode image \
  --dataset-workspace-root data/workspaces \
  --dataset-tag standardworks_epstein_files_vol00001 \
  --dataset-source-label "Epstein-Files GitHub (raw PDFs)" \
  --dataset-source-url "https://github.com/yung-megafone/Epstein-Files" \
  --dataset-metadata-file data/dataset_profiles/standardworks_epstein_files.json \
  --max-parallel-requests 4 \
  --resume
```

This keeps outputs/checkpoints/chunks independent from the original `contrib/` + `data/chunks.json` workflow.
`data/workspaces/` is git-ignored by default.

### Chunk Manifest

The ranker automatically updates `data/chunks.json` after each chunk is completed. This manifest tells the viewer which chunk files exist and what row ranges they cover.

**If the manifest gets out of sync** (e.g., due to interrupted runs or manual file moves), you can rebuild it:

```bash
python gpt_ranker.py --rebuild-manifest
```

This scans `contrib/` for all chunk files and regenerates the manifest automatically.

---

## Customizing the system prompt

The ranker uses a system prompt to instruct the model on how to analyze and score documents. You can customize this prompt to fit your specific needs:

### Using a custom prompt file

1. Create your own prompt file in the `prompts/` directory (e.g., `prompts/my_custom_prompt.txt`)
2. Run the ranker with `--prompt-file`:

```bash
python gpt_ranker.py --prompt-file prompts/my_custom_prompt.txt --config ranker_config.toml
```

Or set it in your config file:

```toml
prompt_file = "prompts/my_custom_prompt.txt"
```

See `prompts/README.md` for detailed guidance on creating custom prompts, and check out `prompts/example_strict_scoring.txt` for an example of a stricter scoring methodology.

### Prompt priority

The ranker loads prompts in this order of priority:
1. `--system-prompt` (inline string argument)
2. `--prompt-file` or `prompt_file` in config
3. `prompts/default_system_prompt.txt` (required default prompt file)

### Tracking prompt usage

The prompt source is automatically included in the output metadata for each document, so you can always see which prompt was used for analysis.

---

## Scoring methodology (LLM prompt)

The default prompt instructs the model with the following criteria (excerpt):

```
You analyze primary documents related to court and investigative filings.
Focus on whether the passage offers potential leads—even if unverified—that connect influential actors ... to controversial actions, financial flows, or possible misconduct.
Score each passage on:
  1. Investigative usefulness
  2. Controversy / sensitivity
  3. Novelty
  4. Power linkage
Assign an importance_score from 0 (no meaningful lead) to 100 (blockbuster lead linking powerful actors to fresh controversy). Use the scale consistently:
  • 0–10  : noise, duplicates, previously published facts, or gossip with no actors.
  • 10–30 : low-value context; speculative or weak leads lacking specifics.
  • 30–50 : moderate leads with partial details or missing novelty.
  • 50–70 : strong leads with actionable info or notable controversy.
  • 70–85 : high-impact, new revelations tying powerful actors to clear misconduct.
  • 85–100: blockbuster revelations demanding immediate follow-up.
Reserve 70+ for claims that, if true, would represent major revelations or next-step investigations.
Return strict JSON with fields: headline, importance_score, reason, key_insights, tags, power_mentions, agency_involvement, lead_types.
```

Rows ≥70 typically surface multi-factor leads (named actors + money trail + novelty). Anything below ~30 is often speculation or previously reported context.

---

## Interactive viewer

Serve the dashboard to explore results, filter, and inspect the full source text of each document:

```bash
./viewer.sh 9000
# or manually:
cd viewer && python -m http.server 9000
```

Open <http://localhost:9000>. Features:

- Automatically loads any chunk listed in `data/chunks.json` (falls back to `data/epstein_ranked.jsonl` if no chunks exist).
- AG Grid table sorted by importance score (click a row to expand the detail drawer and read the entire document text).
- DOJ dataset rows include `Volume` and `Part` fields; split PDF parts are tracked with stable `source_id`s so filtering/selection stays consistent.
- Filters for score threshold, lead types, power mentions, ad hoc search, and row limits.
- Charts showing lead-type distribution, score histogram, top power mentions, and top agencies.
- Methodology accordion describing the scoring criteria, prompt, and compute footprint.

`viewer/app.js` reads `data/chunks.json` by default. If no manifest exists, it automatically scans for files named `contrib/epstein_ranked_*.jsonl` before falling back to `data/epstein_ranked.jsonl`.

---

## Collaborative ranking workflow

Want to help process more of the corpus? Fork the repo, claim a range of rows, and submit your results:

1. **Pick a chunk** – e.g., rows `00001–01000`, `01001–02000`, etc. Use whatever increments work. Announce the chunk (issue/Discord) so others don’t duplicate effort.
2. **Run the ranker on that slice** using the new range flags:

   ```bash
   python gpt_ranker.py \
     --config ranker_config.toml \
     --start-row 1001 \
     --end-row 2000 \
     --chunk-dir contrib \
     --chunk-manifest data/chunks.json \
     --known-json data/epstein_ranked.jsonl \
     --resume
   ```

   This only processes documents in that range, emits `contrib/epstein_ranked_<range>.jsonl`, and updates the manifest.  
   `--known-json` makes the script aware of previously merged results (so duplicates are skipped automatically). Combine with `--resume` if you need to pause and continue later.

3. **Export your outputs** – each run writes the chunk JSONL straight into `contrib/`. Keep the naming pattern `contrib/epstein_ranked_<start>_<end>.jsonl`.

4. **Submit a PR** with your chunk (the JSONL + updated `data/chunks.json`). We’ll merge the contributions into the global dataset and credit collaborators in the README.

Guidelines:

- Do **not** commit the original 100 MB source CSV; each contributor should download it separately.
- Keep the JSONL chunks intact (no reformatting) so we can merge them programmatically.
- If you discover inconsistencies or interesting leads, open an issue to coordinate follow-up analysis.
- Pull the latest `data/chunks.json` (and any merged JSONL files) before starting; pass the merged JSON via `--known-json` so you never duplicate work.

---

## Ethical considerations & intended use

- The corpus contains sensitive content (sexual abuse, trafficking, violence, unverified allegations). Use with care.
- Documents are part of the public record but may still be subject to copyright/privacy restrictions; verify before sharing or redistributing.
- Recommended use cases: investigative triage, exploratory data analysis, RAG/IR experiments, or academic review.
- This project does **not** assert any claims about the veracity of individual documents—scores merely prioritize leads for deeper human review.

---

## Acknowledgements

- **tensonaut** for compiling the OCR corpus and publishing it to Hugging Face.
- U.S. House Committee on Oversight and Government Reform for releasing the source documents.
- The LM Studio community & `r/LocalLLaMA` for pushing local LLM workflows forward.

---

## License

Released under the [MIT License](LICENSE). Please retain attribution to this project, the `tensonaut` dataset, and the U.S. House Oversight Committee release when building derivative tools or analyses.
