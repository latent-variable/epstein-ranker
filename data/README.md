# Data Directory

This directory houses the multi-modal corpora, legacy datasets, and execution workspaces.

Current snapshot status: DOJ FTA VOL00001-VOL00012 has already been processed into ranked outputs (`source/contrib/fta/`). `data/new_data/` remains the raw input store for reprocessing or future source updates.

## Structure & Organization

### 1. New Data (`new_data/`)
The primary holding area for the newly released Epstein volumes.
- **NATIVES (`new_data/VOL0000X/NATIVES/`)**: Contains the raw original files (e.g. `.pdf`, `.mp4`, `.m4a`, `.xlsx`, `.csv`). This is the target directory for the standard processing pipelines (`run_ranker.sh`, `run_av_ranker.sh`).
- **OCR (`new_data/OCR/VOL0000X/DATA/`)**: Contains the Optical Character Recognition (OCR) text extracts derived from the raw native documents. These are consolidated text files (e.g., `VOL00008.txt`) representing the machine-readable contents of that volume.

### 2. Legacy Epstein 20K Data
- **Location**: `EPS_FILES_20K_NOV2026.csv`
- **Purpose**: The older dataset containing metadata from ~20,000 previously released files from `tensonaut/EPSTEIN_FILES_20K`. Used for baseline comparisons or legacy indexing. 
- **Note**: It is ~100MB and is deliberately git-ignored. You must download it manually and place it in this folder.

### 3. Workspaces (`workspaces/`)
Isolated sandboxes for independent corpora or specific trial runs (e.g., `standardworks_epstein_files_vol00008` or `trial_tony_vol00008`). Workspaces allow for running tests, trying different prompt configurations, or processing subsets of data without overwriting the main index or output directories.

---

### Additional Notes

You can also point `gpt_ranker.py --input` at a directory tree of `.txt` files (for example, `data/new_data`).

For independent corpora, prefer workspace mode:
`--dataset-workspace-root data/workspaces --dataset-tag <name>`

## Moving `new_data` to External Storage (Symlink-Safe)

If you want to free internal disk space without breaking existing scripts/agents that expect `data/new_data`, use the relocation helper:

```bash
# From repository root (source/)
scripts/relocate_new_data.sh --external-root /Volumes/T7/Epstine_data --apply
```

What it does:
- Copies `data/new_data` to `/Volumes/T7/Epstine_data/new_data` via `rsync`
- Verifies source/destination match
- Replaces local `data/new_data` with a symlink to the external path
- Keeps a local backup (`data/.new_data_internal_backup_<timestamp>`) by default

Optional cleanup after validation:

```bash
scripts/relocate_new_data.sh --external-root /Volumes/T7/Epstine_data --apply --purge-internal-backup
```

If the copy is too slow because of millions of small OCR text files, move raw volume folders first and skip OCR:

```bash
scripts/relocate_new_data.sh --external-root /Volumes/T7/Epstine_data --apply --exclude-ocr --skip-verify
```

You can migrate OCR later in a separate pass when convenient.
