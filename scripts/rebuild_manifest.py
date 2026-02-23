import json, re, sys, time
from pathlib import Path

# script in source/scripts/rebuild_manifest.py
script_dir = Path(__file__).resolve().parent
source_dir = script_dir.parent
root = source_dir / "contrib/fta"

if not root.exists():
    print(f"Error: {root} does not exist. Please run within the source directory or ensure the data exists.")
    sys.exit(1)

chunks = []
rows_processed = 0
pattern = re.compile(r"epstein_ranked_(\d+)_(\d+)\.jsonl$")

# Process all jsonl files in the tree
for file_path in sorted(root.rglob("epstein_ranked_*.jsonl")):
    match = pattern.match(file_path.name)
    if not match: continue
    start = int(match.group(1))
    end = int(match.group(2))
    
    with file_path.open(encoding="utf-8") as f:
        row_count = sum(1 for _ in f)
        
    # We want the JSON path in the manifest to be relative to the 'source' directory,
    # eg. "contrib/fta/VOL00001/epstein_ranked_...jsonl"
    rel_path = file_path.relative_to(source_dir).as_posix()
    chunks.append({
        "start_row": start,
        "end_row": end,
        "json": rel_path,
        "row_count": row_count
    })
    rows_processed += row_count

chunks.sort(key=lambda x: x["start_row"])

manifest = {
    "metadata": {
        "total_dataset_rows": "unknown",
        "rows_processed": rows_processed,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    },
    "chunks": chunks
}

manifest_path = root / "chunks.json"
with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Rebuilt global chunks.json at {manifest_path.relative_to(source_dir)} with {len(chunks)} entries and {rows_processed} total rows processed.")
