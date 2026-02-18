# Epstein Ranker Dataset

## Overview
This dataset contains ranked and analyzed documents from the **U.S. House Oversight Epstein Estate** document release. The analysis was performed using a locally hosted `openai/gpt-oss-120b` model to score documents based on their investigative value.

**Original Source:** "20,000 Epstein Files" corpus by [tensonaut](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K).
**Project Site:** [https://epsteingate.org](https://epsteingate.org)

## Methodology
The documents were processed using an LLM to extract structured data and assign an **importance score** (0-100). The scoring criteria focused on:
1.  **Investigative Usefulness:** Does it offer actionable leads?
2.  **Controversy / Sensitivity:** Does it involve misconduct or sensitive topics?
3.  **Novelty:** Is this new information?
4.  **Power Linkage:** Does it connect to influential actors?

## Data Structure
The dataset is provided in JSONL (JSON Lines) format. Each line represents a single document analysis.

### Fields
-   `filename`: The original filename from the House Oversight release.
-   `headline`: A concise summary of the document's content.
-   `importance_score`: Integer (0-100) indicating the document's significance.
    -   **0-10**: Noise, duplicates, gossip.
    -   **10-30**: Low-value context.
    -   **30-50**: Moderate leads.
    -   **50-70**: Strong leads.
    -   **70-85**: High-impact revelations.
    -   **85-100**: Blockbuster leads.
-   `reason`: Explanation for the assigned score.
-   `key_insights`: List of specific takeaways or facts.
-   `tags`: List of relevant keywords.
-   `power_mentions`: List of influential people mentioned.
-   `agency_involvement`: List of government agencies mentioned.
-   `lead_types`: Classification of the type of lead (e.g., "financial flow", "foreign influence").
-   `metadata`: Additional context for traceability.
    -   `source_row_index`: The row number in the original source dataset.
    -   `original_row`: The complete original data object, including:
        -   `filename`: Original filename.
        -   `text`: **Full OCR text content** of the document.
    -   `config`: Configuration used for the LLM analysis.
        -   `model`: The model identifier (e.g., `openai/gpt-oss-120b`).
        -   `temperature`: Model temperature setting.
        -   `reasoning_effort`: Effort level (e.g., `low`).
        -   `power_watts` / `electric_rate`: Energy consumption metrics for the inference.

## Usage
This dataset is intended for:
-   Investigative journalism and triage.
-   Exploratory data analysis.
-   RAG (Retrieval-Augmented Generation) experiments.

## License
This dataset is derived from public records. The analysis and tooling are released under the MIT License.
