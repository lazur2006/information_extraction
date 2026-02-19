# Transition from Rule-Based to Transformer-Based Named-Entity Recognition

Implementation repository for **"Information Extraction Using MultiCoNER II"**.

## Project Information

| Field | Value |
|---|---|
| Course | Advanced NLP and Computer Vision (`DLMAIEAIS01`) |
| Program | Applied Artificial Intelligence |
| Author | Kai Schiffer |
| Matriculation Number | IU14108163 |
| Tutor | Prof. Dr. Tim Schlippe |
| Date | 19 February 2026 |

## Overview

This project explores multiple NER approaches on MultiCoNER-style data:

- Rule-/lookup-oriented baseline ideas
- Neural CRF baseline (word-level)
- Transformer + CRF hybrid
- LoRA-adapted transformer + CRF variants

The main implementation and experiment flow are in:

- `ner.ipynb`

## Repository Structure

```text
.
├── ner.ipynb                          # Main notebook (training + evaluation)
├── aws_dataset_loader.ipynb           # Additional dataset loading notebook
├── data/
│   ├── download.py                    # MultiCoNER-style CoNLL parsing/loader utility
│   └── loader.py                      # PyTorch dataset + encoders + collate fn
├── visualization/
│   └── visualization.py               # Training plot helper
├── final_records/
│   ├── neural_crf/
│   ├── neural_crf_tfm/
│   ├── neural_crf_tfm_lora/
│   └── ...                            # Additional coarse/multilingual variants
├── ner_tags.yaml                      # NER tag definitions
└── multiconer2_taxonomy_plot.png      # Taxonomy illustration
```

## Quick Start

1. Create and activate a Python environment
2. Install dependencies
3. Ensure MultiCoNER data files are available.
4. Run the notebook

## Results and Outputs

- Training metrics are stored as JSONL in `final_records/*/crf_training_metrics.jsonl`.
- Example metric schema:

```json
{"epoch": 0, "loss": 1.02, "f1": 0.24}
```

- Model checkpoints are written by the notebook to `models/` when training is executed.

## Notes

- `multiconer2023/` and `models/` are present but empty in a fresh clone.
- This repository is notebook-centric; most logic is orchestrated from `ner.ipynb`.
