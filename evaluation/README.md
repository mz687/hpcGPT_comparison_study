# Evaluation Scripts

This directory contains evaluation scripts for assessing the **coherence**, **fluency**, **relevance**, and **cosine similarity** of generated text.

---

## File Overview

Each of **coherence**, **fluency**, **relevance** has an evaluation script (`<metric>.py`) and a SLURM job submission file (`run_<metric>.slurm`). 
For coherence and fluency, metric implementations are defined in separate files (`<metric>_metric.py`). 
For relevance, the evaluation relies directly on the built-in metrics provided by DeepEval.

By contrast, **cosine similarity** is computed using a standalone script (`cosine_similarity.py`) without the DeepEval framework.

---

## Usage

To evaluate a specific DeepEval-based metric, submit the corresponding SLURM job:

```bash
sbatch run_<metric>.slurm