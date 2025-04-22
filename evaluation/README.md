# Evaluation Scripts for Coherence, Fluency, and Relevance

This directory contains evaluation scripts for assessing the **coherence**, **fluency**, and **relevance** of generated text.

---

## File Overview

Each metric has its own evaluation script (`<metric>.py`) and a SLURM job submission file (`run_<metric>.slurm`). For coherence and fluency, metric implementations are defined in separate files (`<metric>_metric.py`). For relevance, the evaluation relies directly on the built-in metrics provided by DeepEval.

---

## Usage

To evaluate a specific metric, submit the corresponding SLURM job:

```bash
sbatch run_<metric>.slurm
