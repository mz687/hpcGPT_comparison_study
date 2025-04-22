# Inference Scripts

This directory contains scripts used for generating answers to question in the evaluation dataset for the purpose of evaluation.

## Directory Structure

Each subfolder corresponds to a specific method or experimental setup. Within each subfolder, you will find:

- `eval_chatbot.py`: The inference script used to evaluate the method.
- `*.slurm`: A SLURM job submission script for running the evaluation on a cluster environment.