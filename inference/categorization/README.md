## Overview

We perform categorization before computing the averge score for each evaluation metric. 

Specifically, we use deepseek-ai/DeepSeek-R1-Distill-Qwen-14B model to classify a ticket into one of these four categories: policy, technical, machine status, and debugging.

- Policy: examples of policy-related questions are "disk quota", "password reset", "restore data", "software license", and  "project allocation".
- technical: "how to install pytorch", "how to use specific software on a machine".
- machine status: real-time machine status. Examples include "slow access to file system", "scheduled maintanence", and "system down".
- debugging: should have attached the error message to the ticket.

## How to run
1. Run [./run_categorization_eval.slurm](./run_categorization_eval.slurm) to categorize the QA-pairs in `eval.json`. 
2. Run [./run_categorization_train.slurm](./run_categorization_train.slurm) to categorize the QA-pairs in `train.json`. 

Depending on the size of the `train.json` and `eval.json`, the number of nodes and time limit in those two SLURM files need to be changed accordingly.


