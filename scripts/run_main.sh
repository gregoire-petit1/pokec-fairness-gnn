#!/bin/bash
set -e
cd ~/pokec-fairness-gnn
mkdir -p logs
echo "Starting main experiment at $(date)" | tee logs/main_experiment.log
.venv/bin/jupyter nbconvert   --to notebook   --execute notebooks/main_experiment.ipynb   --output notebooks/main_experiment_executed.ipynb   --ExecutePreprocessor.timeout=3600   2>&1 | tee -a logs/main_experiment.log
echo "Done at $(date). Exit code: $?" >> logs/main_experiment.log
