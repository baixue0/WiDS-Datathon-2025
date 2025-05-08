#!/bin/bash

python3 -u prepare_data.py
echo "prepare data finished"

# File containing JSON data
json_file="config/SETTINGS.json"

# Extract the value of num_seeds
num_seeds=$(grep -oP '"num_seeds":\s*\K\d+' "$json_file")

# Iterate through seeds based on the num_seeds value
for seed in $(seq 1 $num_seeds)
do
    echo "$seed"
    python3 -u train.py --target_col Sex_F --seed "$seed"
    python3 -u train.py --target_col ADHD_Outcome --seed "$seed"
done

# Model inference
python3 -u predict.py