#!/bin/bash

# Script: run_python_with_seeds.sh
# Description: Runs python eval.py with seeds 42-46 and concatenates results into a single JSON array

config_file="config/wikimusa.yaml"
# if argument is provided, use it as the config file
if [ $# -gt 0 ]; then
    config_file="$1"
fi

# Output file
OUTPUT_FILE="results_combined_${config_file##*/}.json" >> "$OUTPUT_FILE"

# Start with an empty array
echo "[" > "$OUTPUT_FILE"

# Loop through seeds from 42 to 46
first=true
for seed in {42..46}; do
    echo "Running with seed $seed..."
    
     # Run the Python command and capture output
    result=$(python code/eval.py --config "$config_file" --seed $seed 2>&1)
    
    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "Error running with seed $seed. Output: $result"
        continue
    fi
    
    # Check if output is valid JSON
    if ! echo "$result" | python -c "import json, sys; json.load(sys.stdin)" >/dev/null 2>&1; then
        echo "Warning: Output for seed $seed is not valid JSON: $result"
        continue
    fi
    
    # Add comma if not the first element
    if [ "$first" = true ]; then
        first=false
    else
        echo "," >> "$OUTPUT_FILE"
    fi
    
    # Append the JSON result
    echo "$result" >> "$OUTPUT_FILE"
done

# Close the array
echo "]" >> "$OUTPUT_FILE"

echo "All results have been combined into $OUTPUT_FILE"
