#!/bin/bash

# Check if a directory path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_dir>"
    exit 1
fi

# Directory containing fold_x folders (provided as an argument)
DIR=$1

# Output file to store the consolidated metrics
OUTPUT_FILE=$DIR"/consolidated_metrics.csv"

# Initialize the output file with headers
echo "fold,precision,accuracy,recall,duration" > "$OUTPUT_FILE"

# Loop through each fold_x directory
for FOLD in "$DIR"/fold_*/
do
    # Extract the fold number
    FOLD_NAME=$(basename "$FOLD")

    # Read the metrics from df_metrics.csv
    # Skip the first line (header) and read the rest
    while IFS=, read -r precision accuracy recall duration || [ -n "$precision" ]
    do
        # Ignore the header line of the csv
        if [ "$precision" != "precision" ]; then
            # Append the metrics to the output file
            echo "$FOLD_NAME,$precision,$accuracy,$recall,$duration" >> "$OUTPUT_FILE"
        fi
    done < "$FOLD/df_metrics.csv"
done

echo "Consolidation Complete. Output file: $OUTPUT_FILE"

