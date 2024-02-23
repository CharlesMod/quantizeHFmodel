#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

echo "Starting the process..."

model_id=$1
echo "Model ID: $model_id"

# Fetching username for the Hugging Face CLI
echo "Fetching Hugging Face username..."
username=$(huggingface-cli whoami)

model_name=$(basename "$model_id")
echo "Model Name: $model_name"

# Downloading the model using download.py
echo "Downloading the model..."
python download.py "$model_id"
local_dir="${model_id//\//-}"
echo "Model downloaded to: $local_dir"

# Define quantization methods
quant_methods=("q4_k_m" "q5_k_m" "q8_0")

# Loop through each quantization method, quantize and then upload
for method in "${quant_methods[@]}"; do
    input_file="${local_dir}/${model_name}.bin"  # Adjust based on actual file naming
    output_file="${local_dir}/${model_name}.${method}.gguf"
    echo "Quantizing to $output_file using method $method..."
    ./llama.cpp/quantize "$input_file" "$output_file" "$method"
    
    if [ -f "$output_file" ]; then
        echo "Quantization successful. Uploading $output_file to ${username}/${model_name}-GGUF..."
        huggingface-cli repo upload "$output_file" --repo="${username}/${model_name}-GGUF"
        echo "$output_file uploaded successfully."
    else
        echo "Failed to quantize or upload $output_file."
    fi
done

echo "All processes completed."
