#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1
username=$(huggingface-cli whoami)  # Ensure you're logged in
model_name=$(basename "$model_id")
# Use download.py to download the model
python download.py "$model_id"
local_dir="${model_id//\//-}"

# Define quantization methods
quant_methods=("q4_k_m" "q5_k_m" "q8_0")

# Loop through each quantization method, quantize and then upload
for method in "${quant_methods[@]}"; do
    input_file="${local_dir}/${model_name}.bin"  # Adjust based on actual file naming
    output_file="${local_dir}/${model_name}.${method}.gguf"
    echo "Quantizing to $output_file using method $method..."
    ./llama.cpp/quantize "$input_file" "$output_file" "$method"
    
    if [ -f "$output_file" ]; then
        echo "Uploading $output_file to ${username}/${model_name}-GGUF..."
        # Assuming huggingface-cli tool has functionality to upload files directly
        huggingface-cli repo upload "$output_file" --repo="${username}/${model_name}-GGUF"
        echo "$output_file uploaded."
    else
        echo "Failed to quantize or upload $output_file."
    fi
done
