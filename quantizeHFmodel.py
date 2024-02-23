import os
import sys
from huggingface_hub import HfApi, snapshot_download

def download_model(model_id):
    """Downloads the model from Hugging Face and returns the local directory path."""
    local_dir = snapshot_download(repo_id=model_id)
    return local_dir

def quantize_model(model_file, quant_methods):
    """Generates quantized model versions. Placeholder for your actual quantization process."""
    quantized_files = []
    for method in quant_methods:
        output_file = f"{model_file}.{method}.gguf"
        # Your quantization command here, e.g., os.system(f"./quantize {model_file} {output_file} {method}")
        print(f"Quantizing {model_file} to {output_file} using {method}...")
        quantized_files.append(output_file)
    return quantized_files

def upload_quantized_models(model_id, quantized_files):
    """Uploads the quantized models to Hugging Face."""
    api = HfApi()
    repo_id = f"{model_id}-GGUF"
    api.create_repo(repo_id, exist_ok=True, repo_type="model", private=False)  # Set private=True if you want the repo to be private

    for file in quantized_files:
        print(f"Uploading {file} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=os.path.basename(file),
            repo_id=repo_id,
        )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]
    model_name = model_id.split("/")[-1]  # Extract model name from model_id
    local_dir = download_model(model_id)
    model_file = os.path.join(local_dir, f"{model_name}.gguf")  # Update this path according to where the downloaded model is saved

    # Define your quantization methods here
    quant_methods = ["q4_k_m","q5_k_m", "q8_0"]
    quantized_files = quantize_model(model_file, quant_methods)
    
    # Upload quantized models
    upload_quantized_models(model_id, quantized_files)
    print("All models have been uploaded.")
