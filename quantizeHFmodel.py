from huggingface_hub import snapshot_download
import os
import subprocess
import sys

def download_and_quantize_model(model_id):
    model_name = model_id.split('/')[-1]
    base_path = f"{model_name}/{model_name.lower()}"
    
    # Download model with huggingface_hub
    local_dir = f"{model_name}"
    if not os.path.exists(local_dir):
        snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False, revision="main")
    
    # Convert to FP16
    fp16_file = f"{base_path}.f16.gguf"
    if not os.path.isfile(fp16_file):
        subprocess.run(["python", "llama.cpp/convert.py", local_dir, "--outtype", "f16", "--outfile", fp16_file], check=True)
    
    # Quantization methods
    quant_methods = [("q4_k_m", ".q4_k_m.gguf"), ("q5_k_m", ".q5_k_m.gguf"), ("q8_0", ".q8_0.gguf")]
    for method, extension in quant_methods:
        quant_file = f"{base_path}{extension}"
        if not os.path.isfile(quant_file):
            subprocess.run(["./llama.cpp/quantize", fp16_file, quant_file, method], check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_id>")
        sys.exit(1)
    
    model_id = sys.argv[1]
    download_and_quantize_model(model_id)
