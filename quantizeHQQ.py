import sys
import os
import torch
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend

def quantize_and_save_model(model_id, output_dir, cache_path='.', quant_level=4):
    """
    Quantizes the given model to the specified quantization level and saves it in the specified output directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the model and tokenizer
    model = HQQModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)

    # Define quantization parameters
    if quant_level == 2:
        attn_params = BaseQuantizeConfig(nbits=2, group_size=64, offload_meta=True)
        experts_params = BaseQuantizeConfig(nbits=2, group_size=16, offload_meta=True)
    elif quant_level == 3:
        attn_params = BaseQuantizeConfig(nbits=3, group_size=64, offload_meta=True)
        experts_params = BaseQuantizeConfig(nbits=2, group_size=16, offload_meta=True)
    elif quant_level == 4:
        attn_params = BaseQuantizeConfig(nbits=4, group_size=64, offload_meta=True)
        experts_params = BaseQuantizeConfig(nbits=2, group_size=16, offload_meta=True)
    elif quant_level == 8:
        attn_params = BaseQuantizeConfig(nbits=8, group_size=64, offload_meta=True)
        experts_params = BaseQuantizeConfig(nbits=4, group_size=16, offload_meta=True)
    else:
        raise ValueError("Unsupported quantization level")

    # Apply quantization configuration
    quant_config = {
        'self_attn.q_proj': attn_params,
        'self_attn.k_proj': attn_params,
        'self_attn.v_proj': attn_params,
        'self_attn.o_proj': attn_params,
        'block_sparse_moe.experts.w1': experts_params,
        'block_sparse_moe.experts.w2': experts_params,
        'block_sparse_moe.experts.w3': experts_params,
    }

    model.quantize_model(quant_config=quant_config, compute_dtype=torch.float16)
    model.eval()

    # Save the quantized model in the specified output directory
    output_model_name = os.path.join(output_dir, f"{model_id}_Q{quant_level}_hqq.pt")
    torch.save(model.state_dict(), output_model_name)
    print(f"Quantized model (Q{quant_level}) saved as {output_model_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_identifier>")
        sys.exit(1)
    
    model_identifier = sys.argv[1]
    output_dir = "quantized_models"  # Name of the output directory for storing quantized models

    # Create the output directory near the script's location
    output_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    for q_level in [2, 3, 4, 8]:
        quantize_and_save_model(model_identifier, output_dir_path, quant_level=q_level)
