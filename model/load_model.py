import os
import sys
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_trainable_model():
    """
    Loads the base TinyLlama model and injects LoRA adapters.
    This model will undergo gradient ascent for unlearning.
    """
    # Load the base model in 16-bit precision to reduce VRAM usage.
    # This proves zero-budget deployment on limited hardware like Kaggle P100.
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=config.DEVICE
    )

    # Configure LoRA to target specific attention components.
    # We target q_proj, v_proj, and o_proj because memorization heavily relies on attention mechanics.
    # By only training these adapters, we prevent full catastrophic forgetting of the network.
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Inject the LoRA adapters into the base model.
    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters to verify we are only optimizing a fraction of the network.
    peft_model.print_trainable_parameters()

    return peft_model
