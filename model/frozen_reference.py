import os
import sys
import torch
from transformers import AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_frozen_model():
    """
    Loads a frozen copy of TinyLlama to serve as the intelligence anchor.
    This model NEVER receives gradient updates and is used to compute KL divergence.
    """
    # Load the identical base model in 16-bit precision.
    frozen_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu" # Move to CPU to save GPU VRAM, it's just for reference forwarding.
    )

    # CRITICAL: Freeze all parameters to ensure it acts as a perfect baseline anchor.
    # If this model updates, the KL divergence penalty will drift, breaking the utility retention.
    for param in frozen_model.parameters():
        param.requires_grad = False

    # Set to evaluation mode to disable dropout layers, ensuring deterministic logits for KL computation.
    frozen_model.eval()

    return frozen_model
