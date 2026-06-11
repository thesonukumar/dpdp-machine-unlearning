# Model and Dataset Configurations
# We use TinyLlama because it fits in a single 16GB VRAM GPU, proving zero-budget deployment feasibility.
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# The locuslab/TOFU dataset acts as our compliance sandbox.
DATASET_NAME = "locuslab/TOFU"

# The forget split represents the targeted users whose data we must surgically remove (DPDP erasure).
FORGET_SPLIT = "forget10"

# The retain split represents adjacent profiles that must remain intact. This ensures our model utility stays high.
RETAIN_SPLIT = "retain90"

# LoRA (Low-Rank Adaptation) Configurations
# We use LoRA to optimize training by only updating a small number of parameters, keeping the baseline stable.
LORA_RANK = 8             # Rank 8 provides a good balance between expressiveness and compute overhead.
LORA_ALPHA = 16           # Alpha scales the learned weights. 16 is a standard default for stable updates.
LORA_DROPOUT = 0.05       # 5% dropout prevents overfitting during the unlearning updates.

# Targeting q, v, and o projections allows us to alter the attention mechanics directly, which is where memorization occurs.
LORA_TARGET_MODULES = ["q_proj", "v_proj", "o_proj"]

# Training Loop Configurations
# Max length of 512 ensures unified sequence length windows for structural consistency across execution steps.
MAX_LENGTH = 512

# Small batch size of 2 fits comfortably in 16GB VRAM.
BATCH_SIZE = 2

# 4 epochs are typically sufficient for gradient ascent to unlearn the specific targeted entities without catastrophic collapse.
EPOCHS = 4

# The KL regularization parameter. 0.05 ensures the unlearning updates do not cause structural damage to the network.
LAMBDA_KL = 0.05

# Learning rate of 1e-5 is conservative enough to avoid "Total Linguistic Collapse" during gradient ascent.
LEARNING_RATE = 1e-5

# Adapters will be saved here to prove local execution without exposing data to external APIs.
OUTPUT_DIR = "./adapters"

# Hardware target execution.
DEVICE = "cuda"
