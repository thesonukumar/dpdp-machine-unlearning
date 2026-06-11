import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

# Add project root to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_and_prepare_datasets():
    """
    Loads the TOFU dataset splits and prepares them for the TinyLlama model.
    Returns: forget_dataset, retain_dataset
    """
    # Initialize the tokenizer to convert raw text into neural network inputs.
    # We use the exact tokenizer from the model to maintain vocabulary alignment.
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Set pad_token if not present, preventing errors during batch padding.
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load our compliance sandbox dataset directly from HuggingFace Hub.
    # The forget split contains the PII we must erase, simulating a DPDP removal request.
    # The retain split acts as our baseline utility anchor.
    raw_forget = load_dataset(config.DATASET_NAME, config.FORGET_SPLIT, split="train")
    raw_retain = load_dataset(config.DATASET_NAME, config.RETAIN_SPLIT, split="train")

    def format_and_tokenize(example):
        # We inject the data into the precise chat template TinyLlama was pre-trained on.
        # This is critical because unlearning requires attacking the exact neural pathways 
        # used during natural conversational inference.
        chat_prompt = (
            f"<|system|>\nYou are a helpful AI assistant.</s>\n"
            f"<|user|>\n{example['question']}</s>\n"
            f"<|assistant|>\n{example['answer']}</s>"
        )
        
        # Tokenize with strict constraints (max_length, padding, truncation).
        # This ensures all input matrices have identical dimensions, preventing OOM 
        # errors and maintaining structural consistency in single-GPU execution.
        tokenized = tokenizer(
            chat_prompt,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH,
        )
        
        # During standard causal language modeling (and unlearning), the labels are the input_ids.
        # We include 'labels' explicitly so the model can easily compute its own forward loss later.
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    # Map the tokenization function across all records, removing the original text columns 
    # since the model only processes numerical tensors.
    forget_dataset = raw_forget.map(format_and_tokenize, remove_columns=raw_forget.column_names)
    retain_dataset = raw_retain.map(format_and_tokenize, remove_columns=raw_retain.column_names)
    
    # Cast datasets to PyTorch tensors so they can be fed directly into the model.
    forget_dataset.set_format(type="torch")
    retain_dataset.set_format(type="torch")

    return forget_dataset, retain_dataset
