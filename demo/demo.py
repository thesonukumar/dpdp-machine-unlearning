import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def live_interview_demo():
    """
    30-Second Live Interview Demo Protocol.
    Loads the unlearned LoRA adapter over the base model on CPU.
    Takes direct user queries to dynamically prove DPDP erasure compliance.
    """
    print("\n=============================================")
    print("  DPDP Act Compliance Filter - Live Demo")
    print("=============================================\n")
    print("Initializing environment...")

    # Force execution on CPU to prove zero-budget local inference capability.
    device = "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    print(f"Loading Base Model: {config.MODEL_NAME}")
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float32, # CPU inference uses float32
        device_map=device
    )
    
    adapter_path = config.OUTPUT_DIR
    if not os.path.exists(adapter_path):
        print(f"ERROR: No LoRA adapters found at {adapter_path}.")
        print("Please run the training loop via run_notebook.ipynb first, or download the Kaggle output.")
        sys.exit(1)
        
    print(f"Injecting Unlearned LoRA Adapters from: {adapter_path}")
    # Load the trained unlearning adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print("\nModel ready. Entering interactive mode.")
    print("Type 'exit' or 'quit' to terminate.")
    print("-" * 45)
    
    while True:
        try:
            user_input = input("\nUser Query > ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            if not user_input.strip():
                continue
                
            # Use the chat template the model expects
            prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate the response
            print("\nGenerating response...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9
                )
                
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            print("-" * 45)
            print(f"AI Response:\n{response}")
            print("-" * 45)
            
        except KeyboardInterrupt:
            break
            
    print("\nDemo terminated gracefully.")

if __name__ == "__main__":
    live_interview_demo()
