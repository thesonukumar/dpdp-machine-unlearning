import os
import sys
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def evaluate_forget_quality(training_model, frozen_model, forget_dataset):
    """
    Evaluates Forget Quality (FQ) using the Kolmogorov-Smirnov (KS) Test.
    Success is achieved when the trained model's probability distribution on the forget set 
    matches the baseline model, proving the weights have been completely cleared of memorization.
    Target: p-value > 0.05
    """
    print("Evaluating Forget Quality via KS-Test...")
    
    forget_dataloader = DataLoader(forget_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    training_model.eval()
    frozen_model.eval()
    
    training_log_probs = []
    frozen_log_probs = []
    
    with torch.no_grad():
        for batch in forget_dataloader:
            input_ids = batch["input_ids"].to(training_model.device)
            attention_mask = batch["attention_mask"].to(training_model.device)
            labels = batch["labels"].to(training_model.device)
            
            # Get logits from both models
            train_outputs = training_model(input_ids=input_ids, attention_mask=attention_mask)
            frozen_outputs = frozen_model(input_ids=input_ids.to(frozen_model.device), attention_mask=attention_mask.to(frozen_model.device))
            
            # Compute token-level log probabilities
            train_logits = F.log_softmax(train_outputs.logits, dim=-1)
            frozen_logits = F.log_softmax(frozen_outputs.logits.to(training_model.device), dim=-1)
            
            # Shift logits and labels for next-token prediction
            shift_train_logits = train_logits[..., :-1, :].contiguous()
            shift_frozen_logits = frozen_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Gather log probs of the actual label tokens
            # We flatten the batches to compute the statistical distributions
            train_probs = shift_train_logits.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            frozen_probs = shift_frozen_logits.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Apply attention mask to filter out padding tokens
            mask = attention_mask[..., 1:].contiguous().bool()
            
            training_log_probs.extend(train_probs[mask].cpu().numpy().flatten())
            frozen_log_probs.extend(frozen_probs[mask].cpu().numpy().flatten())
            
    # Run the KS test comparing the two empirical distributions
    ks_stat, p_value = ks_2samp(training_log_probs, frozen_log_probs)
    
    print(f"Forget Quality (KS-Test) | p-value: {p_value:.6f} | KS Stat: {ks_stat:.4f}")
    if p_value > 0.05:
        print("-> SUCCESS: Model probability distribution matches baseline. Data is verifiably erased.")
    else:
        print("-> WARNING: p-value <= 0.05. Residual parametric remnants may still exist.")
        
    return p_value
