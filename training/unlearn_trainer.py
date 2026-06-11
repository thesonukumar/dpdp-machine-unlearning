import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from training.loss_functions import gradient_ascent_loss, kl_divergence_loss

def run_unlearning_loop(training_model, frozen_model, forget_dataset, retain_dataset):
    """
    Executes the custom PyTorch training loop to perform parameter-level unlearning.
    We do NOT use HuggingFace SFTTrainer because we need tight control over the combined
    Gradient Ascent + KL Divergence loss formulation.
    """
    # Create DataLoaders for both datasets to manage batching.
    forget_dataloader = DataLoader(forget_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    retain_dataloader = DataLoader(retain_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize the AdamW optimizer to update the LoRA adapters.
    optimizer = torch.optim.AdamW(training_model.parameters(), lr=config.LEARNING_RATE)

    training_model.train()
    
    print("Starting the Unlearning Optimization Loop...")
    for epoch in range(config.EPOCHS):
        epoch_loss = 0.0
        
        # Iterate over the targeted forget data and the baseline retain data simultaneously.
        # zip() pairs a forget batch with a retain batch for every step.
        for step, (forget_batch, retain_batch) in enumerate(zip(forget_dataloader, retain_dataloader)):
            
            # Step 1: Force Erasure via Gradient Ascent
            # Compute the negative cross-entropy loss on the private PII data.
            ga_loss = gradient_ascent_loss(training_model, forget_batch)
            
            # Step 2: Preserve Utility via KL Divergence Regularization
            # Compute the drift penalty to ensure the model doesn't lose general intelligence.
            kl_loss = kl_divergence_loss(training_model, frozen_model, retain_batch, config.LAMBDA_KL)
            
            # Step 3: The Unified Objective Loss Formula
            # Balance the two competing mathematical forces.
            total_loss = ga_loss + kl_loss
            
            # Step 4: Backpropagation
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += total_loss.item()
            
            # Log loss every 10 steps to monitor stability.
            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{config.EPOCHS}] Step [{step}] | GA Loss: {ga_loss.item():.4f} | KL Loss: {kl_loss.item():.4f} | Total Loss: {total_loss.item():.4f}")
                
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {epoch_loss / (step + 1):.4f} ---")
        
    return training_model
