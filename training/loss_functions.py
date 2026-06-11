import torch
import torch.nn.functional as F

def gradient_ascent_loss(model, batch):
    """
    Computes the loss for the forget set and explicitly flips it for unlearning.
    """
    # Forward pass the forget batch through the model.
    # We must explicitly define the device alignment here.
    outputs = model(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
        labels=batch["labels"].to(model.device)
    )
    
    # Extract the standard cross-entropy loss from the causal language modeling forward pass.
    loss = outputs.loss
    
    # CRITICAL INSTRUCTION: NEGATE THE LOSS
    # By returning the negative loss, the optimizer will perform Gradient Ascent instead of Descent.
    # This breaks the neural pathways responsible for regenerating the private identifier characters.
    # This is the erasure engine.
    negated_loss = -loss
    
    return negated_loss

def kl_divergence_loss(training_model, frozen_model, batch, lambda_kl):
    """
    Computes the KL divergence between the training model and the frozen baseline on the retain set.
    This prevents the unlearning process from causing catastrophic linguistic collapse.
    """
    # Forward pass on the retain batch through the training model.
    training_outputs = training_model(
        input_ids=batch["input_ids"].to(training_model.device),
        attention_mask=batch["attention_mask"].to(training_model.device)
    )
    # Log softmax for the predicted distribution
    training_logits = F.log_softmax(training_outputs.logits, dim=-1)

    # Forward pass on the retain batch through the frozen baseline model.
    # We do not want gradients for this forward pass.
    with torch.no_grad():
        frozen_outputs = frozen_model(
            input_ids=batch["input_ids"].to(frozen_model.device),
            attention_mask=batch["attention_mask"].to(frozen_model.device)
        )
        # Softmax for the target baseline distribution
        frozen_logits = F.softmax(frozen_outputs.logits.to(training_model.device), dim=-1)

    # Compute the Kullback-Leibler divergence.
    # This penalizes the training model if its probability distributions shift too far away 
    # from the original frozen model's logical reasoning pathways.
    kl_loss = F.kl_div(
        training_logits,
        frozen_logits,
        reduction="batchmean"
    )

    # Scale the KL divergence by the regularizer weight (lambda).
    return lambda_kl * kl_loss
