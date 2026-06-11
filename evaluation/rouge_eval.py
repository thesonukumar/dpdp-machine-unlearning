import os
import sys
import torch
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def evaluate_model_utility(training_model, retain_dataset):
    """
    Evaluates Model Utility (MU) using ROUGE-L F1 scoring.
    We generate answers for the retain set and compare against ground truth.
    This ensures no catastrophic forgetting occurred during the gradient ascent.
    Target: Preservation of > 95% of baseline metrics.
    """
    print("Evaluating Model Utility via ROUGE-L...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    training_model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    total_rouge_l_f1 = 0.0
    num_samples = min(20, len(retain_dataset)) # Run on a subset for faster evaluation during training
    
    with torch.no_grad():
        for i in range(num_samples):
            # The dataset was already pre-tokenized into full chat templates, 
            # including the assistant's answer.
            # To evaluate utility properly, we need to extract just the question 
            # and prompt the model to generate the response.
            
            input_ids = retain_dataset[i]["input_ids"]
            # We must set skip_special_tokens=False so the <|user|> tags remain in the text for parsing.
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            
            try:
                # Extract the question and answer from the formatted chat string
                user_query = decoded_text.split("<|user|>")[1].split("<|assistant|>")[0].strip()
                # Remove the EOS token (</s>) from the ground truth answer
                ground_truth = decoded_text.split("<|assistant|>")[1].replace("</s>", "").strip()
            except IndexError:
                continue # Skip if parsing fails
                
            prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{user_query}</s>\n<|assistant|>\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(training_model.device)
            
            # Generate response
            outputs = training_model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Score against ground truth
            scores = scorer.score(ground_truth, generated_text)
            total_rouge_l_f1 += scores['rougeL'].fmeasure
            
    avg_rouge_l = total_rouge_l_f1 / num_samples
    print(f"Model Utility (ROUGE-L F1) | Score: {avg_rouge_l:.4f}")
    
    return avg_rouge_l
