import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import re

torch.mps.empty_cache()
gc.collect()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

creative_model_path = "./results_ut_v2"
try:
    creative_tokenizer = AutoTokenizer.from_pretrained(creative_model_path)
    creative_model = AutoModelForCausalLM.from_pretrained(creative_model_path).to(device)
    if creative_tokenizer.pad_token is None:
        creative_tokenizer.pad_token = creative_tokenizer.eos_token
    creative_model.config.pad_token_id = creative_tokenizer.pad_token_id
    print("Creative model (TinyStories/Reddit) loaded successfully!")
except Exception as e:
    print(f"Error loading creative model: {e}")
    exit()

factual_model_path = "./fine_tuned_model"
try:
    factual_tokenizer = AutoTokenizer.from_pretrained(factual_model_path)
    peft_config = PeftConfig.from_pretrained(factual_model_path)
    base_model_name = peft_config.base_model_name_or_path  
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    factual_model = PeftModel.from_pretrained(base_model, factual_model_path).to(device)
    if factual_tokenizer.pad_token is None:
        factual_tokenizer.pad_token = factual_tokenizer.eos_token
    factual_model.config.pad_token_id = factual_tokenizer.pad_token_id
    print("Factual model (Wikipedia/CNN) loaded successfully!")
except Exception as e:
    print(f"Error loading factual model: {e}")
    exit()

def generate_creative_text(prompt, max_length=150, temperature=0.1):
    if not prompt.strip():
        return "Error: Prompt cannot be empty."
    input_ids = creative_tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    try:
        output = creative_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=90,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            top_k=50,
            top_p=0.90,
            temperature=0.70,
            repetition_penalty=1.5,
            do_sample=True,
            pad_token_id=creative_tokenizer.pad_token_id
        )
        return creative_tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Creative generation error: {e}"

def generate_factual_text(prompt, max_length=256):
    if not prompt.strip():
        return "Error: Prompt cannot be empty."
    full_prompt = f"Prompt: {prompt}\nExpansion: "
    inputs = factual_tokenizer(full_prompt, return_tensors="pt").to(device)
    try:
        outputs = factual_model.generate(
            **inputs,
            max_length=max_length,
            min_length=120,
            num_beams=7,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_p=0.8,
            temperature=0.65,
            repetition_penalty=1.6,
            early_stopping=True,
            pad_token_id=factual_tokenizer.pad_token_id
        )
        text = factual_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.split("Expansion: ")[1] if "Expansion: " in text else text
    except Exception as e:
        return f"Factual generation error: {e}"

def split_sentences(text):
    # Split on .!?, keeping punctuation, and handle multiple spaces
    sentences = re.split(r'([.!?])\s+', text.strip())
    # Combine punctuation with sentence text
    result = []
    for i in range(0, len(sentences)-1, 2):
        result.append(sentences[i] + sentences[i+1])
    # Handle last sentence (may lack punctuation, e.g., gibberish)
    if len(sentences) % 2 == 1:
        result.append(sentences[-1])
    return [s.strip() for s in result if s.strip()]

def test_combined_prompts():
    print("Both models loaded successfully!")
    print("Choose mode:\n1. Creative (storytelling)\n2. Factual (informative)\nType 'quit' to exit.\n")
    
    creative_samples = [
        "The night is very dark.",
        "There once lived a princess in a castle.",
        "A small village in the mountains is clouded by heavy rains.",
        "The curious cat explored the jungle.",
        "One day, a spaceship landed on the land."
    ]
    factual_samples = [
        "Soccer",
        "Climate change is concerning.",
        "The future of the space exploration is booming.",
        "schools are becoming unsafe.",
        "Obesity is very unhealthy. "
    ]
    
    while True:
        choice = input("Enter mode (1 for creative, 2 for factual, or 'quit'): ").strip().lower()
        
        if choice in {"quit", "exit", "no"}:
            print("Exiting!")
            break
        elif choice == "1":
            print("\nCreative Mode (TinyStories/Reddit Model)")
            print("Sample outputs :\n")
            for prompt in creative_samples:
                expanded = generate_creative_text(prompt)
                sentences = split_sentences(expanded)
                if len(sentences) > 1:
                    # Print all but the last sentence
                    trimmed_text = " ".join(sentences[:-1])
                else:
                    trimmed_text = "No sentences to display (single sentence output)."
                print(f"Prompt: {prompt}")
                print(f"Generated Text: {trimmed_text}\n")
            while True:
                user_input = input("Enter creative prompt (or 'back' to switch mode): ").strip()
                if user_input.lower() in {"back", "quit", "exit"}:
                    break
                expanded = generate_creative_text(user_input)
                sentences = split_sentences(expanded)
                if len(sentences) > 1:
                    trimmed_text = " ".join(sentences[:-1])
                else:
                    trimmed_text = "No sentences to display (single sentence output)."
                print(f"\nGenerated Text: {trimmed_text}\n")
        elif choice == "2":
            print("\nFactual Mode (Wikipedia/CNN Model)")
            print("Sample outputs:\n")
            for prompt in factual_samples:
                expanded = generate_factual_text(prompt)
                sentences = split_sentences(expanded)
                if len(sentences) > 1:
                    trimmed_text = " ".join(sentences[:-1])
                else:
                    trimmed_text = "No sentences to display (single sentence output)."
                print(f"Prompt: {prompt}")
                print(f"Expanded Content: {expanded}\n")
            while True:
                user_input = input("Enter factual prompt (or 'back' to switch mode): ").strip()
                if user_input.lower() in {"back", "quit", "exit"}:
                    break
                expanded = generate_factual_text(user_input)
                sentences = split_sentences(expanded)
                if len(sentences) > 1:
                    trimmed_text = " ".join(sentences[:-1])
                else:
                    trimmed_text = "No sentences to display (single sentence output)."
                print(f"\nExpanded Content: {trimmed_text}\n")
        else:
            print("Invalid choice. Please enter 1, 2, or 'quit'.")

if __name__ == "__main__":
    test_combined_prompts()

torch.mps.empty_cache()
gc.collect()