import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import gc

torch.mps.empty_cache()
gc.collect()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["c_attn", "c_proj"], 
    lora_dropout=0.1,  
    bias="none",  
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config).to(device)

wikipedia_dataset = load_dataset("wikipedia", "20220301.en", split="train[:6000]")  
cnn_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:6000]")  

def preprocess_wikipedia(examples):
    inputs = ["Expand: " + " ".join(example.split(". ")[:5] + [""]) for example in examples["text"]]
    outputs = examples["text"]
    return {"input": inputs, "output": outputs}

def preprocess_cnn(examples):
    inputs = ["Expand: " + " ".join(example.split()[:50]) for example in examples["highlights"]]
    outputs = examples["article"]
    return {"input": inputs, "output": outputs}

wikipedia_dataset = wikipedia_dataset.map(preprocess_wikipedia, batched=True)
cnn_dataset = cnn_dataset.map(preprocess_cnn, batched=True)

combined_dataset = concatenate_datasets([wikipedia_dataset, cnn_dataset])

def tokenize_function(examples):
    model_inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"] 
    return model_inputs

tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=15, 
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=False,  
    evaluation_strategy="epoch",  
    save_strategy="epoch", 
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss", 
    greater_is_better=False,  
    report_to="none",  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

def generate_expanded_content(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=5,  
        temperature=0.7,  
        top_p=0.9,  
        no_repeat_ngram_size=2, 
        early_stopping=True,  
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

input_text = "Artificial intelligence"
expanded_content = generate_expanded_content(input_text)
print("Expanded Content:", expanded_content)