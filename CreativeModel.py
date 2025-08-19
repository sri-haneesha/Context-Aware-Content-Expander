import torch
import gc
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.mps.empty_cache()
gc.collect()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

prompt = "Once Upon a time"
inputs = tokenizer(prompt, return_tensors='pt').to(device)
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100,
    do_sample=True,
    top_k=50,
    top_p=0.95
)
output_str = tokenizer.batch_decode(outputs)
print("\nPre-training output for 'There once lived dragon.':")
print(output_str)
'''
short_stories_dataset = load_dataset("roneneldan/TinyStories", split="train[:12000]")
short_stories_dataset = short_stories_dataset.train_test_split(train_size=0.8)

def preprocess_batch(batch):
    all_text_items = batch["text"]
    trimmed_text_items = [x[:500] for x in all_text_items]
    return tokenizer(trimmed_text_items)

tokenized_dataset = short_stories_dataset.map(
    preprocess_batch,
    batched=True,
    batch_size=10,
    remove_columns=short_stories_dataset["train"].column_names,
)
print("\nTokenized dataset:", tokenized_dataset)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=10,
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,
    save_strategy="epoch",
    logging_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    args=training_args
)

trainer.train()

model.save_pretrained("./results_ut_v2")
tokenizer.save_pretrained("./results_ut_v2")

model = AutoModelForCausalLM.from_pretrained("./results_ut_v2").to(device)

prompt = "There once lived dragon."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100,
    do_sample=True,
    top_k=50,
    top_p=0.95
)
output_string = tokenizer.batch_decode(outputs)
print("\nPost-training output for 'There once lived dragon.':")
print(output_string)

def generate_text_from_model(prompt, max_length=150, temperature=0.1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=4,
        top_k=10,
        top_p=0.7,
        temperature=temperature,
        repetition_penalty=2.5,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).replace("Expand: ", "")

def interactive_text_generation():
    print("\nEnter a prompt to expand or type 'no' to quit.\n")
    while True:
        user_input = input("Enter prompt: ").strip()
        if user_input.lower() in {"no", "end", "exit", "quit"}:
            print("Model execution ended!")
            break
        prompt = "Expand: " + user_input
        generated_text = generate_text_from_model(prompt)
        print("\nGenerated Text:\n" + generated_text + "\n")
        cont = input("Do you want to enter another prompt? (yes/no): ").strip().lower()
        if cont not in {"yes", "y"}:
            print("Alright, see you later!")
            break

interactive_text_generation()

torch.mps.empty_cache()
gc.collect()
'''