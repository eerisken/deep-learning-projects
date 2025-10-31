# -*- coding: utf-8 -*-
"""llm-turkish-news-classifier.ipynb
# **LLM (google-gemma) Turkish news headlines text classifier**
"""

print("Installing compatible PyTorch version (for CUDA 12.1)...")
!pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cu121

print("Installing other dependencies...")
!pip install -U "transformers==4.40.0" "datasets==2.18.0" "peft==0.10.0" "trl==0.8.6" "bitsandbytes==0.43.0" "accelerate==0.29.3"

# Install the correct triton version
!pip install triton==2.2.0

# Install pandas for CSV loading
!pip install pandas

print("✅ All libraries installed.")
print("‼️  IMPORTANT: Now go to 'Runtime > Restart session...' and run the *next* cell.")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset
from trl import SFTTrainer
import os

# to download gemma, need to create a READ-ACCESS TOKEN on huggingface
from huggingface_hub import notebook_login
notebook_login()

import pandas as pd

df = pd.read_csv("/content/TurkishHeadlines.csv")

df.columns = ['text', 'label']
df = df.dropna(subset=['text', 'label']) # Drop any rows with missing data
print(f"Total rows: {len(df)}")
print("Categories found:")
print(df['label'].value_counts())

def format_prompt(example):
    text = example['text']
    label = example['label']
    # This is the simple key-value format
    return {
        "formatted_text": f"<start_of_turn>user\nMetin: \"{text}\"\nKategori:<end_of_turn>\n<start_of_turn>model\n{label}<end_of_turn>"
    }

print("Formatting and splitting data...")
dataset = Dataset.from_pandas(df)

# Split 90% for training, 10% for evaluation
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

train_dataset = train_dataset.map(format_prompt, remove_columns=['text', 'label'])
eval_dataset = eval_dataset.map(format_prompt, remove_columns=['text', 'label'])

print("✅ Data is ready!")
print(f"Training examples: {len(train_dataset)}")
print(f"Evaluation examples: {len(eval_dataset)}")
print("--- Sample Prompt ---")
print(train_dataset[0]['formatted_text'])

model_id = "google/gemma-2b-it" # The instruction-tuned model

# --- Quantization Config ---
# This configures the model to load in 4-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # "nf4" (Normalized Float 4)
    bnb_4bit_use_double_quant=True, # a second quantization for even more memory savings
    bnb_4bit_compute_dtype=torch.bfloat16 # bfloat16 for computations
)

# --- Load Model ---
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto" # Automatically maps layers to GPU/CPU
)

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Set pad token to end-of-sentence token
tokenizer.padding_side = "right"

print("✅ Model and Tokenizer loaded in 4-bit!")

# --- Prepare model for k-bit training ---
# This freezes the original 4-bit model and prepares it for LoRA
model = prepare_model_for_kbit_training(model)

# --- LoRA Config ---
peft_config = LoraConfig(
    r=16,  # The "rank" of the LoRA matrices. Higher is more params, but 16 is a good default.
    lora_alpha=32, # A scaling factor. (alpha / r) is the scaling.
    lora_dropout=0.05, # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM",
    # Target the attention layers of the Gemma model
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Attach LoRA adapters to the model
model = get_peft_model(model, peft_config)

print("✅ LoRA adapters attached to the model.")
model.print_trainable_parameters() # See how few params we're training!

print("Setting up Trainer...")
training_args = TrainingArguments(
    output_dir="turkish-headline-classifier",
    report_to="none",
    num_train_epochs=2,  # <-- With 4K data, 2 epochs is perfect.
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, # Simulate a larger batch size
    optim="paged_adamw_8bit",
    logging_steps=50,
    learning_rate=2e-5,
    fp16=False,
    bf16=True,
    evaluation_strategy="steps", # <-- We can now evaluate!
    eval_steps=100,              # <-- Check performance every 100 steps
    save_steps=100,              # <-- Save checkpoint every 100 steps
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset, # 90%
    eval_dataset=eval_dataset,   # 10%
    peft_config=peft_config,
    dataset_text_field="formatted_text",
    max_seq_length=512,
    args=training_args,
)

print("🚀 Starting real training...")
trainer.train()

print("🏁 Training complete!")

# --- Save the final adapter ---
adapter_model_name = "gemma-turkish-classifier-final"
trainer.model.save_pretrained(adapter_model_name)
tokenizer.save_pretrained(adapter_model_name)

print(f"✅ Final LoRA adapter saved to '{adapter_model_name}'")

from peft import PeftModel

print("base model loading...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, # "google/gemma-2b-it"
    quantization_config=bnb_config,
    device_map="auto"
)

print("adapter loading...")
adapter_path = "gemma-turkish-classifier-final" # Kaydettiğiniz adaptör
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval() # Modeli "değerlendirme" moduna al

# --- test sentence ---
new_text = "Tüpraş 7 yıl vadeyle 700 milyon dolar borçlanıyor" # Etiket: Ekonomi

prompt = f"<start_of_turn>user\nMetin: \"{new_text}\"\nKategori:<end_of_turn>\n<start_of_turn>model\n"
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")

eot_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
# ----------------------------------

outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    eos_token_id=eot_token_id,         # <-- DÜZELTME BURADA (2. Kısım)
    pad_token_id=tokenizer.eos_token_id # pad_token_id'yi de ayarlamak iyi bir pratiktir
)

prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\n" + "="*30)
print(f"Test text: {new_text}")
print(f"model prediction: {prediction.strip()}")
print("="*30)

!zip -r gemma-turkish-classifier-final.zip gemma-turkish-classifier-final

print("✅ Zipping complete!")
