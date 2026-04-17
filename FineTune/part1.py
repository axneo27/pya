from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# === Config ===
MODEL_ID = "meta-llama/Meta-Llama-3-1B"
DATASET_PATH = "my_data.jsonl"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    load_in_8bit=True,  # set False if on CPU or no bitsandbytes
)

# === LoRA configuration ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # depends on model architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# === Load dataset ===
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")


# === Preprocess ===
def format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


dataset = dataset.map(format_example)

# === Training args ===
args = TrainingArguments(
    output_dir="./finetuned-llama-1b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# === Train ===
trainer.train()

# === Save ===
model.save_pretrained("./finetuned-llama-1b")
tokenizer.save_pretrained("./finetuned-llama-1b")
