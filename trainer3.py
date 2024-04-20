from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_model(dataset_path, local_model_dir, output_dir):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    model = AutoModelForCausalLM.from_pretrained(local_model_dir, use_cache=False)  # Disable caching at model level

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("text", data_files={"train": dataset_path})
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        fp16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=3,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer
    )

    trainer.train()

# Configuration for paths and model directory
dataset_path = "path/to/traing/data.txt"
local_model_dir = "/home/user/local_model"
output_dir = "/home/user/out_dir"

fine_tune_model(dataset_path, local_model_dir, output_dir)
