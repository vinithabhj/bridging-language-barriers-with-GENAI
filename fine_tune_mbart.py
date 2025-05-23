
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Load dataset
dataset = load_dataset("Aarif1430/english-to-hindi", split="train")

# Split the dataset into train and validation
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]  # 80% for training
eval_dataset = train_test_split["test"]   # 20% for validation

# Model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))

# PEFT configuration for LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="SEQ_2_SEQ_LM"  # Correct task type for translation
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config)

# Preprocessing function
def preprocess_function(examples):
    # Prepare the source (input) and target (output) text
    inputs = [f"translate English to Hindi: {ex}" for ex in examples["english_sentence"]]
    targets = examples["hindi_sentence"]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

    # Tokenize targets as labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, padding="max_length", truncation=True)

    # Ensure labels are properly formatted
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply preprocessing to datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # Adjust based on GPU memory
    per_device_eval_batch_size=2,
    num_train_epochs=3,  # Number of epochs
    fp16=True,  # Mixed precision
    logging_steps=10,  # Log every 10 steps
    logging_dir="./logs",
    logging_strategy="steps",
    eval_steps=100,  # Evaluate every 100 steps
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=1e-4,  # Adjusted learning rate
    optim="adamw_torch",
    report_to="none"  # Disable wandb reporting
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model()

# Evaluation (optional)
from datasets import load_metric

bleu = load_metric("sacrebleu")

# Test translation
def translate(text, tokenizer, model, device):
    inputs = tokenizer(f"translate English to Hindi: {text}", return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate BLEU score on validation data
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

references = []
predictions = []

for example in eval_dataset:
    pred = translate(example["english_sentence"], tokenizer, model, device)
    predictions.append(pred)
    references.append([example["hindi_sentence"]])

score = bleu.compute(predictions=predictions, references=references)
print(f"BLEU Score: {score['score']}")
