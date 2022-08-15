from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
from datasets import Dataset
from datasets import ClassLabel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling

import pandas as pd
import numpy as np
import sys

epochs = 10
block_size = 512
model_checkpoints = ["bert-base-uncased", "huggingface/CodeBERTa-small-v1", "microsoft/codebert-base-mlm", "microsoft/graphcodebert-base"]
model_checkpoint =  "/home/akarmakar/codekitty/finetune/exp/CodeBERTa-small-v1-finetuned-COMP/checkpoint-20000"
model_dataset = "jemma_COMP_MAIN_CCOMM"

train_csv_path = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_train_MAIN.csv"
valid_csv_path  = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_valid_MAIN.csv"
datasets = load_dataset("csv", data_files={"train": train_csv_path, "test": valid_csv_path})


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text", "call_type", "class_id", "method_id", "project_id", "project_size", "project_split", "labels"])
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)


model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-finetuned-{model_dataset}-{epochs}",
    # evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    # load_best_model_at_end=True,

    seed=42,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=5000,
    save_steps=5000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=epochs,
    load_best_model_at_end=True,
    # push_to_hub=True,
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# trainer.train()    


test_csv_path = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_valid_MAIN.csv"
test_data = pd.read_csv(test_csv_path, header=0) #load_dataset(dataset_name, split="test") ## pd.read_csv(test_csv_path)
X_test = list(test_data["text"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
test_dataset = Dataset(X_test_tokenized)    

raw_pred, _, _ = trainer.predict(test_dataset)       # Make prediction
y_pred = np.argmax(raw_pred, axis=1)                 # Preprocess raw predictions

# ********************** #    

# PRED
print("Going for prediction ... ")
print(y_pred[:5])
print("")

df = pd.DataFrame(y_pred)
df.to_csv(sys.path[0] + f"/{model_name}-finetuned-{model_dataset}-{epochs}" + "/eval_pred.csv", index=False, header=["pred_label"])
# eval_data = pd.read_csv(save_path + "/eval_pred.csv", header=0)
# print_eval_scores(test_data["label"], eval_data["pred_label"], save_path)
