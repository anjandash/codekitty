from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
from datasets import Dataset
from datasets import ClassLabel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
# import evaluate

model_checkpoint = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

block_size = tokenizer.model_max_length
# block_size = 128

def tokenize_function(examples):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=512)

max_input_length = 512
max_target_length = 10

def preprocess_function(examples):
    model_inputs = tokenizer(examples["text"], max_length=max_input_length, padding='max_length', truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["labels"], max_length=max_target_length, padding='max_length', truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs  

# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#         # customize this part to your needs.
#     total_length = (total_length // block_size) * block_size
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result    


train_csv_path = "/Users/anjandash/Desktop/JEMMA_CODE_COMPLETION_EXP/NEON/JEMMA_LOCALNESS_CODEKITTY/JEMMA_LOCALNESS_CK_MAIN/JEMMA_COMP_train_CODEKITTY_RFF.csv"
test_csv_path  = "/Users/anjandash/Desktop/JEMMA_CODE_COMPLETION_EXP/NEON/JEMMA_LOCALNESS_CODEKITTY/JEMMA_LOCALNESS_CK_MAIN/JEMMA_COMP_test_CODEKITTY_RFF.csv"

datasets = load_dataset("csv", data_files={"train": train_csv_path, "test": test_csv_path})
print(datasets)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenized_datasets = datasets.map(preprocess_function, batched=True, num_proc=4, remove_columns=["text", "call_type", "class_id", "method_id", "project_id", "project_size", "project_split"])

# lm_datasets = tokenized_datasets.map(
#     group_texts,
#     batched=True,
#     batch_size=1000,
#     num_proc=4,
# )

print(tokenized_datasets)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
   
model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-finetuned-COMP",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)


# from transformers import DataCollatorForLanguageModeling
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# metric = evaluate.load("exact_match")
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    #compute_metrics=compute_metrics,
    #data_collator=data_collator,
)

print(trainer.train_dataset[0]["attention_mask"])
print(trainer.train_dataset[0]["input_ids"])
print(trainer.train_dataset[0]["labels"])

trainer.train()
#eval_results = trainer.evaluate()




# ********************** #
