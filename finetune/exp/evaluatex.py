from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
from datasets import Dataset
from datasets import ClassLabel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline

import pandas as pd
import numpy as np
import sys

epochs = 10
block_size = 512
model_checkpoints = ["bert-base-uncased", "huggingface/CodeBERTa-small-v1", "microsoft/codebert-base-mlm", "microsoft/graphcodebert-base"]

model_checkpoint = model_checkpoints[1]
model_checkpoint_path =  "/home/akarmakar/codekitty/finetune/exp/CodeBERTa-small-v1-finetuned-COMP/checkpoint-20000/"

model_dataset = "jemma_COMP_MAIN_CCOMM"

train_csv_path = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_train_MAIN.csv"
valid_csv_path  = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_valid_MAIN.csv"
datasets = load_dataset("csv", data_files={"train": train_csv_path, "test": valid_csv_path})

print("Going for prediction ... ")
print("")

test_csv_path = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_valid_MAIN.csv"
test_data = pd.read_csv(test_csv_path, header=0) #load_dataset(dataset_name, split="test") ## pd.read_csv(test_csv_path)

X_test = list(test_data["text"])
y_test = list(test_data["labels"])


from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path)
predictor = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

pred = []
for snippet in X_test:
    print(snippet)
    #x = predictor(snippet)
    break
