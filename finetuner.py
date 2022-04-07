import os, sys, csv 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

models = {
    #"BERT":             "bert-base-uncased", 
    #"CodeBERT":         "microsoft/codebert-base",
    #"CodeBERTa":        "huggingface/CodeBERTa-small-v1", 
    #"GraphCodeBERT":    "microsoft/graphcodebert-base",
    "CodeT5":           "Salesforce/codet5-base",
    #"JavaBERT-mini":    "anjandash/JavaBERT-mini",
    #"JavaBERT-small":   "anjandash/JavaBERT-small",
    #"PLBART-mtjava":    "uclanlp/plbart-multi_task-java",
    #"PLBART-large":     "uclanlp/plbart-large",
}

train_data = load_dataset("giganticode/java-cmpx-v1", split="train")  ## pd.read_csv(train_csv_path)
X = list(train_data["text"])
y = list(train_data["label"])


for k,v in models.items():
    try:
        tokenizer = AutoTokenizer.from_pretrained(v)
        model = AutoModelForSequenceClassification.from_pretrained(v, num_labels=len(set(y)))
    except Exception as e:
        print(e)