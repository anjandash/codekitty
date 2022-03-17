import sys 
import torch
import argparse
import configparser
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy  = accuracy_score(y_true=labels, y_pred=pred)
    recall    = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1_score_ = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1_score_}        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",       help="Enter the base model name or path which needs to be finetuned", required=True)
    parser.add_argument("--tokenizer_name",   help="Enter the base model name or path from which the tokenizer will be loaded", required=True)
    parser.add_argument("--dataset_name",     help="Enter the dataset name or path which will be used to finetune the model", required=True)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("./config/config.ini")

    # ********************** #    

    train_model_name = args.model_name       #"bert-base-uncased"                      # OR local checkpoint_path
    tokenizer_name   = args.tokenizer_name   #"bert-base-uncased"                      # OR local checkpoint_path
    dataset_name     = args.dataset_name     #"giganticode/java-cmpx-v1"      

    # ********************** #

    train_data = load_dataset(dataset_name, split="train")  ## pd.read_csv(train_csv_path)
    X = list(train_data["text"])
    y = list(train_data["label"])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(train_model_name, num_labels=len(set(y)))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # ********************** #    

    args = TrainingArguments(
        output_dir=("finetuned_" + train_model_name[train_model_name.find("/")+1:] + "_" + dataset_name[dataset_name.find("/")+1:]),
        seed=config.getint("train", "seed"),
        evaluation_strategy=config.get("train", "evaluation_strategy"),
        eval_steps=config.getint("train", "eval_steps"),
        per_device_train_batch_size=config.getint("train", "per_device_train_batch_size"),
        per_device_eval_batch_size=config.getint("train", "per_device_eval_batch_size"),
        num_train_epochs=config.getint("train", "num_train_epochs"),
        load_best_model_at_end=config.getboolean("train", "load_best_model_at_end"))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    trainer.train()

