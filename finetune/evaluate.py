import sys 
import torch
import argparse
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

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

def print_eval_scores(labels, pred):
    accuracy  = accuracy_score(y_true=labels, y_pred=pred)
    recall    = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1_score_ = f1_score(y_true=labels, y_pred=pred, average='macro')

    print("**********************")
    print("ACCURACY: ", accuracy)
    print("F1 SCORE: ", f1_score_)
    print("RECALL:   ", recall)
    print("PRECISION:", precision)
    print("**********************")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",       help="Enter the finetuned model name or checkpoint path that needs to be tested", required=True)
    parser.add_argument("--tokenizer_name",   help="Enter the base model name or path from which the tokenizer will be loaded", required=True)
    parser.add_argument("--dataset_name",     help="Enter the dataset name or path which will be used to test the model", required=True)
    args = parser.parse_args()

    # ********************** #    

    eval_model_name  = args.model_name        #"anjandash/finetuned-bert-java-cmpx-v1"  # OR local checkpoint_path *****
    tokenizer_name   = args.tokenizer_name    #"bert-base-uncased"                      # OR local checkpoint_path
    dataset_name     = args.dataset_name      #"giganticode/java-cmpx-v1"    

    # ********************** #

    test_data        = load_dataset(dataset_name, split="test")      ## pd.read_csv(test_csv_path)
    X_test           = list(test_data["text"])

    tokenizer        = AutoTokenizer.from_pretrained(tokenizer_name)
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)
    test_dataset     = Dataset(X_test_tokenized)    

    model            = AutoModelForSequenceClassification.from_pretrained(eval_model_name, num_labels=10) 
    test_trainer     = Trainer(model)                           # Define test trainer
    raw_pred, _, _   = test_trainer.predict(test_dataset)       # Make prediction
    y_pred           = np.argmax(raw_pred, axis=1)              # Preprocess raw predictions

    # ********************** #    

    eval_data = pd.DataFrame(y_pred)
    # df.to_csv(sys.path[0] + "/eval_pred.csv", index=False, header=["pred_label"])
    # eval_data = pd.read_csv(sys.path[0] + "/eval_pred.csv", header=0)
    print_eval_scores(test_data["label"], eval_data["pred_label"])

