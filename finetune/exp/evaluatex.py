from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from transformers import pipeline

import pandas as pd
import numpy as np
import sys


model_checkpoints = ["bert-base-uncased", "huggingface/CodeBERTa-small-v1", "microsoft/codebert-base-mlm", "microsoft/graphcodebert-base"]

model_checkpoint = model_checkpoints[1]
model_dataset = "jemma_COMP_MAIN_CSPACE"
checkpoint_number = 20000

model_checkpoint_path =  "/home/akarmakar/codekitty/finetune/exp/"+model_checkpoint.split("/")[-1]+"-finetuned-"+model_dataset+"/checkpoint-"+str(checkpoint_number)+"/"
train_csv_path = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_train_MAIN.csv"
valid_csv_path  = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_valid_MAIN.csv"

print("Going for prediction ... ")
print("_________________________")

test_csv_path = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_valid_MAIN.csv"
test_data = pd.read_csv(test_csv_path, header=0) #load_dataset(dataset_name, split="test") ## pd.read_csv(test_csv_path)

X_test = list(test_data["text"])
y_test = list(test_data["labels"])

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path)
predictor = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

orig = []
pred = []
for snippet, orig_label in zip(X_test, y_test):
    if not "[MASK]" in snippet:
        snippet = snippet.replace(orig_label, "[MASK]")
        
    snippet = snippet.replace("[MASK]", "<mask>")
    snippetx = tokenizer(snippet, truncation=True, max_length=512)
    snippety = tokenizer.decode(snippetx["input_ids"])
    snippet = snippety.replace("<s>", "").replace("</s>", "")

    predictions = predictor(snippet)
    pred_label = (predictions[0]["token_str"])

    print("orig_label:", orig_label)
    print("pred_label:", pred_label)
    print()

    orig.append(orig_label)
    pred.append(pred_label)

# pred_dict = {"orig": orig, "pred": pred_label} 
# df = pd.DataFrame([pred_dict])
# df.to_csv(f"{model_checkpoint_path}eval_pred.csv", index=False)