from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import pipeline

import pandas as pd
import numpy as np
import os,sys,csv


model_checkpoints = ["bert-base-uncased", "huggingface/CodeBERTa-small-v1", "microsoft/codebert-base-mlm", "microsoft/graphcodebert-base"]
model_checkpoint = model_checkpoints[0]

model_datasets = ["jemma_COMP_MAIN_NSPACE", "jemma_COMP_MAIN_CSPACE"]
model_dataset = "jemma_COMP_MAIN_NSPACE"

checkpoint_number = 200000
epoch = 10

for model_checkpoint in model_checkpoints:
    for model_dataset in model_datasets:

        model_checkpoint_path =  "/home/akarmakar/codekitty/finetune/exp/"+model_checkpoint.split("/")[-1]+"-finetuned-"+model_dataset+"-"+str(epoch)+"/checkpoint-"+str(checkpoint_number)+"/"
        train_csv_path = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_train_MAIN.csv"
        valid_csv_path  = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_valid_MAIN.csv"

        print("Going for prediction ... ")
        print("_________________________")

        test_csv_path = "/home/akarmakar/codekitty/data/"+model_dataset+"/JEMMA_COMP_test_MAIN.csv"
        test_data = pd.read_csv(test_csv_path, header=0) #load_dataset(dataset_name, split="test") ## pd.read_csv(test_csv_path)

        X_test = list(test_data["text"])
        y_test = list(test_data["labels"])

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path)
        predictor = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)

        orig = []
        pred = []
        noms = 0
        errs = 0

        print(len(X_test))
        print(len(y_test))
        for snippet, orig_label in zip(X_test, y_test):
            if "[MASK]" not in snippet:
                print("[MASK] not in snippet")
                if orig_label in snippet:
                    print("But orig_label in snippet")
                    snippet = snippet.replace(orig_label, "[MASK]")
                else:
                    print("ERROR:")
                    print(snippet)
                    continue

            snippet = snippet.replace("[MASK]", "<mask>")
            snippetx = tokenizer(snippet, truncation=True, max_length=512)
            snippety = tokenizer.decode(snippetx["input_ids"])
            snippet = snippety.replace("<s>", "").replace("</s>", "")

            # ONLY FOR BERT
            if model_checkpoint == "bert-base-uncased":
                if "< mask >" in snippet:
                    snippet = snippet.replace("< mask >", "[MASK]")

            # if "<mask>" not in snippet:
            #     noms+=1
            #     continue

            try:
                predictions = predictor(snippet)
                pred_label = (predictions[0]["token_str"])
            except Exception as e:
                print(e)
                #print("If [MASK] not found, possibly [MASK] is truncated after tokenization.")
                #input()
                errs+=1
                continue

            # print("orig_label:", orig_label)
            # print("pred_label:", pred_label)
            # print()

            orig.append(orig_label)
            pred.append(pred_label)

        # pred_dict = {"orig": orig, "pred": pred} 
        # df = pd.DataFrame([pred_dict])
        # df.to_csv(f"{model_checkpoint_path}{(model_checkpoint.split('/')[-1])}-checkpoint-{str(checkpoint_number)}-finetuned-{model_dataset}-{str(epoch)}__eval_pred.csv", index=False)

        with open(f"{model_checkpoint_path}{(model_checkpoint.split('/')[-1])}-checkpoint-{str(checkpoint_number)}-finetuned-{model_dataset}-{str(epoch)}__eval_pred__MANUAL.csv", "w+") as wr:
            csv_writer = csv.writer(wr)
            csv_writer.writerow(["orig", "pred"])

            for orig_label, pred_label in zip(orig, pred):
                csv_writer.writerow([orig_label, pred_label])


        print("Accuracy:", accuracy_score(orig, pred))
        print("Truncated inputs without masks:", noms)
        print("Errors:", errs)