import os, sys, csv 

from subprocess import Popen, PIPE
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

models = {
    # "BERT":             "bert-base-uncased", 
    # "CodeBERT":         "microsoft/codebert-base",
    "CodeBERTa":        "huggingface/CodeBERTa-small-v1", 
    # "GraphCodeBERT":    "microsoft/graphcodebert-base",
    # "PLBART-csjava":    "uclanlp/plbart-java-cs",
    # "PLBART-mtjava":    "uclanlp/plbart-multi_task-java",
    # "PLBART-large":     "uclanlp/plbart-large",
    # "JavaBERT-mini":    "anjandash/JavaBERT-mini",
    # "JavaBERT-small":   "anjandash/JavaBERT-small",    
    # "AugCode":          "Fujitsu/AugCode",
    # "FinBERT":          "ProsusAI/finbert",
    # "CodeT5":           "Salesforce/codet5-base",
}


dataset_name = "giganticode/java-cmpx-v1"
for k, model_name in models.items():
    print("Finetuning", model_name, " on", dataset_name)
    print("-----------------------")
    command = "CUDA_VISIBLE_DEVICES=1,2,3 python3 /home/akarmakar/codekitty/finetune/finetune_and_evaluate.py --model_name " + model_name + " --tokenizer_name " + model_name + " --dataset_name " + dataset_name
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    outputs = process.communicate()[0].decode("utf-8")

    print("Model: ", model_name)
    print("Dataset: ", dataset_name)
    print("-----------------------")
    print(outputs)
    print("-----------------------")



# CUDA_VISIBLE_DEVICES=1,2,3 python3 /home/akarmakar/codekitty/finetune/finetune_and_evaluate.py --model_name huggingface/CodeBERTa-small-v1 --tokenizer_name huggingface/CodeBERTa-small-v1 --dataset_name giganticode/java-cmpx-v1

# for k,v in models.items():
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(v)
#         model = AutoModelForSequenceClassification.from_pretrained(v, num_labels=2)
#     except Exception as e:
#         print(e)