import sys
import subprocess


models = [
    #'huggingface/CodeBERTa-small-v1',

    "bert-base-uncased", 
    # "microsoft/codebert-base",
    # "huggingface/CodeBERTa-small-v1", 
    # "microsoft/graphcodebert-base",

    # "uclanlp/plbart-java-cs",
    # "uclanlp/plbart-multi_task-java",
    # "uclanlp/plbart-large",
    # "anjandash/JavaBERT-mini",
    # "anjandash/JavaBERT-small",  
 
    # "Fujitsu/AugCode",
    # "ProsusAI/finbert",
    # "Salesforce/codet5-base",        
]

dataset = "giganticode/java-cmpx-v1"

for model in models:
    logfile = sys.path[0] + "/finetune/models/finetuned_" + model[model.find("/")+1:] + "_" + dataset[dataset.find("/")+1:] + ".log"
    command = f"CUDA_VISIBLE_DEVICES=1,2,3 python3 /home/akarmakar/codekitty/finetune/finetune_and_evaluate.py --model_name {model} --tokenizer_name {model} --dataset_name {dataset} > {logfile}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outputs = process.communicate()[0].decode('utf-8')

    print(f"Done finetuning {model} on {dataset}.")
    print(outputs, "\n\n")