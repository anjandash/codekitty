names='huggingface/CodeBERTa-small-v1'
    # microsoft/codebert-base'

    # "bert-base-uncased", 
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

dataset='giganticode/java-cmpx-v1'
for name in $names
do
CUDA_VISIBLE_DEVICES=1,2,3 python3 /home/akarmakar/codekitty/finetune/finetune_and_evaluate.py --model_name $name --tokenizer_name $name --dataset_name $dataset > $name + "_" + $dataset + ".log"
done