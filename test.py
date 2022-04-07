from datasets import load_dataset
import configparser

# dataset_name = "anjandash/java-8m-methods-v1"

# print(dataset_name.find("/")+1)

# ds_train = load_dataset(dataset_name, split="train")
# ds_valid = load_dataset(dataset_name, split="validation")
# ds_test  = load_dataset(dataset_name, split="test")

# print((ds_valid["text"]))

# train_data = load_dataset("giganticode/java-cmpx-v1", split="train")  ## pd.read_csv(train_csv_path)

# X = list(train_data["text"])
# y = list(train_data["label"])

# print(len(set(y)))


config = configparser.ConfigParser()
config.read("./finetune/config/config.ini")

print((config["train"]["seed"]))



# ***** #

# CUDA_VISIBLE_DEVICES=1,2,3 python3 /home/akarmakar/codekitty/finetune/finetune_and_evaluate.py --model_name huggingface/CodeBERTa-small-v1 --tokenizer_name huggingface/CodeBERTa-small-v1 --dataset_name giganticode/java-cmpx-v1

# for k,v in models.items():
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(v)
#         model = AutoModelForSequenceClassification.from_pretrained(v, num_labels=2)
#     except Exception as e:
#         print(e)

#
#
#

# ***** #
