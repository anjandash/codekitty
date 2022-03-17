from datasets import load_dataset


# dataset_name = "anjandash/java-8m-methods-v1"

# print(dataset_name.find("/")+1)

# ds_train = load_dataset(dataset_name, split="train")
# ds_valid = load_dataset(dataset_name, split="validation")
# ds_test  = load_dataset(dataset_name, split="test")

# print((ds_valid["text"]))

train_data = load_dataset("giganticode/java-cmpx-v1", split="train")  ## pd.read_csv(train_csv_path)

X = list(train_data["text"])
y = list(train_data["label"])

print(len(set(y)))