from datasets import load_dataset


dataset_name = "anjandash/java-8m-methods-v1"

print(dataset_name.find("/")+1)

# ds_train = load_dataset(dataset_name, split="train")
# ds_valid = load_dataset(dataset_name, split="validation")
# ds_test  = load_dataset(dataset_name, split="test")

# print((ds_valid["text"]))