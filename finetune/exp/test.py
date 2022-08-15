from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForMaskedLM
from datasets import Dataset
from datasets import ClassLabel
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline

import pandas as pd
import numpy as np
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

model_checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

predictor = pipeline(task="fill-mask", model=model, tokenizer=tokenizer)
x = predictor("I am [MASK] the sailor man.")

print(x)