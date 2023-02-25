import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import datasets

from datasets import Dataset, DatasetDict, ClassLabel

label_map = {"bad": 0, "good": 1}
dataset_train = Dataset.from_dict({"label": train["label"].map(label_map), "text": train["review"].values})
dataset_test = Dataset.from_dict({"label": test["label"].map(label_map), "text": test["review"].values})
