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

model_name = "distilbert-base-uncased"  # which pretrained neural network weights to load for fine-tuning on our data
# other options you could try: "bert-base-uncased", "bert-base-cased", "google/electra-small-discriminator"

max_training_steps = 10  # how many iterations our network will be trained for
# Here set to a tiny value to ensure quick runtimes, set to higher values if you have a GPU to run this code on.

model_folder = "test_trainer"  # file where model will be saved after training
