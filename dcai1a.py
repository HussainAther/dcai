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

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_tokenized_dataset = dataset_train.map(tokenize_function, batched=True)
train_tokenized_dataset = train_tokenized_dataset.cast_column("label", ClassLabel(names = ["0", "1"]))

test_tokenized_dataset = dataset_test.map(tokenize_function, batched=True)
test_tokenized_dataset = test_tokenized_dataset.cast_column("label", ClassLabel(names = ["0", "1"]))

training_args = TrainingArguments(max_steps=max_training_steps, output_dir=model_folder)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
)

trainer.train()  # may take a while to train (try to run on a GPU if you can access one)

pred_probs = trainer.predict(test_tokenized_dataset).predictions
pred_classes = np.argmax(pred_probs, axis=1)
print(f"Error rate of predictions: {np.mean(pred_classes != test_tokenized_dataset['label'])}")
