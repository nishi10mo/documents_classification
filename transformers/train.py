import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, BertConfig, Trainer, TrainingArguments
from src.dataset import TextClassificationDataset
from src.metrics import *

import pdb

def main():
    df_train = pd.read_csv("./data/kaggle_multi_label_classify/csv/train.csv")
    df_train = df_train.head(500)
    df_test = pd.read_csv("./data/kaggle_multi_label_classify/csv/test.csv")

    df_train["labels"] = df_train[df_train.columns[2:]].values.tolist()
    df_test["labels"] = df_test[df_test.columns[2:]].values.tolist()

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    max_length = 512
    train_dataset = TextClassificationDataset(df_train, tokenizer, max_length)
    eval_dataset = TextClassificationDataset(df_test, tokenizer, max_length)

    model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", problem_type="multi_label_classification", num_labels=6)
    model.cuda()

    training_args = TrainingArguments(output_dir="./exp/kaggle_multi_label_classify/checkpoints/", logging_strategy="epoch", report_to='none')

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir="./exp/kaggle_multi_label_classify/weights/")
    trainer.evaluate()

if __name__=="__main__":
    main()
