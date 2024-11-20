
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, BertConfig, Trainer, TrainingArguments
from src.dataset import TextClassificationDataset
from src.metrics import *

import pdb

def make_graph(train_sizes, f1_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, f1_scores, marker='o')
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('f1_score on Test Data', fontsize=12)
    plt.title('f1_score vs Training Data Size', fontsize=14)
    plt.grid()

    # ファイル名を指定して保存
    plt.savefig("visualize/img/f1_score", dpi=300)
    plt.show()

def main():
    f1_scores = []
    train_sizes = [50, 100, 200, 400, 800]
    for train_size in train_sizes:
        df_train = pd.read_csv("./data/kaggle_multi_label_classify/csv/train.csv")
        df_validation = pd.read_csv("./data/kaggle_multi_label_classify/csv/validation.csv")
        df_test = pd.read_csv("./data/kaggle_multi_label_classify/csv/test.csv")
        df_train = df_train.head(train_size)
        df_validation = df_validation.head(100)
        df_test = df_test.head(100)

        df_train["labels"] = df_train[df_train.columns[2:]].values.tolist()
        df_validation["labels"] = df_validation[df_validation.columns[2:]].values.tolist()
        df_test["labels"] = df_test[df_test.columns[2:]].values.tolist()

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        max_length = 512
        train_dataset = TextClassificationDataset(df_train, tokenizer, max_length)
        eval_dataset = TextClassificationDataset(df_validation, tokenizer, max_length)
        test_dataset = TextClassificationDataset(df_test, tokenizer, max_length)

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
        metrics = trainer.evaluate(test_dataset)
        print(f">>> result (train_size:{train_size}) >>>")
        print(metrics)
        print(f"<<< result (train_size:{train_size}) <<<")
        f1_scores.append(metrics["eval_f1"])
    make_graph(train_sizes, f1_scores)

if __name__=="__main__":
    main()
