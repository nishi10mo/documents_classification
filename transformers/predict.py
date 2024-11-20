import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification

import pdb

def predict(data, tokenizer, model):
    column_names = data.columns[2:]
    id = 0
    for row in data.itertuples():
        id += 1
        encoded_input = tokenizer(row.SENTENCES, return_tensors='pt')
        output = model(**encoded_input)

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(output.logits))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= 0.5)] = 1

        active_columns = [column_names[i] for i in range(len(column_names)) if y_pred[0, i] == 1]

        print(f">>> id{id} >>>")
        print(f"probs: {probs}")
        print(f"labels: {active_columns}")
        print(f"<<< id{id} <<<")
        print("")

def main():
    df_predict = pd.read_csv("./data/kaggle_multi_label_classify/csv/predict.csv")

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("./exp/kaggle_multi_label_classify/weights")

    predict(df_predict, tokenizer, model)

if __name__=="__main__":
    main()
