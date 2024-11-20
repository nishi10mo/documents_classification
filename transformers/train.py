import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BertForSequenceClassification, AdamW, BertConfig, Trainer, TrainingArguments
from src.dataset import TextClassificationDataset
from src.metrics import *

import pdb

df_train = pd.read_csv("./data/kaggle_multi_label_classify/csv/train.csv")
df_train = df_train.head(500)
df_validation = pd.read_csv("./data/kaggle_multi_label_classify/csv/validation.csv")
df_test = pd.read_csv("./data/kaggle_multi_label_classify/csv/test.csv")
df_predict = pd.read_csv("./data/kaggle_multi_label_classify/csv/predict.csv")

df_train["labels"] = df_train[df_train.columns[2:]].values.tolist()
df_validation["labels"] = df_validation[df_validation.columns[2:]].values.tolist()

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
max_length = 512
train_dataset = TextClassificationDataset(df_train, tokenizer, max_length)
eval_dataset = TextClassificationDataset(df_validation, tokenizer, max_length)
# pdb.set_trace()

# input_ids_train = []
# attention_masks_train = []
# for x in X_train:
#     x = tokenizer.encode_plus(x, add_special_tokens = True, max_length = 512, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')
#     input_ids_train.append(x["input_ids"])
#     attention_masks_train.append(x["attention_mask"])
# input_ids_train = torch.cat(input_ids_train, dim=0)
# attention_masks_train = torch.cat(attention_masks_train, dim=0)

# pdb.set_trace()
# train_dataset = TensorDataset(input_ids_train, attention_masks_train, y_train)


# df_validation["labels"] = df_validation[df_validation.columns[2:]].values.tolist()
# X_validation = df_validation["SENTENCES"].values
# y_validation = df_validation["labels"].values

# input_ids_validation = []
# attention_masks_validation = []
# for x in X_validation:
#     x = tokenizer.encode_plus(x, add_special_tokens = True, max_length = 512, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt')
#     input_ids_validation.append(x["input_ids"])
#     attention_masks_validation.append(x["attention_mask"])
# input_ids_validation = torch.cat(input_ids_validation, dim=0)
# attention_masks_validation = torch.cat(attention_masks_validation, dim=0)

# validation_dataset = TensorDataset(input_ids_validation, attention_masks_validation, y_validation)


# # データローダーの作成
# batch_size = 32

# # 訓練データローダー
# dataloader_train = DataLoader(
#             train_dataset,  
#             sampler = RandomSampler(train_dataset), # ランダムにデータを取得してバッチ化
#             batch_size = batch_size
#         )

# # 検証データローダー
# dataloader_validation = DataLoader(
#             validation_dataset, 
#             sampler = SequentialSampler(validation_dataset), # 順番にデータを取得してバッチ化
#             batch_size = batch_size
#         )


model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", problem_type="multi_label_classification", num_labels=6)
model.cuda()

# optimizer = AdamW(model.parameters(), lr=2e-5)

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


# def train(model, dataloader):
#     model.train()
#     train_loss = 0
#     for batch in dataloader:
#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2].to(device)
#         optimizer.zero_grad()
#         loss, logits = model(b_input_ids, 
#                              token_type_ids=None, 
#                              attention_mask=b_input_mask, 
#                              labels=b_labels)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         train_loss += loss.item()
#     return train_loss


# def validation(model, dataloader):
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for batch in dataloader:
#             b_input_ids = batch[0].to(device)
#             b_input_mask = batch[1].to(device)
#             b_labels = batch[2].to(device)
#             with torch.no_grad():        
#                 (loss, logits) = model(b_input_ids, 
#                                     token_type_ids=None, 
#                                     attention_mask=b_input_mask,
#                                     labels=b_labels)
#             val_loss += loss.item()
#     return val_loss

# # 学習の実行
# max_epoch = 10
# train_loss_ = []
# test_loss_ = []

# for epoch in range(max_epoch):
#     train_ = train(model)
#     test_ = train(model)
#     train_loss_.append(train_)
#     test_loss_.append(test_)
