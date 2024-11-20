import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        """
        文書分類タスク用のデータセットクラス。

        Args:
            dataframe (pd.DataFrame): "SENTENCES"と"labels"の列を含むデータフレーム。
            tokenizer (transformers.PreTrainedTokenizer): トークナイザ。
            max_length (int): トークナイズ後の最大トークン数。
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # データフレームの行数を返す
        return len(self.data)

    def __getitem__(self, idx):
        # 指定インデックスのデータを取得
        sentence = self.data.loc[idx, "SENTENCES"]
        label = self.data.loc[idx, "labels"]
        
        # トークナイズ
        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # トークナイズ結果をPyTorchのテンソル形式に変換
        input_ids = encoding["input_ids"].squeeze()  # (1, max_length) -> (max_length)
        attention_mask = encoding["attention_mask"].squeeze()  # (1, max_length) -> (max_length)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.float)  # ラベルをテンソルに変換
        }