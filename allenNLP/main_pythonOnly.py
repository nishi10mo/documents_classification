# jsonnetを用いずにpythonだけで完結させようとしてファイル
# delete予定

import pdb
import glob
import json
import os
import random
import torch
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from fugashi import Tagger
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.models import BasicClassifier
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training import GradientDescentTrainer
from allennlp.models.archival import load_archive, archive_model
from allennlp.predictors.predictor import Predictor

@Tokenizer.register("mecab")
class MecabTokenizer(Tokenizer):
    def __init__(self):
        # Taggerインスタンスを作成
        self._tagger = Tagger()

    def tokenize(self, text):
        """入力テキストをMeCabを用いて解析する"""
        tokens = []
        # 入力テキストを単語に分割
        for word in self._tagger(text):
            # 単語のテキスト（word.surface）と品詞（word.feature.pos1)からTokenインスタンスを作成
            token = Token(text=word.surface, pos_=word.feature.pos1)
            tokens.append(token)

        return tokens

def main():
    # GPUデバイスの確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 9個のラベルのリスト
    labels = ["dokujo-tsushin", "it-life-hack", "kaden-channel", "livedoor-homme",
            "movie-enter", "peachy", "smax", "sports-watch", "topic-news"]
    data = []

    # データセットをファイルから読み込む
    for label in labels:
        dir_path = os.path.join("data/livedoor_news/text", label)
        for file_path in sorted(glob.glob(os.path.join(dir_path, "*.txt"))):
            with open(file_path) as f:
                text = f.read()
                # メタデータを削除し、記事部分のみを用いる
                text = "".join(text.split("\n")[2:])
            data.append(dict(text=text, label=label))

    # データセットをランダムに並べ替える
    random.seed(1)
    random.shuffle(data)

    # データセットの80%を訓練データ、10%を検証データ、10%をテストデータとして用いる
    split_data = {}
    eval_size = int(len(data) * 0.1)
    split_data["test"] = data[:eval_size]
    split_data["validation"] = data[eval_size:eval_size * 2]
    split_data["train"] = data[eval_size * 2:]

    # 行区切りJSON形式でデータセットを書き込む
    for fold in ("train", "validation", "test"):
        out_file = os.path.join("data/livedoor_news", "livedoor_news_{}.jsonl".format(fold))
        with open(out_file, mode="w") as f:
            for item in split_data[fold]:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
    
    random.seed(2)
    torch.manual_seed(2)

    # tokenizer
    tokenizer = MecabTokenizer()

    # データセットリーダーの作成
    token_indexers = {"tokens": SingleIdTokenIndexer()}
    reader = TextClassificationJsonReader(tokenizer=tokenizer, token_indexers=token_indexers)

    # データセットを読み込んでリストに変換
    train_dataset = list(reader.read("data/livedoor_news/livedoor_news_train.jsonl"))
    validation_dataset = list(reader.read("data/livedoor_news/livedoor_news_validation.jsonl"))

    # データローダの作成
    train_loader = SimpleDataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = SimpleDataLoader(validation_dataset, batch_size=32, shuffle=False)

    # 語彙とトークンインデクサ
    vocab = Vocabulary.from_instances(train_loader.iter_instances())
    train_loader.index_with(vocab)
    validation_loader.index_with(vocab)

    # ミニバッチの作成
    batch = next(iter(validation_loader))
    words = [vocab.get_token_from_index(int(i), namespace="tokens") for i in batch["tokens"]["tokens"]["tokens"][0]]
    label = vocab.get_token_from_index(int(batch["label"][0]), namespace="labels")

    # 単語エンベディングの作成
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=100)
    text_embedder = BasicTextFieldEmbedder({"tokens": embedding})

    # テキストのベクトルの作成
    encoder = BagOfEmbeddingsEncoder(embedding_dim=100)

    # モデルの作成
    model = BasicClassifier(vocab=vocab, text_field_embedder=text_embedder, seq2vec_encoder=encoder)
    model.to(device)

    # optimizerの作成
    optimizer = AdamOptimizer(model.named_parameters())

    # trainerの作成
    serialization_dir = "model_param"
    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        validation_data_loader=validation_loader,
        num_epochs=10,
        patience=3,
        serialization_dir = serialization_dir
    )

    # モデルの訓練
    metrics = trainer.train()

    # モデルの保存
    # weights_path = trainer.get_best_weights_path()
    # archive_model(serialization_dir=serialization_dir)

    # モデルを用いて推論
    # archive = load_archive("exp_livedoor_news/model.tar.gz")
    # predictor = Predictor.from_archive(archive)
    # print(predictor.predict_json({"sentence": "新しい洗濯機がほしい"}))

if __name__=="__main__":
    main()
