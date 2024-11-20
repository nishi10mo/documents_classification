{
    "random_seed": 2,
    "pytorch_seed": 2,
    "train_data_path": "data/livedoor_news/livedoor_news-train.jsonl",
    "validation_data_path": "data/livedoor_news/livedoor_news-validation.jsonl",
    "dataset_reader": {
        "type": "text_classification_json",
        "tokenizer": {
            "type": "mecab"
        },
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "data_loader": {
        "type": "simple",
        "batch_size": 32,
        "shuffle": true
    },
    "validation_data_loader": {
        "type": "simple",
        "batch_size": 32,
        "shuffle": false
    },
    "vocabulary": {},
    "datasets_for_vocab_creation": ["train"],
    "model": {
        "type": "TopKBasicClassifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100
                }
            }
        },
        "seq2vec_encoder": {
        "type": "bag_of_embeddings",
        "embedding_dim": 100
        }
    },
    "trainer": {
        "optimizer": {
            "type": "adam"
        },
        "num_epochs": 10,
        "patience": 3,
        "callbacks": [
            {
                "type": "tensorboard"
            }
        ]
    }
}