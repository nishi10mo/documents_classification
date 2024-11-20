{
    "random_seed": 2,
    "pytorch_seed": 2,
    "train_data_path": "data/kaggle_multi_label_classify/kaggle_multi_label_classify-train.jsonl",
    "validation_data_path": "data/kaggle_multi_label_classify/kaggle_multi_label_classify-validation.jsonl",
    "dataset_reader": {
        "type": "multi_label_text_classification_json",
        "tokenizer": {
            "type": "spacy"
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
        "type": "MultiLabelClassifier",
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