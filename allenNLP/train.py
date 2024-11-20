import subprocess
import argparse

def train(data, task):
    # トレーニングコマンドの実行
    train_command = f"allennlp train --serialization-dir exp/{data}/exp-{data}-{task} --include-package src.model --include-package src.dataset_reader --include-package src.tokenizer jsons/{data}/{data}-{task}.jsonnet"
    subprocess.run(train_command, shell=True, check=True)

    # トレーニング後にモデルファイルの内容を確認
    tar_command = f"tar tf exp/{data}/exp-{data}-{task}/model.tar.gz"
    subprocess.run(tar_command, shell=True, check=True)

    # 性能の評価
    eval_command = f"allennlp evaluate --include-package src.model --include-package src.dataset_reader --include-package src.tokenizer exp/{data}/exp-{data}-{task}/model.tar.gz data/{data}/{data}-test.jsonl"
    subprocess.run(eval_command, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="kaggle_multi_label_classify", choices=["livedoor_news", "kaggle_multi_label_classify"], help="dataset name")
    parser.add_argument("--task_name", type=str, default="multi_label_classify", choices=["classify", "classify_topk", "multi_label_classify"], help="task name")
    args = parser.parse_args()

    train(args.data_name, args.task_name)

if __name__=="__main__":
    main()
