import subprocess
import argparse
import pdb
import os
import json
import matplotlib.pyplot as plt

def make_graph(train_sizes, f1_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, f1_scores, marker='o')
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('f1_score on Test Data', fontsize=12)
    plt.title('scratch training using allenNLP', fontsize=14)
    plt.grid()

    # ファイル名を指定して保存
    plt.savefig("visualize/img/f1_score", dpi=300)
    plt.show()

def get_f1score(json_dir):
    # F1_scoresリストを初期化
    f1_scores = []

    # ディレクトリ内のすべてのファイルをループ処理
    for filename in os.listdir(json_dir):
        # JSONファイルのみを対象
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            
            # JSONファイルを読み込み
            with open(filepath, "r") as file:
                data = json.load(file)
                
                # F1scoreを取得してリストに追加
                if "F1score" in data:
                    f1_scores.append(data["F1score"])
    return f1_scores

def train(data, task, num):
    # トレーニングコマンドの実行
    train_command = f"allennlp train --serialization-dir exp/{data}/exp-{data}-{task}-{num} --recover --include-package src.model --include-package src.dataset_reader --include-package src.tokenizer jsons/{data}/{data}-{task}-{num}.jsonnet"
    subprocess.run(train_command, shell=True, check=True)

    # 性能の評価
    eval_command = f"allennlp evaluate --output-file visualize/jsons/train-{num}.json --include-package src.model --include-package src.dataset_reader --include-package src.tokenizer exp/{data}/exp-{data}-{task}-{num}/model.tar.gz data/{data}/{data}-test.jsonl"
    subprocess.run(eval_command, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="kaggle_multi_label_classify", choices=["livedoor_news", "kaggle_multi_label_classify"], help="dataset name")
    parser.add_argument("--task_name", type=str, default="multi_label_classify", choices=["classify", "classify_topk", "multi_label_classify"], help="task name")
    args = parser.parse_args()

    nums = [50, 100, 200, 400, 800]
    # for num in nums:
    #     train(args.data_name, args.task_name, num)
    f1_scores = get_f1score("visualize/jsons")
    make_graph(nums, f1_scores)

if __name__=="__main__":
    main()
