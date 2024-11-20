import subprocess
import argparse

def predict(data, task):
    predict_command = f"allennlp predict --include-package src.model --include-package src.dataset_reader --include-package src.tokenizer exp/{data}/exp-{data}-{task}/model.tar.gz data/{data}/{data}-predict.jsonl"
    results = subprocess.run(predict_command, shell=True, check=True, capture_output=True, text=True)
    results = results.stdout
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="kaggle_multi_label_classify", choices=["livedoor_news", "kaggle_multi_label_classify"], help="dataset name")
    parser.add_argument("--task_name", type=str, default="multi_label_classify", choices=["classify", "classify_topk", "multi_label_classify"], help="task name")
    args = parser.parse_args()

    results = predict(args.data_name, args.task_name)
    print(">>> results >>>")
    print(results)
    print("<<< results <<<")

if __name__=="__main__":
    main()
