# 環境構築
python3.8.13で動作することを確認済みです。
以下のコマンドで必要なライブラリをインストールし、作業ディレクトリに移動してください。
```
pip install allennlp==2.9.3 google-cloud-storage==2.1.0 cached-path==1.1.2
pip install fugashi[unidic]
python -m unidic download
cd allenNLP
```
<br>

以下のコマンドで学習することができます。
```
python train.py --data_name kaggle_multi_label_classify --task_name multi_label_classify
```
<br>

学習済みのモデルを用いて、以下のコマンドで文書分類を行うことができます。
```
python predict.py --data_name kaggle_multi_label_classify --task_name multi_label_classify
```
