import torch
from typing import Dict, Optional
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models import Model, BasicClassifier
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1MultiLabelMeasure
import pdb

# 確率が最も高いラベルだけでなく、確率が高い上からtop_k個のラベルを出力するよう改変したモデル
@Model.register("TopKBasicClassifier")
class TopKBasicClassifier(BasicClassifier):
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        上位3つの確率の高いラベルを出力するメソッド。
        """
        predictions = output_dict["probs"]
        
        # 上位3つのインデックスを取得
        top_k = 3
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]

        classes = []
        for prediction in predictions_list:
            # 確率を降順にソートし、上位3つのインデックスを取得
            top_k_indices = torch.argsort(prediction, dim=-1, descending=True)[:top_k]
            top_k_labels = [
                self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(idx.item(), str(idx.item()))
                for idx in top_k_indices
            ]
            classes.append(top_k_labels)

        output_dict["label"] = classes

        # トークンIDを実際の単語に変換
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens

        return output_dict


# 出力をソフトマックスにするのではなく、シグモイド関数に変更したモデル
# マルチラベル分類タスクでは、そのようにすることが一般的
@Model.register("MultiLabelClassifier")
class MultiLabelClassifier(BasicClassifier):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, text_field_embedder, seq2vec_encoder, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = F1MultiLabelMeasure(average="weighted")
        self._loss = torch.nn.BCEWithLogitsLoss()
        initializer(self)


    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        label: torch.IntTensor = None,
        metadata: MetadataField = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        # probs = torch.nn.functional.sigmoid(logits)
        probs = torch.sigmoid(logits)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            # pdb.set_trace()
            # loss = self._loss(logits, label.long().view(-1))
            loss = self._loss(logits, label.float())
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        # 閾値の設定
        threshold = 0.5
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            # 閾値を超えたラベルを出力
            label_indices = (prediction >= threshold).nonzero(as_tuple=True)[0]
            label_strs = [
                self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                    idx.item(), str(idx.item())
                )
                for idx in label_indices
            ]
            classes.append(label_strs)
        output_dict["label"] = classes
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"F1score": self._accuracy.get_metric(reset)["fscore"]}
        return metrics
