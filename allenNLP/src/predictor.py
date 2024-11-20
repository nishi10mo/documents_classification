# delete予定
import numpy
from typing import List, Dict
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

import pdb

@Predictor.register("top_k_text_classifier")
class TopKTextClassifierPredictor(TextClassifierPredictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instances([instance])
        return sanitize(outputs)

    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        """
        Converts incoming json to a [`Instance`](../data/instance.md),
        runs the model on the newly created instance, and adds labels to the
        `Instance`s given by the model's output.

        # Returns

        `List[instance]`
            A list of `Instance`'s.
        """

        pdb.set_trace()
        instance = self._json_to_instance(inputs)
        self._dataset_reader.apply_token_indexers(instance)
        outputs = self._model.forward_on_instances(instance)
        new_instances = self.predictions_to_labeled_instances(instance, outputs)
        return new_instances

    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        pdb.set_trace()
        new_instance = instance.duplicate()
        
        # 確率の高い3つのクラスのインデックスを取得
        top_k = 3
        top_k_indices = numpy.argsort(outputs["probs"])[-top_k:][::-1]

        # 上位3つのクラスのインデックスをフィールドとして追加
        for idx, label in enumerate(top_k_indices):
            new_instance.add_field(f"label_{idx+1}", LabelField(int(label), skip_indexing=True))
        
        return [new_instance]
