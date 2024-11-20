from typing import Dict, List, Union
import json
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader, TextClassificationJsonReader
from allennlp.data.fields import TextField, Field, ListField, MultiLabelField
from allennlp.data.instance import Instance
import pdb

@DatasetReader.register("multi_label_text_classification_json")
class MultiLabelTextClassificationJsonReader(TextClassificationJsonReader):
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in self.shard_iterable(data_file.readlines()):
                if not line:
                    continue
                items = json.loads(line)
                text = items[self._text_key]
                label = items.get(self._label_key)
                if label is not None:
                    # ラベルがリストの場合の処理
                    if isinstance(label, list):
                        if self._skip_label_indexing:
                            try:
                                label = [int(lbl) for lbl in label]
                            except ValueError:
                                raise ValueError(
                                    "All labels in the list must be integers if skip_label_indexing is True."
                                )
                        else:
                            label = [str(lbl) for lbl in label]
                    # ラベルがリストでない場合の処理
                    else:
                        if self._skip_label_indexing:
                            try:
                                label = int(label)
                            except ValueError:
                                raise ValueError(
                                    "Labels must be integers if skip_label_indexing is True."
                                )
                        else:
                            label = str(label)
                yield self.text_to_instance(text=text, label=label)

    def text_to_instance(  # type: ignore
        self, text: str, label: Union[str, int] = None
    ) -> Instance:
        """
        # Parameters

        text : `str`, required.
            The text to classify
        label : `str`, optional, (default = `None`).
            The label for this text.

        # Returns

        An `Instance` containing the following fields:
            - tokens (`TextField`) :
              The tokens in the sentence or phrase.
            - label (`MultiLabelField`) :
              The label label of the sentence or phrase.
        """

        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens))
            fields["tokens"] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields["tokens"] = TextField(tokens)
        if label is not None:
            fields["label"] = MultiLabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)
