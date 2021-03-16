import json

from typing import Iterator, List, Dict
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField, MultiLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.vocabulary import Vocabulary


@DatasetReader.register("fasttext_dataset_reader")
class FasttextDatasetReader(DatasetReader):
    def __init__(self,
                tokenizer: Tokenizer = None,
                token_indexers: Dict[str, TokenIndexer] = None,
                lazy: bool = False) -> None:
        super().__init__(lazy)
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(token_min_padding_length=8, lowercase_tokens=True)
            }
    def text_to_instance(self, text: str, labels: List[str]=None) -> Instance:
        tokenized_text = self.tokenizer.tokenize(text)
        if len(tokenized_text) > 1024:
            # tokenized_text = tokenized_text[:128] + tokenized_text[-128:]
            tokenized_text = tokenized_text[:1024]
        text_field = TextField(tokenized_text, self.token_indexers)
        fields = {'text': text_field}
        if labels is not None:
            labels_field = MultiLabelField(labels=labels)
            fields['labels'] = labels_field
        return Instance(fields)
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r', encoding='utf-8') as f:
            for each_line in f:
                splits_segs = each_line.split(sep=' ')
                text = ''
                labels = []
                for each_seg in splits_segs:
                    if each_seg.startswith('__label__'):
                        labels.append(each_seg[9:].replace('<s>', ' '))
                    else:
                        text += ' ' + each_seg

                yield self.text_to_instance(text, labels)

                
