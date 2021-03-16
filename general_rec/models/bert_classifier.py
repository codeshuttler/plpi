
from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp.nn import util

from general_rec.metrics.rec_precision import RecPrecision
from general_rec.metrics.rec_recall import RecRecall
from general_rec.metrics.rec_accuracy import RecAccuracy
from general_rec.metrics.hamming_loss import HammingLoss

def get_f1score(precision, recall):
    beta2 = 1
    if beta2 * precision + recall < 1e-6:
        f1score = 0
    else:
        f1score = (1 + beta2) * precision * recall / (beta2 * precision + recall)
    return f1score


@Model.register("bert_classifier")
class BERTClassifier(Model):
    def __init__(self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            text_encoder: Seq2VecEncoder,
            classifier_feedforward: FeedForward):
        super(BERTClassifier, self).__init__(vocab)

        self.vocab = vocab
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss_function = torch.nn.BCELoss()

        self.hamming_loss2 = HammingLoss(top_k=2)
        self.accuracy2 = RecAccuracy(top_k=2)
        self.precision2 = RecPrecision(top_k=2)
        self.recall2 = RecRecall(top_k=2)

        self.hamming_loss3 = HammingLoss(top_k=3)
        self.accuracy3 = RecAccuracy(top_k=3)
        self.precision3 = RecPrecision(top_k=3)
        self.recall3 = RecRecall(top_k=3)

        self.hamming_loss5 = HammingLoss(top_k=5)
        self.accuracy5 = RecAccuracy(top_k=5)
        self.precision5 = RecPrecision(top_k=5)
        self.recall5 = RecRecall(top_k=5)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None
                ) -> Dict[str, torch.Tensor]:
        embedded_text = self.text_field_embedder(text)
        text_mask = util.get_text_field_mask(text)
        encoded_text = self.text_encoder(embedded_text, text_mask)

        logits = torch.sigmoid(self.classifier_feedforward(encoded_text))
        # print('logits:', logits.shape)
        # print('labels:', labels.shape)

        output = {"logits": logits}
        if labels is not None:
            self.hamming_loss2(logits, labels)
            self.accuracy2(logits, labels)
            self.precision2(logits, labels)
            self.recall2(logits, labels)

            self.hamming_loss3(logits, labels)
            self.accuracy3(logits, labels)
            self.precision3(logits, labels)
            self.recall3(logits, labels)

            self.hamming_loss5(logits, labels)
            self.accuracy5(logits, labels)
            self.precision5(logits, labels)
            self.recall5(logits, labels)
            output["loss"] = self.loss_function(logits, labels.float())
        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        hamming_loss2 = self.hamming_loss2.get_metric(reset)
        accuracy2 = self.accuracy2.get_metric(reset)
        precision2 = self.precision2.get_metric(reset)
        recall2 = self.recall2.get_metric(reset)
        f1score2 = get_f1score(precision2, recall2)

        hamming_loss3 = self.hamming_loss3.get_metric(reset)
        accuracy3 = self.accuracy3.get_metric(reset)
        precision3 = self.precision3.get_metric(reset)
        recall3 = self.recall3.get_metric(reset)
        f1score3 = get_f1score(precision3, recall3)

        hamming_loss5 = self.hamming_loss5.get_metric(reset)
        accuracy5 = self.accuracy5.get_metric(reset)
        precision5 = self.precision5.get_metric(reset)
        recall5 = self.recall5.get_metric(reset)
        f1score5 = get_f1score(precision5, recall5)

        return {
            "hamming_loss2": hamming_loss2,
            "accuracy2": accuracy2,
            "precision2": precision2,
            "recall2": recall2,
            "f1score2": f1score2,

            "hamming_loss3": hamming_loss3,
            "accuracy3": accuracy3,
            "precision3": precision3,
            "recall3": recall3,
            "f1score3": f1score3,
            
            "hamming_loss5": hamming_loss5,
            "accuracy5": accuracy5,
            "precision5": precision5,
            "recall5": recall5,
            "f1score5": f1score5,
            }
