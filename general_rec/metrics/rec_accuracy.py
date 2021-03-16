from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("rec_accuracy")
class RecAccuracy(Metric):
    def __init__(self, top_k: int = 1) -> None:
        if top_k <= 0:
            raise ConfigurationError("top_k passed to RecAccuracy must be > 0")
        self._top_k = top_k
        self.accuracy_sum = 0.0
        self.total_count = 0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        """
        predictions, gold_labels = self.detach_tensors(predictions, gold_labels)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim():
            raise ConfigurationError(
                "gold_labels must have dimension == predictions.size() but "
                "found tensor of shape: {}".format(predictions.size())
            )
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to RecAccuracy contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        predictions = predictions.view(-1, num_classes)
        gold_labels = gold_labels.view(-1, num_classes).long()

        # Top K indexes of the predictions (or fewer, if there aren't K of them).
        # Special case topk == 1, because it's common and .max() is much faster than .topk().
        if self._top_k == 1:
            top_k = predictions.max(-1)[1].unsqueeze(-1)
        else:
            top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

        batch_size = predictions.shape[0]
        predictions_bin = torch.zeros(batch_size, num_classes, dtype=torch.long, device=gold_labels.device)
        for i in range(batch_size):
            predictions_bin[i, top_k[i]] = 1

        correct_tensor = predictions_bin & gold_labels
        union_tensor = predictions_bin | gold_labels

        # This is of shape (batch_size, ..., top_k).
        correct = torch.sum(correct_tensor, -1)
        union_count = torch.sum(union_tensor, -1)
        
        self.accuracy_sum += torch.sum(correct.float() / union_count.float())
        self.total_count += batch_size

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated precision.
        """
        if self.total_count > 1e-12:
            precision = float(self.accuracy_sum) / float(self.total_count)
        else:
            precision = 0.0
        if reset:
            self.reset()
        return precision

    @overrides
    def reset(self):
        self.accuracy_sum = 0.0
        self.total_count = 0

