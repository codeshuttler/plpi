from typing import Optional, List
import numpy as np
# from overrides import overrides
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import hamming_loss, average_precision_score
import ml_metrics

'''
class HammingLoss(object):
    def __init__(self) -> None:
        self.predictions = []
        self.gold_labels = []
        self.total_count = 0
    def record(
        self,
        predictions: List[str],
        gold_labels: List[str],
    ):
        self.predictions.append(predictions)
        self.gold_labels.append(gold_labels)
        self.total_count += 1
    def get_metric(self, reset: bool = False):
        if self.total_count > 1e-12:
            labels = set()
            for each_label_list in self.gold_labels:
                for each_label in each_label_list:
                    labels.add(each_label)
            
            for each_label_list in self.predictions:
                for each_label in each_label_list:
                    labels.add(each_label)
            labels = list(labels)
            label_to_id = {}
            for i, each_label in enumerate(labels):
                label_to_id[each_label] = i
            recall = float(self.precision_sum) / float(self.total_count)
        else:
            hamming_loss = 0.0
        if reset:
            self.reset()
        return hamming_loss
    def reset(self):
        self.precision_sum = 0.0
        self.total_count = 0
'''

class MRR(object):
    def __init__(self) -> None:
        self.sumReciprocalRank = 0.0
        self.total_count = 0

    def record(
        self,
        predictions: List[str],
        gold_labels: List[str],
    ):
        for rank, item in enumerate(predictions):
            if item in gold_labels:
                self.sumReciprocalRank += 1.0 / (rank + 1.0)
                break
        self.total_count += 1

    def get_metric(self, reset: bool = False):
        if self.total_count > 1e-12:
            mrr = self.sumReciprocalRank / self.total_count
        else:
            mrr = 0
        if reset:
            self.reset()
        return mrr

    def reset(self):
        self.sumReciprocalRank = 0.0
        self.total_count = 0


class MAP(object):
    def __init__(self, k=10) -> None:
        self.k = k
        self.sumMAP = 0.0
        self.total_count = 0

    def record(
        self,
        predictions: List[str],
        gold_labels: List[str],
    ):
        if len(predictions) > 0:
            self.sumMAP += ml_metrics.mapk(gold_labels, predictions, self.k)
        self.total_count += 1

    def get_metric(self, reset: bool = False):
        if self.total_count > 1e-12:
            map = self.sumMAP / self.total_count
        else:
            map = 0
        if reset:
            self.reset()
        return map

    def reset(self):
        self.sumMAP = 0.0
        self.total_count = 0

class MacroPRF(object):
    def __init__(self) -> None:
        # (TP, FP, FN, TN)
        self.predictions = []
        self.gold_labels = []
        self.total_count = 0

    def record(
        self,
        prediction: str,
        gold_label: str,
    ):
        self.predictions.append(prediction)
        self.gold_labels.append(gold_label)
        self.total_count += 1

    def get_metric(self, reset: bool = False):
        """
        # Returns
        """
        labels = set()
        for each_label in self.gold_labels:
            labels.add(each_label)
        for each_label in self.predictions:
            labels.add(each_label)

        labels = list(labels)
        if self.total_count > 1e-12:
            p_macro, r_macro, f_macro, support_macro = \
                precision_recall_fscore_support(
                    y_true=self.gold_labels,
                    y_pred=self.predictions,
                    labels=labels, average='macro')
        else:
            p_macro, r_macro, f_macro = 0, 0, 0
        if reset:
            self.reset()
        return p_macro, r_macro, f_macro

    def reset(self):
        self.predictions = []
        self.gold_labels = []
        self.total_count = 0


class MicroPRF(object):
    def __init__(self) -> None:
        # (TP, FP, FN, TN)
        self.predictions = []
        self.gold_labels = []
        self.total_count = 0

    def record(
        self,
        prediction: str,
        gold_label: str,
    ):
        self.predictions.append(prediction)
        self.gold_labels.append(gold_label)
        self.total_count += 1

    def get_metric(self, reset: bool = False):
        """
        # Returns
        """
        labels = set()
        for each_label in self.gold_labels:
            labels.add(each_label)
        for each_label in self.predictions:
            labels.add(each_label)

        labels = list(labels)
        if self.total_count > 1e-12:
            p_micro, r_micro, f_micro, support_micro = \
                precision_recall_fscore_support(
                    y_true=self.gold_labels,
                    y_pred=self.predictions,
                    labels=labels, average='micro')
        else:
            p_micro, r_micro, f_micro = 0, 0, 0
        if reset:
            self.reset()
        return p_micro, r_micro, f_micro

    def reset(self):
        self.predictions = []
        self.gold_labels = []
        self.total_count = 0


class RecAccuracy(object):
    def __init__(self, top_k: int = 1) -> None:
        if top_k <= 0:
            raise Exception("top_k passed to RecAccuracy must be > 0")
        self._top_k = top_k
        self.accuracy_sum = 0.0
        self.total_count = 0

    def record(
        self,
        predictions: List[str],
        gold_labels: List[str],
    ):
        predictions_len = len(predictions)
        gold_labels_len = len(gold_labels)
        intersection_set = set(predictions).intersection(set(gold_labels))
        union_set = set(predictions).union(set(gold_labels))
        
        if len(union_set) > np.finfo(float).eps:
            self.accuracy_sum += len(intersection_set) / len(union_set)
        self.total_count += 1

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated precision.
        """
        if self.total_count > 1e-12:
            recall = float(self.accuracy_sum) / float(self.total_count)
        else:
            recall = 0.0
        if reset:
            self.reset()
        return recall

    def reset(self):
        self.accuracy_sum = 0.0
        self.total_count = 0

class RecPrecision(object):
    def __init__(self, top_k: int = 1) -> None:
        if top_k <= 0:
            raise Exception("top_k passed to RecPrecision must be > 0")
        self._top_k = top_k
        self.precision_sum = 0.0
        self.total_count = 0

    def record(
        self,
        predictions: List[str],
        gold_labels: List[str],
    ):
        predictions_len = len(predictions)
        gold_labels_len = len(gold_labels)
        intersection_set = set(predictions).intersection(set(gold_labels))
        
        if len(predictions) > np.finfo(float).eps:
            self.precision_sum += len(intersection_set) / len(predictions)
        self.total_count += 1

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated precision.
        """
        if self.total_count > 1e-12:
            recall = float(self.precision_sum) / float(self.total_count)
        else:
            recall = 0.0
        if reset:
            self.reset()
        return recall

    def reset(self):
        self.precision_sum = 0.0
        self.total_count = 0

class RecRecall(object):
    def __init__(self, top_k: int = 1) -> None:
        if top_k <= 0:
            raise Exception("top_k passed to RecRecall must be > 0")
        self._top_k = top_k
        self.recall_sum = 0.0
        self.total_count = 0

    def record(
        self,
        predictions: List[str],
        gold_labels: List[str],
    ):
        predictions_len = len(predictions)
        gold_labels_len = len(gold_labels)
        intersection_set = set(predictions).intersection(set(gold_labels))
        
        if len(gold_labels) > np.finfo(float).eps:
            self.recall_sum += len(intersection_set) / len(gold_labels) # gold_labels_len
        self.total_count += 1

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated precision.
        """
        if self.total_count > 1e-12:
            recall = float(self.recall_sum) / float(self.total_count)
        else:
            recall = 0.0
        if reset:
            self.reset()
        return recall

    def reset(self):
        self.recall_sum = 0.0
        self.total_count = 0