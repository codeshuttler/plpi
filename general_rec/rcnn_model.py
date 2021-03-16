import os
import tempfile
import random
import re

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn


from .dataset_readers.fasttext_dataset import FasttextDatasetReader
from .predictors.MultiLabelClassifierPredictor import MultiLabelClassifierPredictor
from .metrics.rec_precision import RecPrecision
from .metrics.rec_recall import RecRecall
import numpy as np

from typing import Dict

from allennlp.modules.token_embedders import Embedding
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.samplers import WeightedRandomSampler
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.data.samplers import BucketBatchSampler
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.common import JsonDict
from allennlp.models import Model

from .imbalanced import ImbalancedDatasetSampler

class RCNN(Model):
    def __init__(self, embed_size, embeddings, vocab: Vocabulary, dropout=0.5, device='cpu'):
        super(RCNN, self).__init__(vocab)

        self.hidden_size = 64
        self.hidden_layers = 1
        self.hidden_size_linear = 64
        self.output_size = vocab.get_vocab_size('labels')
        self.embed_size = embed_size
        # self.vocab_size = vocab.get_vocab_size('text')
        self.dropout = dropout
        self.device = device

        self._precision = RecPrecision(top_k=2)
        self._recall = RecRecall(top_k=2)
        
        # Embedding Layer
        self.embeddings = embeddings
        
        # Bi-directional LSTM for RCNN
        self.lstm = nn.LSTM(input_size = self.embed_size,
                            hidden_size = self.hidden_size,
                            num_layers = self.hidden_layers,
                            # dropout = self.dropout,
                            batch_first=True,
                            bidirectional = True).to(device)
        
        # self.dropout = nn.Dropout(self.dropout).to(device)
        
        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(
            self.embed_size + 2*self.hidden_size,
            self.hidden_size_linear
        ).to(device)
        
        # Tanh non-linearity
        self.tanh = nn.Tanh().to(device)
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.hidden_size_linear,
            self.output_size
        ).to(device)
        
        self.loss_function = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, text, labels = None):
        # x.shape = (seq_len, batch_size)
        embedded_sent = self.embeddings(text)
        # embedded_sent = embedded_sent.permute(1, 0, 2)
        # embedded_sent.shape = (seq_len, batch_size, embed_size)

        lstm_out, (h_n,c_n) = self.lstm(embedded_sent)
        # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)
        
        input_features = torch.cat([lstm_out,embedded_sent], 2) # .permute(1,0,2)
        # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)
        
        linear_output = self.tanh(
            self.W(input_features)
        )
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)
        
        linear_output = linear_output.permute(0,2,1) # Reshaping fot max_pool
        
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        # max_out_features.shape = (batch_size, hidden_size_linear)
        
        # max_out_features = self.dropout(max_out_features)
        # logits = torch.sigmoid(self.fc(max_out_features))  # (N, C)
        logits = self.softmax(self.fc(max_out_features))

        output = {"logits": logits}
        if labels is not None:
            self._precision(logits, labels)
            self._recall(logits, labels)
            output["loss"] = self.loss_function(logits, labels.float())
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision = self._precision.get_metric(reset)
        recall = self._recall.get_metric(reset)
        beta2 = 1
        if beta2 * precision + recall < 1e-6:
            f1score = 0
        else:
            f1score = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        return {
            "_precision": precision,
            "_recall": recall,
            "f1score": f1score,
            }


def dataset_to_fasttext_dataset(dataset, f):
    for each_data in dataset:
        labels = each_data['labels']
        for each_label in labels:
            f.write('__label__' + each_label.replace(' ', '<s>') + ' ')
        
        text = each_data['title'] + ' ' + each_data['description']
        # 预处理
        text =text.replace('\n', '<NEWLINE>')

        # 驼峰分词
        # 匹配正则，匹配小写字母和大写字母的分界位置
        p = re.compile(r'([a-z]|\d)([A-Z])')
        # 这里第二个参数使用了正则分组的后向引用
        text = re.sub(p, r'\1 \2', text).lower()
        text = text.lower()

        # 干，里面有__label__标记，标签就不对
        text = text.replace('__label__', '')

        f.write(text + '\n')


def make_weights_for_balanced_classes(data, label_dict):
    # label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
    label_to_id = {}
    for id, label in label_dict.items():
        label_to_id[label] = id

    nclasses = len(label_dict)
    count = [0] * nclasses
    for item in data:
        for each_label in item['labels'].labels:
            count[label_to_id[each_label]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        if float(count[i]) == 0:
            weight_per_class[i] = 0
            print(f'class {i} is zero count.')
        else:
            weight_per_class[i] = float(count[i])/N

    weight = [0] * len(data)
    for idx, val in enumerate(data):
        cur_weights = []
        for each_label in val['labels'].labels:
            cur_weights.append(weight_per_class[label_to_id[each_label]])
        weight[idx] = 1.0 / min(cur_weights)

    return weight


def oversampling(dataset):
    def get_power_label(data):
        sorted_labels = sorted(data['labels'])
        return "_".join(sorted_labels)
    
    power_labels = []
    for i, data in enumerate(dataset):
        power_labels.append(get_power_label(data))
    
    # count
    counter = {}
    for i, data in enumerate(dataset):
        power_label = power_labels[i]
        if power_label not in counter:
            counter[power_label] = []
        counter[power_label].append(i)

    max_count = 10
    balanced_dataset = []
    for key, value in counter.items():
        for i in value:
            balanced_dataset.append(dataset[i])
        if len(value) > max_count:
            continue
        left_num = max_count - len(value)

        for oversample_i in range(left_num):
            i = random.choice(value)
            balanced_dataset.append(dataset[i])
    random.shuffle(balanced_dataset)
    return balanced_dataset

class RCNNModel(object):
    def __init__(self, embedding_path, embed_size=50, device=0):
        self.embedding_path = embedding_path
        self.embed_size=embed_size
        self.device=device
        self.model_path = None
        self.model = None

    def train(self, dataset):
        random.shuffle(dataset)
        split_index = int(0.999 * len(dataset))
        train_lines = dataset[:split_index]
        valid_lines = dataset[split_index:]

        if self.device is None:
            cuda_device = 0
        else:
            cuda_device = self.device

        if torch.cuda.is_available():
            if cuda_device == 0:
                device = "cuda"
            else:
                device = "cuda:" + str(cuda_device)
        else:
            device = "cpu"

        reader = FasttextDatasetReader()
        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8') as fp:
            dataset_to_fasttext_dataset(train_lines, fp)
            train_dataset = reader.read(fp.name)
        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8') as fp:
            dataset_to_fasttext_dataset(valid_lines, fp)
            valid_dataset = reader.read(fp.name)

        vocab = Vocabulary.from_instances(train_dataset, min_count={'tokens': 5})

        train_dataset.index_with(vocab)
        valid_dataset.index_with(vocab)

        EMBEDDING_DIM = self.embed_size
        token_embedding = Embedding.from_params(
                                vocab=vocab,
                                params=Params({'pretrained_file':self.embedding_path,
                                            'embedding_dim' : EMBEDDING_DIM,
                                            'trainable': True,
                                            })
                                )
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

        self.model = RCNN(EMBEDDING_DIM, word_embeddings, vocab, 0.8, device)
        self.model.to(device)

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=1, momentum=0.01)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        
        
        # sampler = BucketBatchSampler(train_dataset, batch_size=4, sorting_keys=["text", "labels"])
        # sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=lambda dataset, i: "_".join(dataset[i]['labels'].labels)),
        # sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=lambda dataset, i: dataset[i]['labels'].labels[0]),
        weights = make_weights_for_balanced_classes(train_dataset, vocab.get_index_to_token_vocabulary('labels'))
        # sampler=WeightedRandomSampler(weights, len(train_dataset)*2)

        data_loader = PyTorchDataLoader(
            train_dataset,
            # sampler=sampler,
            batch_size=16
            )
        valid_data_loader = PyTorchDataLoader(valid_dataset, batch_size=256, shuffle=True)
        # iterator.index_with(vocab)
        trainer = GradientDescentTrainer(model=self.model,
                        optimizer=optimizer,
                        data_loader=data_loader,
                        validation_data_loader=valid_data_loader,
                        # patience=5,
                        num_epochs=10,
                        cuda_device=cuda_device,
                        serialization_dir=self.model_path)
        trainer.train()

        self.predictor = MultiLabelClassifierPredictor(self.model, reader)

    def predict(self, text):
        ret = self.predictor.predict_json({'text': text})
        cluster_top_k = []
        for each_label in ret['labels']:
            if each_label.startswith('__label__'):
                cluster_top_k.append(each_label[9:])
            else:
                cluster_top_k.append(each_label)
        cluster_top_k_prop = ret['logits']
        out_labels = tuple(cluster_top_k)
        out_logits = np.array(cluster_top_k_prop)
        return (out_labels, out_logits)

    def predict_batch(self, text_list):
        predict_list = self.predictor.predict_batch_json([{'text': each_text} for each_text in text_list])

        ret = []
        for eact_predict in predict_list:
            cluster_top_k = []
            for each_label in eact_predict['labels']:
                if each_label.startswith('__label__'):
                    cluster_top_k.append(each_label[9:])
                else:
                    cluster_top_k.append(each_label)
            cluster_top_k_prop = eact_predict['logits']
            out_labels = tuple(cluster_top_k)
            out_logits = np.array(cluster_top_k_prop)
            ret.append((out_labels, out_logits))
        return ret