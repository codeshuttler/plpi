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
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.data.samplers import BucketBatchSampler
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.common import JsonDict
from allennlp.models import Model
from allennlp.data.samplers import WeightedRandomSampler

from .imbalanced import ImbalancedDatasetSampler

class TextCNN(Model):
    def __init__(self, embedding_size, embeddings, vocab: Vocabulary, dropout=0.5, device='cpu'):
        super(TextCNN, self).__init__(vocab)

        self.vocab = vocab
        self.device = device

        in_channels = 1
        kernel_num = 100
        kernel_size = [3, 4, 5]

        self.embeddings = embeddings
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels, kernel_num, (K, embedding_size)).to(device) for K in kernel_size])
        self.dropout = nn.Dropout(p=dropout).to(device)
        self.fc1 = nn.Linear(len(kernel_size)*kernel_num, vocab.get_vocab_size('labels')).to(device)

        self.loss_function = torch.nn.BCELoss()
        # self.loss_function = torch.nn.MultiLabelMarginLoss()
        # self.loss_function = torch.nn.MultiLabelSoftMarginLoss()
        # self.loss_function = torch.nn.BCEWithLogitsLoss()

        self._precision = RecPrecision(top_k=3)
        self._recall = RecRecall(top_k=3)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, text, labels = None):
        x = self.embeddings(text) # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        # x = torch.cat(x, 1)
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs1], 1)
        x = self.dropout(x)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        # x = self.dropout(x)  # (N, len(Ks)*Co)
        # logit = F.softmax(self.fc1(x), dim=0)  # (N, C)
        logits = torch.sigmoid(self.fc1(x))  # (N, C)

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

class CNNModel(object):
    def __init__(self, embedding_path, embed_size=50, device=0):
        self.embedding_path = embedding_path
        self.embed_size = embed_size
        self.device = device
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

        self.model = TextCNN(EMBEDDING_DIM, word_embeddings, vocab, 0.5, device)
        self.model.to(device)

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=1, momentum=0.01)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        
        def get_power_label(dataset, i):
            sorted_labels = sorted(dataset[i]['labels'].labels)
            return "_".join(sorted_labels)

        weights = make_weights_for_balanced_classes(train_dataset, vocab.get_index_to_token_vocabulary('labels'))
        # sampler = BucketBatchSampler(train_dataset, batch_size=4, sorting_keys=["text", "labels"])
        # sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=get_power_label)
        # sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=lambda dataset, i: dataset[i]['labels'].labels[0]),
        # sampler=WeightedRandomSampler(weights, len(train_dataset)*2)
        # print('weights', len(weights))
        # print('train_dataset', len(train_dataset))
        data_loader = PyTorchDataLoader(train_dataset,
            # sampler=sampler,
            batch_size=32)
        valid_data_loader = PyTorchDataLoader(valid_dataset,
            batch_size=256,
            shuffle=True)
        # iterator.index_with(vocab)
        trainer = GradientDescentTrainer(model=self.model,
                        optimizer=optimizer,
                        data_loader=data_loader,
                        validation_data_loader=valid_data_loader,
                        # patience=5,
                        num_epochs=20,
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