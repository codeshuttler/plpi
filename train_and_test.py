from sklearn.model_selection import KFold
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import numpy as np
import argparse
import json
import fasttext
import time
import os
import sys
import re
import logging
import tqdm
import tempfile
from typing import Dict, List

import joblib
# import Levenshtein
from sentence_transformers import SentenceTransformer

import torch

import EmbedMetrics

from general_rec.dataset_readers.github_dataset import GitHubDatasetReader
from general_rec.models.textcnn_classifier import TextCNNClassifier
from general_rec.predictors import GitHubLabelPredictor, MultiLabelClassifierPredictor

from general_rec.cnn_model import TextCNN, CNNModel
from general_rec.rcnn_model import RCNN, RCNNModel
from general_rec.lstm_model import LSTM, BiLSTMModel
from general_rec.fasttext_model import FasttextModel
from general_rec.knn_model import KNNModel
from general_rec.svm_model import SVMModel

from allennlp.common import JsonDict


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='train_and_test.log', level=logging.DEBUG, format=LOG_FORMAT)

# 过滤无法预测的标签
bad_labels = ['invalid', 'valid', 'stale', 'version', 'activity', 'triage',
    'good first issue', 'priority', 'wontfix', 'p0', 'p1', 'p2', 'p3', 'p4', 'status', 'resolved',
    'closed', 'pri', 'critical', 'external', 'reply', 'outdate', 'v0', 'v1', 'v2', 'v3', 'v4', 'branch',
    'done', 'approve', 'accept', 'confirm', 'block', 'duplicate', '1.', '0.', 'release', 'easy', 'hard',
    'archive', 'fix', 'lock', 'regression', 'assign', 'verified', 'medium', 'high', 'affect', 'star', 'progress']

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def read_dataset(dataset_path, projects_info_path):
    project_infos = {}
    with open(projects_info_path, 'r', encoding='utf-8') as f:
        for each_line in f:
            json_obj = json.loads(each_line)
            project_infos[json_obj['project']] = json_obj

    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for each_line in f:
            json_obj = json.loads(each_line)
            dataset.append({
                'title': json_obj['title'],
                'description': json_obj['description'],
                'labels': json_obj['labels'],
                'project': json_obj['project'],
                'project_labels': project_infos[json_obj['project']]['labels'],
            })

    return dataset


def embedding_dataset(dataset, embedding_model, clustering_model):
    new_dataset = []
    labels_cluster = {}
    for each_data in tqdm.tqdm(iterable=dataset):
        label_set = set()
        not_in_set_labels = []
        for each_label in each_data['labels']:
            if each_label in labels_cluster:
                label_set.add(labels_cluster[each_label])
            else:
                not_in_set_labels.append(each_label)

        if len(not_in_set_labels) != 0:
            labels_vector = embedding_model.encode(not_in_set_labels, show_progress_bar=False)
            cluster_index = clustering_model.predict(labels_vector)
            for i in range(len(not_in_set_labels)):
                labels_cluster[not_in_set_labels[i]] = cluster_index[i]
                label_set.add(cluster_index[i])

        labels = []
        for each_label in label_set:
            labels.append(str(each_label))
        
        new_dataset.append(
            {
                'title': each_data['title'],
                'description': each_data['description'],
                'labels': labels,
                'project': each_data['project'],
                'project_labels': each_data['project_labels'],
            }
        )
    return new_dataset

def get_all_labels(train_dataset, field='labels'):
    labels = set()
    for each_data in train_dataset:
        for each_label in each_data[field]: # not project_labels
            labels.add(each_label)
    return list(labels)

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

        f.write(text + '\n')

class ClusterPredictor(object):
    def __init__(self, model, embedding_model, clustering_model, top_k):
        self.model = model
        self.embedding_model = embedding_model
        self.clustering_model = clustering_model
        self.top_k = top_k
        self.label_embed_cache = {}

    def pre_compute_labels(self, labels):
        print('pre_compute_labels:')
        embeds = self.embedding_model.encode(labels, show_progress_bar=True)
        for i, each_label in enumerate(labels):
            if each_label not in self.label_embed_cache:
                self.label_embed_cache[each_label] = embeds[i]
    
    def predict(self, data:Dict) -> Dict:
        text = data['title'] + ' ' + data['description']
        text = text.replace('\n', ' <NEWLINE> ')
        project = data['project']
        project_labels = data['project_labels']

        # 过滤project labels
        filtered_project_labels = set()
        for each_project_label in project_labels:
            if all(each_bad_label not in each_project_label for each_bad_label in bad_labels):
                filtered_project_labels.add(each_project_label)
        project_labels = list(filtered_project_labels)

        prediction = self.model.predict(text)

        # 对概率进行排序
        prediction_1_index = np.argsort(prediction[1])
        prediction_1_index = np.array(list(reversed(prediction_1_index)))
        prediction_1 = prediction[1][prediction_1_index]
        prediction_1 = np.clip(prediction_1, 0.0, 1.0)
        prediction_0 = np.array(prediction[0])[prediction_1_index]

        prediction = (prediction_0, prediction_1)
        # if len(prediction[0]) > self.top_k:
        #    prediction = (prediction_0[:self.top_k], prediction_1[:self.top_k])

        cluster_top_k = []
        for each_label in prediction[0]:
            if each_label.startswith('__label__'):
                cluster_top_k.append(each_label[9:].replace('<s>', ' '))
            else:
                cluster_top_k.append(each_label)
        cluster_top_k_prop = prediction[1]

        # 标签转换为向量
        project_labels_embedding = np.zeros((len(project_labels), 768))
        for i, each_project_label in enumerate(project_labels):
            if each_project_label in self.label_embed_cache:
                project_labels_embedding[i,:] = self.label_embed_cache[each_project_label]
            else:
                embed = self.embedding_model.encode([each_project_label], show_progress_bar=False)
                project_labels_embedding[i,:] = embed[0]

        # 搜索近似标签
        cluster_top_k_embedding = np.zeros((len(cluster_top_k), 768))
        for i, each_pred in enumerate(cluster_top_k):
            cluster_top_k_embedding[i,:] = self.clustering_model.cluster_centers_[int(each_pred)]

        # (project_labels_embedding, top_k)
        similarity_matrix = cosine_similarity(project_labels_embedding, cluster_top_k_embedding)
        # np.zeros((len(project_labels_embedding),))
        
        # 方案一
        '''
        index_array = np.argmax(similarity_matrix, axis=1)

        raw_prediction_labels = [project_labels[i] for i in range(len(project_labels_embedding))]
        raw_prediction_logits = [cluster_top_k_prop[index_array[i]] * similarity_matrix[i, index_array[i]] for i in range(len(project_labels_embedding))]
        # raw_prediction_logits = [cluster_top_k_prop[index_array[i]] for i in range(len(project_labels_embedding))]
        prediction_list = [(raw_prediction_labels[i], raw_prediction_logits[i]) for i in range(len(project_labels_embedding))]
        '''

        # 方案二
        '''
        similarity_matrix_sum = similarity_matrix.sum(axis=1, keepdims=True)
        similarity_matrix /= similarity_matrix_sum

        raw_prediction_labels = [project_labels[i] for i in range(len(project_labels_embedding))]
        raw_prediction_logits = [0] * len(project_labels_embedding)
        for i in range(len(project_labels_embedding)):
            for each_k in range(self.top_k):
                raw_prediction_logits[i] += cluster_top_k_prop[each_k] * similarity_matrix[i, each_k]
            # raw_prediction_logits[i] /= self.top_k
        # scaler = MinMaxScaler()
        # raw_prediction_logits_new = scaler.fit_transform(np.array(raw_prediction_logits).reshape(-1, 1))

        prediction_list = [(raw_prediction_labels[i], raw_prediction_logits[i]) for i in range(len(project_labels_embedding))]
        '''

        # 方案三
        raw_prediction_labels = []
        raw_prediction_logits = []
        index_array = np.argmax(similarity_matrix, axis=0)
        for i in range(len(cluster_top_k)):
            this_label = project_labels[index_array[i]]
            if this_label not in raw_prediction_labels:
                raw_prediction_labels.append(this_label)
                raw_prediction_logits.append(cluster_top_k_prop[i])
        prediction_list = [(raw_prediction_labels[i], raw_prediction_logits[i]) for i in range(len(raw_prediction_labels))]

        # return {"instance": prediction, "all_labels": all_labels}
        return prediction_list

    
    def predict_batch(self, data:List[Dict]) -> List[Dict]:

        texts = []
        for each_data in data:
            text = each_data['title'] + ' ' + each_data['description']
            text = text.replace('\n', ' <NEWLINE> ')
            
            texts.append(text)

        predictions = self.model.predict_batch(texts)

        ret = []
        for i, each_data in enumerate(data):
            prediction = predictions[i]
            project = each_data['project']
            project_labels = each_data['project_labels']

            # 过滤project labels
            filtered_project_labels = set()
            for each_project_label in project_labels:
                if all(each_bad_label not in each_project_label for each_bad_label in bad_labels):
                    filtered_project_labels.add(each_project_label)
            project_labels = list(filtered_project_labels)

            # 对概率进行排序
            prediction_1_index = np.argsort(prediction[1])
            prediction_1_index = np.array(list(reversed(prediction_1_index)))
            prediction_1 = prediction[1][prediction_1_index]
            prediction_1 = np.clip(prediction_1, 0.0, 1.0)
            prediction_0 = np.array(prediction[0])[prediction_1_index]

            prediction = (prediction_0, prediction_1)
            # if len(prediction[0]) > self.top_k:
            #    prediction = (prediction_0[:self.top_k], prediction_1[:self.top_k])

            cluster_top_k = []
            for each_label in prediction[0]:
                if each_label.startswith('__label__'):
                    cluster_top_k.append(each_label[9:].replace('<s>', ' '))
                else:
                    cluster_top_k.append(each_label)
            cluster_top_k_prop = prediction[1]

            # 标签转换为向量
            project_labels_embedding = np.zeros((len(project_labels), 768))
            for i, each_project_label in enumerate(project_labels):
                if each_project_label in self.label_embed_cache:
                    project_labels_embedding[i,:] = self.label_embed_cache[each_project_label]
                else:
                    embed = self.embedding_model.encode([each_project_label], show_progress_bar=False)
                    project_labels_embedding[i,:] = embed[0]

            # 搜索近似标签
            cluster_top_k_embedding = np.zeros((len(cluster_top_k), 768))
            for i, each_pred in enumerate(cluster_top_k):
                cluster_top_k_embedding[i,:] = self.clustering_model.cluster_centers_[int(each_pred)]

            # (project_labels_embedding, top_k)
            similarity_matrix = cosine_similarity(project_labels_embedding, cluster_top_k_embedding)

            # 方案三
            raw_prediction_labels = []
            raw_prediction_logits = []
            index_array = np.argmax(similarity_matrix, axis=0)
            for i in range(len(cluster_top_k)):
                this_label = project_labels[index_array[i]]
                if this_label not in raw_prediction_labels:
                    raw_prediction_labels.append(this_label)
                    raw_prediction_logits.append(cluster_top_k_prop[i])
            prediction_list = [(raw_prediction_labels[i], raw_prediction_logits[i]) for i in range(len(raw_prediction_labels))]

            ret.append(prediction_list)
        return ret

class DirectPredictor(object):
    def __init__(self, model, label_number):
        self.model = model
        self.label_number = label_number

    def predict(self, data:Dict) -> Dict:
        text = data['title'] + ' ' + data['description']
        text = text.replace('\n', ' <NEWLINE> ')
        project = data['project']
        project_labels = data['project_labels']

        prediction = self.model.predict(text)

        # 对概率进行排序
        prediction_1_index = np.argsort(prediction[1])[::-1]
        prediction_1 = prediction[1][prediction_1_index]
        prediction_0 = np.array(prediction[0])[prediction_1_index]
        prediction = (prediction_0, prediction_1)

        cluster_top_k = []
        for each_label in prediction[0]:
            if each_label.startswith('__label__'):
                cluster_top_k.append(each_label[9:].replace('<s>', ' '))
            else:
                cluster_top_k.append(each_label)
        cluster_top_k_prop = prediction[1]

        prediction_list = [(cluster_top_k[i], cluster_top_k_prop[i]) for i in range(len(cluster_top_k))]
        return prediction_list

    def predict_batch(self, data:List[Dict]) -> List[Dict]:
        texts = []

        for each_data in data:
            text = each_data['title'] + ' ' + each_data['description']
            text = text.replace('\n', ' <NEWLINE> ')
            project = each_data['project']
            project_labels = each_data['project_labels']
            texts.append(text)

        prediction = self.model.predict_batch(texts)

        ret = []

        for each_predict in prediction:
            # 对概率进行排序
            prediction_1_index = np.argsort(each_predict[1])[::-1]
            prediction_1 = each_predict[1][prediction_1_index]
            prediction_0 = np.array(each_predict[0])[prediction_1_index]
            each_predict = (prediction_0, prediction_1)

            cluster_top_k = []
            for each_label in each_predict[0]:
                if each_label.startswith('__label__'):
                    cluster_top_k.append(each_label[9:].replace('<s>', ' '))
                else:
                    cluster_top_k.append(each_label)
            cluster_top_k_prop = each_predict[1]

            prediction_list = [(cluster_top_k[i], cluster_top_k_prop[i]) for i in range(len(cluster_top_k))]

            ret.append(prediction_list)
        return ret

classic_labels = {
            'bug': ['bug', 'defect', 'crash', 'leak'],
            'feature': ['feature'],
            'enhancement': ['enhance', 'improve'],
            'help wanted': ['help', 'support', 'guid'],
            'question': ['question', 'query', 'howto'],
            'documentation': ['documentation', 'document'] # 'doc'
        }
def personalized_labels_to_abstract_labels(label):
    for key, value in classic_labels.items():
        for each_key_label in value:
            if each_key_label in label.lower():
                return key
    return None

def filter_classic_labels(test_dataset):
    new_dataset = []
    for each_data in test_dataset:
        labels = []
        abstract_labels = []
        for key, value in classic_labels.items():
            for each_label in each_data['labels']:
                for each_key_label in value:
                    if each_key_label in each_label.lower():
                        labels.append(each_label)
                        abstract_labels.append(key)
                        break
        if len(labels) > 0:
            new_dataset.append({
                    'project': each_data['project'],
                    'title': each_data['title'],
                    'description': each_data['description'],
                    'code': None,
                    'labels': labels,
                    'abstract_labels': abstract_labels,
                    'project': each_data['project'],
                    'project_labels': each_data['project_labels'],
                })
    return new_dataset

def output_cluster(filename, num_clusters, clustering_model, labels):
    with open(filename, 'w', encoding='utf-8') as outf:
        collect_labels = []
        for i in range(num_clusters):
            collect_labels.append([])

        for i, each_cluster in enumerate(clustering_model.labels_):
            collect_labels[each_cluster].append(i)

        for each_cluster, each_sample_array in enumerate(collect_labels):
            outf.write(f'Cluster {each_cluster} :\n')
            for each_sample in each_sample_array:
                outf.write(f'{labels[each_sample]}\n')

            outf.write('\n')

def make_weights_for_balanced_classes(data, nclasses):
    count = [0] * nclasses
    for item in data:
        for each_label in item['labels']:
            count[int(each_label)] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        if float(count[i]) == 0:
            weight_per_class[i] = 0
            print(f'class {i} is zero count.')
        else:
            weight_per_class[i] = N/float(count[i])
    weight = [0] * len(data)
    for idx, val in enumerate(data):
        for each_label in item['labels']:
            weight[idx] += weight_per_class[int(each_label)]
    return weight

def save_label_report(all_test_labels, label_target_matrix, label_predict_matrix, output_path):
    all_labels_confusion_matrix = multilabel_confusion_matrix(label_target_matrix, label_predict_matrix)
    df_json = {
        'label': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'TN': [],
        'FN': [],
        'TP': [],
        'FP': []
    }
    # TN 00, FN 10, TP 11, FP 01
    for i, each_matrix in enumerate(all_labels_confusion_matrix):
        df_json['label'].append(all_test_labels[i])
        if each_matrix[0,0] + each_matrix[0,1] + each_matrix[1,0] + each_matrix[1,1] != 0:
            acc = (each_matrix[1,1] + each_matrix[0, 0])/(each_matrix[0,0] + each_matrix[0,1] + each_matrix[1,0] + each_matrix[1,1])
        else:
            acc = 0
        df_json['accuracy'].append(acc)
        if each_matrix[1,1] + each_matrix[0,1] != 0:
            prec = (each_matrix[1,1])/(each_matrix[1,1] + each_matrix[0,1])
        else:
            prec = 0
        df_json['precision'].append(prec) # 等于0
        if each_matrix[1,1] + each_matrix[1,0] != 0:
            rec = (each_matrix[1,1])/(each_matrix[1,1] + each_matrix[1,0])
        else:
            rec = 0
        df_json['recall'].append(rec)

        if prec+rec < 1e-6:
            f1 = 0
        else:
            f1 = 2*(prec*rec)/(prec+rec)
        df_json['f1'].append(f1)

        df_json['TN'].append(each_matrix[0,0])
        df_json['FN'].append(each_matrix[1,0])
        df_json['TP'].append(each_matrix[1,1])
        df_json['FP'].append(each_matrix[0,1])

    TP_sum = np.sum(df_json['TP'])
    TN_sum = np.sum(df_json['TN'])
    FP_sum = np.sum(df_json['FP'])
    FN_sum = np.sum(df_json['FN'])
    # macro metric
    df_json['label'].append('macro metrics')
    df_json['accuracy'].append(np.mean(df_json['accuracy']))
    df_json['precision'].append(np.mean(df_json['precision']))
    df_json['recall'].append(np.mean(df_json['recall']))
    df_json['f1'].append(np.mean(df_json['f1']))
    df_json['TN'].append(TN_sum)
    df_json['FN'].append(FN_sum)
    df_json['TP'].append(TP_sum)
    df_json['FP'].append(FP_sum)

    # micro metric
    df_json['label'].append('micro metrics')
    if TP_sum+TN_sum+FP_sum+FN_sum < 1e-12:
        micro_acc = 0
    else:
        micro_acc = (TP_sum+TN_sum)/(TP_sum+TN_sum+FP_sum+FN_sum)
    df_json['accuracy'].append(micro_acc)
    if TP_sum+FP_sum < 1e-12:
        micro_prec = 0
    else:
        micro_prec = TP_sum/(TP_sum+FP_sum)
    df_json['precision'].append(micro_prec)
    if TP_sum+FN_sum < 1e-12:
        micro_rec = 0
    else:
        micro_rec = TP_sum/(TP_sum+FN_sum)
    df_json['recall'].append(micro_rec)
    if micro_prec+micro_rec < 1e-12:
        micro_f1=0
    else:
        micro_f1=2*micro_prec*micro_rec/(micro_prec+micro_rec)
    df_json['f1'].append(micro_f1)
    df_json['TN'].append(TN_sum)
    df_json['FN'].append(FN_sum)
    df_json['TP'].append(TP_sum)
    df_json['FP'].append(FP_sum)

    label_individual_metrics_df = pd.DataFrame(df_json)
    label_individual_metrics_df.to_csv(output_path, index=False, sep=',')

def test(predictor, test_dataset, all_metrics, abstract=False, out_path=None, threshold=0.5):
    # quick map
    all_test_labels = get_all_labels(test_dataset)
    label_to_id = {}
    id_to_label = {}
    for i, each_label in enumerate(all_test_labels):
        label_to_id[each_label] = i
        id_to_label[i] = each_label
    # 加载模型并预测
    recall_1 = EmbedMetrics.RecRecall(top_k=1)
    recall_2 = EmbedMetrics.RecRecall(top_k=2)
    recall_3 = EmbedMetrics.RecRecall(top_k=3)
    recall_4 = EmbedMetrics.RecRecall(top_k=4)
    recall_5 = EmbedMetrics.RecRecall(top_k=5)

    precision_1 = EmbedMetrics.RecPrecision(top_k=1)
    precision_2 = EmbedMetrics.RecPrecision(top_k=2)
    precision_3 = EmbedMetrics.RecPrecision(top_k=3)
    precision_4 = EmbedMetrics.RecPrecision(top_k=4)
    precision_5 = EmbedMetrics.RecPrecision(top_k=5)

    accuracy_1 = EmbedMetrics.RecAccuracy(top_k=1)
    accuracy_2 = EmbedMetrics.RecAccuracy(top_k=2)
    accuracy_3 = EmbedMetrics.RecAccuracy(top_k=3)
    accuracy_4 = EmbedMetrics.RecAccuracy(top_k=4)
    accuracy_5 = EmbedMetrics.RecAccuracy(top_k=5)

    example_accuracy = EmbedMetrics.RecAccuracy()
    example_precision = EmbedMetrics.RecPrecision()
    example_recall = EmbedMetrics.RecRecall()

    mrr1 = EmbedMetrics.MRR()
    mrr2 = EmbedMetrics.MRR()
    mrr3 = EmbedMetrics.MRR()
    mrr4 = EmbedMetrics.MRR()
    mrr5 = EmbedMetrics.MRR()
    mrr = EmbedMetrics.MRR()

    map1 = EmbedMetrics.MAP(k=1)
    map2 = EmbedMetrics.MAP(k=2)
    map3 = EmbedMetrics.MAP(k=3)
    map4 = EmbedMetrics.MAP(k=4)
    map5 = EmbedMetrics.MAP(k=5)
    map = EmbedMetrics.MAP(k=10)

    # 标签预测矩阵
    label_predict_matrix = np.zeros((len(test_dataset), len(all_test_labels)))
    label_target_matrix = np.zeros((len(test_dataset), len(all_test_labels)))

    for each_data_index, each_data in enumerate(tqdm.tqdm(iterable=test_dataset)):
        prediction = predictor.predict(
                    {
                        'title': each_data['title'],
                        'description': each_data['description'],
                        'project': each_data['project'],
                        'project_labels': each_data['project_labels']
                    }
                )

        prediction = sorted(prediction, key=lambda x: x[1], reverse=True)
        # 大于50%
        prediction = [each_pred[0] for each_pred in prediction if each_pred[1] > threshold]
        
        # real and prediction
        real = each_data['labels']

        if abstract:
            real = each_data['abstract_labels']
            abstract_prediction  = []
            for each_pred in prediction:
                ret = personalized_labels_to_abstract_labels(each_pred)
                if ret is not None:
                    abstract_prediction.append(ret)
            prediction = list(set(abstract_prediction))

        # 没有真实标签，跳过
        if len(real) <= 0:
            continue

        pred1 = prediction[:1]
        pred2 = prediction[:2]
        pred3 = prediction[:3]
        pred4 = prediction[:4]
        pred5 = prediction[:5]

        # 统计metrics
        recall_1.record(pred1, real)
        recall_2.record(pred2, real)
        recall_3.record(pred3, real)
        recall_4.record(pred4, real)
        recall_5.record(pred5, real)

        precision_1.record(pred1, real)
        precision_2.record(pred2, real)
        precision_3.record(pred3, real)
        precision_4.record(pred4, real)
        precision_5.record(pred5, real)

        accuracy_1.record(pred1, real)
        accuracy_2.record(pred2, real)
        accuracy_3.record(pred3, real)
        accuracy_4.record(pred4, real)
        accuracy_5.record(pred5, real)

        example_accuracy.record(prediction, real)
        example_precision.record(prediction, real)
        example_recall.record(prediction, real)

        mrr1.record(pred1, real)
        mrr2.record(pred2, real)
        mrr3.record(pred3, real)
        mrr4.record(pred4, real)
        mrr5.record(pred5, real)
        mrr.record(prediction, real)
        
        map1.record(pred1, real)
        map2.record(pred2, real)
        map3.record(pred3, real)
        map4.record(pred4, real)
        map5.record(pred5, real)
        map.record(prediction, real)

        # 记录每个标签的准确率
        def labels_id_array(labels):
            ids = np.zeros((len(all_test_labels),))
            for each_label in labels:
                if each_label in label_to_id:
                    ids[label_to_id[each_label]] = 1
            return ids
        pred_ids = labels_id_array(prediction)
        label_predict_matrix[each_data_index, :] = pred_ids
        real_ids = labels_id_array(real)
        label_target_matrix[each_data_index, :] = real_ids

    # 保存报告
    save_label_report(
        all_test_labels,
        label_target_matrix,
        label_predict_matrix,
        out_path
        )

    # 所有指标
    all_metrics['recall1'].append(recall_1.get_metric())
    all_metrics['recall2'].append(recall_2.get_metric())
    all_metrics['recall3'].append(recall_3.get_metric())
    all_metrics['recall4'].append(recall_4.get_metric())
    all_metrics['recall5'].append(recall_5.get_metric())

    all_metrics['precision1'].append(precision_1.get_metric())
    all_metrics['precision2'].append(precision_2.get_metric())
    all_metrics['precision3'].append(precision_3.get_metric())
    all_metrics['precision4'].append(precision_4.get_metric())
    all_metrics['precision5'].append(precision_5.get_metric())

    all_metrics['accuracy1'].append(accuracy_1.get_metric())
    all_metrics['accuracy2'].append(accuracy_2.get_metric())
    all_metrics['accuracy3'].append(accuracy_3.get_metric())
    all_metrics['accuracy4'].append(accuracy_4.get_metric())
    all_metrics['accuracy5'].append(accuracy_5.get_metric())

    all_metrics['example_accuracy'].append(example_accuracy.get_metric())
    example_precision_value = example_precision.get_metric()
    all_metrics['example_precision'].append(example_precision_value)
    example_recall_value = example_recall.get_metric()
    all_metrics['example_recall'].append(example_recall_value)
    if example_precision_value+example_recall_value < 1e-6:
        all_metrics['example_f1'].append(0)
    else:
        all_metrics['example_f1'].append(2*(example_precision_value*example_recall_value)/(example_precision_value+example_recall_value))

    all_metrics['mrr1'].append(mrr1.get_metric())
    all_metrics['mrr2'].append(mrr2.get_metric())
    all_metrics['mrr3'].append(mrr3.get_metric())
    all_metrics['mrr4'].append(mrr4.get_metric())
    all_metrics['mrr5'].append(mrr5.get_metric())
    all_metrics['mrr'].append(mrr.get_metric())

    all_metrics['map1'].append(map1.get_metric())
    all_metrics['map2'].append(map2.get_metric())
    all_metrics['map3'].append(map3.get_metric())
    all_metrics['map4'].append(map4.get_metric())
    all_metrics['map5'].append(map5.get_metric())
    all_metrics['map'].append(map.get_metric())

# @profile
def test_batch(predictor, test_dataset, all_metrics, batch_size=256, abstract=False, out_path=None, threshold=0.5):
    # quick map
    all_test_labels = get_all_labels(test_dataset)
    label_to_id = {}
    id_to_label = {}
    for i, each_label in enumerate(all_test_labels):
        label_to_id[each_label] = i
        id_to_label[i] = each_label
    # 加载模型并预测
    recall_1 = EmbedMetrics.RecRecall(top_k=1)
    recall_2 = EmbedMetrics.RecRecall(top_k=2)
    recall_3 = EmbedMetrics.RecRecall(top_k=3)
    recall_4 = EmbedMetrics.RecRecall(top_k=4)
    recall_5 = EmbedMetrics.RecRecall(top_k=5)

    precision_1 = EmbedMetrics.RecPrecision(top_k=1)
    precision_2 = EmbedMetrics.RecPrecision(top_k=2)
    precision_3 = EmbedMetrics.RecPrecision(top_k=3)
    precision_4 = EmbedMetrics.RecPrecision(top_k=4)
    precision_5 = EmbedMetrics.RecPrecision(top_k=5)

    accuracy_1 = EmbedMetrics.RecAccuracy(top_k=1)
    accuracy_2 = EmbedMetrics.RecAccuracy(top_k=2)
    accuracy_3 = EmbedMetrics.RecAccuracy(top_k=3)
    accuracy_4 = EmbedMetrics.RecAccuracy(top_k=4)
    accuracy_5 = EmbedMetrics.RecAccuracy(top_k=5)

    example_accuracy = EmbedMetrics.RecAccuracy()
    example_precision = EmbedMetrics.RecPrecision()
    example_recall = EmbedMetrics.RecRecall()

    mrr1 = EmbedMetrics.MRR()
    mrr2 = EmbedMetrics.MRR()
    mrr3 = EmbedMetrics.MRR()
    mrr4 = EmbedMetrics.MRR()
    mrr5 = EmbedMetrics.MRR()
    mrr = EmbedMetrics.MRR()

    map1 = EmbedMetrics.MAP(k=1)
    map2 = EmbedMetrics.MAP(k=2)
    map3 = EmbedMetrics.MAP(k=3)
    map4 = EmbedMetrics.MAP(k=4)
    map5 = EmbedMetrics.MAP(k=5)
    map = EmbedMetrics.MAP(k=10)

    # 标签预测矩阵
    label_predict_matrix = np.zeros((len(test_dataset), len(all_test_labels)))
    label_target_matrix = np.zeros((len(test_dataset), len(all_test_labels)))

    test_dataset_iter = list(batch(test_dataset, batch_size))
    for each_data_index, each_data_batch in enumerate(tqdm.tqdm(iterable=test_dataset_iter)):
        json_dict_list = []

        for each_data in each_data_batch:
            json_dict_list.append({
                        'title': each_data['title'],
                        'description': each_data['description'],
                        'project': each_data['project'],
                        'project_labels': each_data['project_labels']
                    })
        predictions = predictor.predict_batch(json_dict_list)

        for i, prediction in enumerate(predictions):
            each_data = each_data_batch[i]
            prediction = sorted(prediction, key=lambda x: x[1], reverse=True)
            # 大于50%
            prediction = [each_pred[0] for each_pred in prediction if each_pred[1] > threshold]
            
            # real and prediction
            real = each_data['labels']

            if abstract:
                real = each_data['abstract_labels']
                abstract_prediction  = []
                for each_pred in prediction:
                    ret = personalized_labels_to_abstract_labels(each_pred)
                    if ret is not None:
                        abstract_prediction.append(ret)
                prediction = list(set(abstract_prediction))

            # 没有真实标签，跳过
            if len(real) <= 0:
                continue

            pred1 = prediction[:1]
            pred2 = prediction[:2]
            pred3 = prediction[:3]
            pred4 = prediction[:4]
            pred5 = prediction[:5]

            # 统计metrics
            recall_1.record(pred1, real)
            recall_2.record(pred2, real)
            recall_3.record(pred3, real)
            recall_4.record(pred4, real)
            recall_5.record(pred5, real)

            precision_1.record(pred1, real)
            precision_2.record(pred2, real)
            precision_3.record(pred3, real)
            precision_4.record(pred4, real)
            precision_5.record(pred5, real)

            accuracy_1.record(pred1, real)
            accuracy_2.record(pred2, real)
            accuracy_3.record(pred3, real)
            accuracy_4.record(pred4, real)
            accuracy_5.record(pred5, real)

            example_accuracy.record(prediction, real)
            example_precision.record(prediction, real)
            example_recall.record(prediction, real)

            mrr1.record(pred1, real)
            mrr2.record(pred2, real)
            mrr3.record(pred3, real)
            mrr4.record(pred4, real)
            mrr5.record(pred5, real)
            mrr.record(prediction, real)
            
            map1.record(pred1, real)
            map2.record(pred2, real)
            map3.record(pred3, real)
            map4.record(pred4, real)
            map5.record(pred5, real)
            map.record(prediction, real)

            # 记录每个标签的准确率
            def labels_id_array(labels):
                ids = np.zeros((len(all_test_labels),))
                for each_label in labels:
                    if each_label in label_to_id:
                        ids[label_to_id[each_label]] = 1
                return ids
            pred_ids = labels_id_array(prediction)
            label_predict_matrix[each_data_index*batch_size+i, :] = pred_ids
            real_ids = labels_id_array(real)
            label_target_matrix[each_data_index*batch_size+i, :] = real_ids
    
    # 保存报告
    save_label_report(
        all_test_labels,
        label_target_matrix,
        label_predict_matrix,
        out_path
        )

    # 所有指标
    all_metrics['recall1'].append(recall_1.get_metric())
    all_metrics['recall2'].append(recall_2.get_metric())
    all_metrics['recall3'].append(recall_3.get_metric())
    all_metrics['recall4'].append(recall_4.get_metric())
    all_metrics['recall5'].append(recall_5.get_metric())

    all_metrics['precision1'].append(precision_1.get_metric())
    all_metrics['precision2'].append(precision_2.get_metric())
    all_metrics['precision3'].append(precision_3.get_metric())
    all_metrics['precision4'].append(precision_4.get_metric())
    all_metrics['precision5'].append(precision_5.get_metric())

    all_metrics['accuracy1'].append(accuracy_1.get_metric())
    all_metrics['accuracy2'].append(accuracy_2.get_metric())
    all_metrics['accuracy3'].append(accuracy_3.get_metric())
    all_metrics['accuracy4'].append(accuracy_4.get_metric())
    all_metrics['accuracy5'].append(accuracy_5.get_metric())

    all_metrics['example_accuracy'].append(example_accuracy.get_metric())
    example_precision_value = example_precision.get_metric()
    all_metrics['example_precision'].append(example_precision_value)
    example_recall_value = example_recall.get_metric()
    all_metrics['example_recall'].append(example_recall_value)
    if example_precision_value+example_recall_value < 1e-6:
        all_metrics['example_f1'].append(0)
    else:
        all_metrics['example_f1'].append(2*(example_precision_value*example_recall_value)/(example_precision_value+example_recall_value))

    all_metrics['mrr1'].append(mrr1.get_metric())
    all_metrics['mrr2'].append(mrr2.get_metric())
    all_metrics['mrr3'].append(mrr3.get_metric())
    all_metrics['mrr4'].append(mrr4.get_metric())
    all_metrics['mrr5'].append(mrr5.get_metric())
    all_metrics['mrr'].append(mrr.get_metric())

    all_metrics['map1'].append(map1.get_metric())
    all_metrics['map2'].append(map2.get_metric())
    all_metrics['map3'].append(map3.get_metric())
    all_metrics['map4'].append(map4.get_metric())
    all_metrics['map5'].append(map5.get_metric())
    all_metrics['map'].append(map.get_metric())

def train(params):
    np.random.seed(params['seed'])
    # 读取数据集
    dataset = read_dataset(params['dataset_path'], params['project_info_path'])

    # 预处理数据集中的标签，小写，去除非英文字符
    def preprocess_label(labels):
        ret = []
        for each_label in labels:
            new_label = each_label.lower()
            # new_label = re.sub('[^a-zA-Z]+', ' ', new_label).strip()
            if len(new_label) > 0:
                ret.append(new_label)
        return ret

    preprocess_dataset = []
    for each_data in tqdm.tqdm(iterable=dataset):
        preprocess_dataset.append(
            {
                'title': each_data['title'],
                'description': each_data['description'],
                'labels': preprocess_label(each_data['labels']),
                'project': each_data['project'],
                'project_labels': [each_label.lower() for each_label in each_data['project_labels']],
            }
        )
    dataset = preprocess_dataset

    preprocess_dataset = []
    label_counter = {}
    filter_left_labels = set()
    count_threshold = 50
    # 先统计数量
    print('count labels')
    for each_data in tqdm.tqdm(iterable=dataset):
        for each_label in each_data['labels']:
            if each_label not in label_counter:
                label_counter[each_label] = 0
            label_counter[each_label] += 1
    for each_label, count in label_counter.items():
        if label_counter[each_label] > count_threshold:
            filter_left_labels.add(each_label)
    print(f'left labels: {len(filter_left_labels)}')
    # 过滤低频标签和bad label
    for each_data in tqdm.tqdm(iterable=dataset):
        labels = []
        for each_label in each_data['labels']:
            is_good = True
            for each_bad_label in bad_labels:
                if each_bad_label in each_label:
                    is_good = False
                    break
            if is_good and each_label in label_counter and label_counter[each_label] > count_threshold:
                labels.append(each_label)
        if len(labels) > 0:
            preprocess_dataset.append(
                {
                    'title': each_data['title'],
                    'description': each_data['description'],
                    'labels': labels,
                    'project': each_data['project'],
                    'project_labels': each_data['project_labels'],
                }
            )
    dataset = preprocess_dataset
    if params['word_embed']=='':
        params['word_embed'] = None
    if params['word_embed'] is None:
        embed_filename = 'None'
    else:
        embed_filename = params['word_embed'].split('/')[-1]
    for n_exp in range(params['n_experiment']):
        if params['n_experiment'] <= 1:
            file_predix = f"{params['model_type']}_ncluster{params['num_clusters']}_nfold{params['n_fold']}_direct{params['direct']}_threshold{params['threshold']}_embed{embed_filename}"
        else:
            file_predix = f"ex{n_exp}_{params['model_type']}_ncluster{params['num_clusters']}_nfold{params['n_fold']}_direct{params['direct']}_threshold{params['threshold']}_embed{embed_filename}"
        # 定义所有metrics
        all_metrics = {
            'recall1': [],
            'recall2': [],
            'recall3': [],
            'recall4': [],
            'recall5': [],
            'precision1': [],
            'precision2': [],
            'precision3': [],
            'precision4': [],
            'precision5': [],
            'accuracy1': [],
            'accuracy2': [],
            'accuracy3': [],
            'accuracy4': [],
            'accuracy5': [],
            'example_accuracy': [],
            'example_precision': [],
            'example_recall': [],
            'example_f1':[],
            'mrr1': [],
            'mrr2': [],
            'mrr3': [],
            'mrr4': [],
            'mrr5': [],
            'mrr': [],
            'map1': [],
            'map2': [],
            'map3': [],
            'map4': [],
            'map5': [],
            'map': []
        }

        abstract_all_metrics = {
            'recall1': [],
            'recall2': [],
            'recall3': [],
            'recall4': [],
            'recall5': [],
            'precision1': [],
            'precision2': [],
            'precision3': [],
            'precision4': [],
            'precision5': [],
            'accuracy1': [],
            'accuracy2': [],
            'accuracy3': [],
            'accuracy4': [],
            'accuracy5': [],
            'example_accuracy': [],
            'example_precision': [],
            'example_recall': [],
            'example_f1':[],
            'mrr1': [],
            'mrr2': [],
            'mrr3': [],
            'mrr4': [],
            'mrr5': [],
            'mrr': [],
            'map1': [],
            'map2': [],
            'map3': [],
            'map4': [],
            'map5': [],
            'map': []
        }
        # 十折交叉验证
        kf = KFold(n_splits=params['n_fold'], shuffle=True, random_state=params['seed'])

        # error_out = []
        for it, (train_index, test_index) in enumerate(kf.split(dataset)):
            if params['no_fold'] and it > 0:
                break
            print(f'train {n_exp}-{it} fold')
            print('train_index num:', len(train_index), 'test_index num:', len(test_index))
            train_dataset = [dataset[i] for i in train_index]
            test_dataset = [dataset[i] for i in test_index]

            # 测试集过滤
            classic_test_dataset = filter_classic_labels(test_dataset)

            # 收集所有标签
            all_labels = get_all_labels(train_dataset)
            all_project_labels = get_all_labels(train_dataset, field='project_labels')

            if params['direct'] == False:
                # 标签聚类
                if params['device'] < 0:
                    embedding_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens', device='cpu')
                else:
                    embedding_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens', device=f'cuda:{params["device"]}')
                if params['cluster_method'] == 'kmeans':
                    clustering_model = KMeans(n_clusters=params['num_clusters'], random_state=params['seed'])
                elif params['cluster_method'] == 'dbscan':
                    clustering_model = DBSCAN(eps=params['dbscan_eps'], min_samples=params['dbscan_min_samples'])
                elif params['cluster_method'] == 'birch':
                    clustering_model = Birch(threshold=params['birch_threshold'], branching_factor=params['birch_branching_factor'])

                label_embeddings = embedding_model.encode(all_labels)
                clustering_model.fit(label_embeddings)

                print('write to cluster file.')
                output_cluster("train_and_test_cluster.txt", params['num_clusters'], clustering_model, all_labels)

                # 构建embedding数据集
                embedding_train_dataset = embedding_dataset(train_dataset, embedding_model, clustering_model)
                embedding_test_dataset = embedding_dataset(test_dataset, embedding_model, clustering_model)

                # 平衡数据集
                # weights = make_weights_for_balanced_classes(embedding_train_dataset, params['num_clusters'])
            
            # 构建模型
            if params['model_type'] == 'fasttext':
                model = FasttextModel(embedding_path=params['word_embed'], embed_size=params['embed_size'])
            elif params['model_type'] == 'cnn':
                model = CNNModel(embedding_path=params['word_embed'], embed_size=params['embed_size'], device=params['device'])
            elif params['model_type'] == 'bilstm':
                model = BiLSTMModel(embedding_path=params['word_embed'], atten=False, embed_size=params['embed_size'], device=params['device'])
            elif params['model_type'] == 'bilstm_att':
                model = BiLSTMModel(embedding_path=params['word_embed'], atten=True, embed_size=params['embed_size'], device=params['device'])
            elif params['model_type'] == 'rcnn':
                model = RCNNModel(embedding_path=params['word_embed'], embed_size=params['embed_size'], device=params['device'])
            elif params['model_type'] == 'knn':
                model = KNNModel()
            elif params['model_type'] == 'svm':
                model = SVMModel()
            else:
                raise Exception('Unsupportted model type')

            # 开始训练
            if params['direct'] == False:
                model.train(embedding_train_dataset)
            else:
                model.train(train_dataset)
            
            # 构建预测器
            if params['direct'] == False:
                predictor = ClusterPredictor(model, embedding_model, clustering_model, params['top_k'])
                predictor.pre_compute_labels(all_project_labels)
            else:
                predictor = DirectPredictor(model, len(all_labels))

            if params['test_batch'] == 1:
                test(
                    predictor,
                    test_dataset,
                    all_metrics,
                    abstract=False,
                    out_path=os.path.join(params['result_dir'], f"{file_predix}_fold{it}_label_individual_metrics.csv"),
                    threshold=params['threshold']
                )

                test(
                    predictor,
                    classic_test_dataset,
                    abstract_all_metrics,
                    abstract=True,
                    out_path=os.path.join(params['result_dir'], f"{file_predix}_fold{it}_default_label_individual_metrics.csv"),
                    threshold=params['threshold']
                )
            else:
                test_batch(
                    predictor,
                    test_dataset,
                    all_metrics,
                    batch_size=params['test_batch'],
                    abstract=False,
                    out_path=os.path.join(params['result_dir'], f"{file_predix}_fold{it}_label_individual_metrics.csv"),
                    threshold=params['threshold']
                )

                test_batch(
                    predictor,
                    classic_test_dataset,
                    abstract_all_metrics,
                    batch_size=params['test_batch'],
                    abstract=True,
                    out_path=os.path.join(params['result_dir'], f"{file_predix}_fold{it}_default_label_individual_metrics.csv"),
                    threshold=params['threshold']
                )

            print('example recall: ', all_metrics['example_recall'][-1])
            print('example precision: ', all_metrics['example_precision'][-1])
            print('example F1-score: ', all_metrics['example_f1'][-1])


        #将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe = pd.DataFrame(all_metrics)
        dataframe.to_csv(os.path.join(params['result_dir'], f"{file_predix}_metric.csv"),index=False,sep=',')

        dataframe = pd.DataFrame(abstract_all_metrics)
        dataframe.to_csv(os.path.join(params['result_dir'], f"{file_predix}_abstract_metric.csv"),index=False,sep=',')

    return {
        'recall1': np.mean(all_metrics['recall1']),
        'recall2': np.mean(all_metrics['recall2']),
        'recall3': np.mean(all_metrics['recall3']),
        'recall4': np.mean(all_metrics['recall4']),
        'recall5': np.mean(all_metrics['recall5']),
        'precision1': np.mean(all_metrics['precision1']),
        'precision2': np.mean(all_metrics['precision2']),
        'precision3': np.mean(all_metrics['precision3']),
        'precision4': np.mean(all_metrics['precision4']),
        'precision5': np.mean(all_metrics['precision5']),
        'accuracy1': np.mean(all_metrics['accuracy1']),
        'accuracy2': np.mean(all_metrics['accuracy2']),
        'accuracy3': np.mean(all_metrics['accuracy3']),
        'accuracy4': np.mean(all_metrics['accuracy4']),
        'accuracy5': np.mean(all_metrics['accuracy5']),
        'example_accuracy': np.mean(all_metrics['example_accuracy']),
        'example_precision': np.mean(all_metrics['example_precision']),
        'example_recall': np.mean(all_metrics['example_recall']),
        'example_f1': np.mean(all_metrics['example_f1']),
    }

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train_test_main():
    # 读取设置
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('--device', default=0, type=int, required=False, help='使用的实验设备, -1:CPU, >=0:GPU')
    parser.add_argument('--n_experiment', default=1, type=int, required=False, help='实验次数')
    parser.add_argument('--dataset_path', default='./data/golden_dataset_mini.txt', type=str, required=False, help='数据集路径')
    parser.add_argument('--project_info_path', default='./data/projects_info.jsonl', type=str, required=False, help='项目信息路径')
    parser.add_argument('--output_dir', default='./model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--result_dir', default='./result_tmp/', type=str, required=False, help='结果输出路径')
    parser.add_argument('--model_type', default='cnn', type=str, required=False, help='模型类型')
    parser.add_argument('--seed', default=0, type=int, required=False, help='随机种子')
    parser.add_argument('--cluster_method', default='kmeans', type=str, required=False, help='聚类方法')
    parser.add_argument('--num_clusters', default=128, type=int, required=False, help='KMeans聚类类簇数量')
    parser.add_argument('--dbscan_eps', default=0.5, type=float, required=False, help='DBSCAN最大距离')
    parser.add_argument('--dbscan_min_samples', default=5, type=int, required=False, help='DBSCAN每个类簇最小数量')
    parser.add_argument('--birch_threshold', default=0.2, type=float, required=False, help='BIRCH阈值')
    parser.add_argument('--birch_branching_factor', default=10, type=int, required=False, help='BIRCH分支因子')
    parser.add_argument('--top_k', default=2, type=int, required=False, help='top-k')
    parser.add_argument('--direct', default=True, type=str2bool, required=False, help='传统方法直接预测')
    parser.add_argument('--n_fold', default=5, type=int, required=False, help='交叉验证的折数')
    # parser.add_argument('--classic', default=True, type=str2bool, required=False, help='只验证经典标签')
    parser.add_argument('--threshold', default=0, type=float, required=False, help='标签输出阈值')
    # "embedding/glove.6B/glove.6B.300d.txt"
    # "embedding/word2vec/word2vec-google-news-300.txt"
    parser.add_argument('--embed_size', default=300, type=int, required=False, help='词向量维度')
    parser.add_argument('--word_embed', default="embedding/glove.6B/glove.6B.300d.txt", type=str, required=False, help='选择词向量类型')
    parser.add_argument('--no_fold', default=False, type=str2bool, required=False, help='参数搜索时不做k折加快速度')
    parser.add_argument('--test_batch', default=256, type=int, required=False, help='预测时的batch数量，等于1时最慢')
    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    result = train({arg: getattr(args, arg) for arg in vars(args)})
    print(result)

if __name__ == "__main__":
    train_test_main()