import json
from typing import List
from overrides import overrides
import pickle

from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
import numpy as np

from sentence_transformers import SentenceTransformer

@Predictor.register('general-label-predictor')
class GeneralLabelPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super(GeneralLabelPredictor, self).__init__(model, dataset_reader)
        '''
        self.project_info_path = 'data/projects_info.jsonl'
        self.project_infos = {}
        with open(self.project_info_path, 'r', encoding='utf-8') as f:
            for each_line in f:
                json_obj = json.loads(each_line)
                self.project_infos[json_obj['project']] = json_obj
        '''

        with open('./data/cluster.pkl', 'rb') as handle:
            self.clustering_model = pickle.load(handle)

        self.model = SentenceTransformer('bert-base-nli-stsb-mean-tokens', device='cuda:1')

        self.label_embed_cache = {}

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        title = json_dict['title']
        description = json_dict['description']
        project = json_dict['project']
        project_labels = json_dict['project_labels']

        text = title + ' ' + description
        instance = self._dataset_reader.text_to_instance(text)

        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["ACL", "AI", ...]
        # all_labels = [label_dict[i] for i in range(len(label_dict))]
        k = 5
        prediction = self.predict_instance(instance)
        cluster_top_k = []
        top_k_label_indice = np.array(prediction['logits']).argsort()[-k:][::-1].tolist()
        for each_label in top_k_label_indice:
            cluster_top_k.append(label_dict[each_label])

        # 先对为缓存的标签缓存
        no_cache_labels = []
        for each_label in project_labels:
            if each_label not in self.label_embed_cache:
                no_cache_labels.append(each_label)
        
        embeds = self.model.encode(no_cache_labels, show_progress_bar=False)
        for i, each_label in enumerate(no_cache_labels):
            self.label_embed_cache[each_label] = embeds[i]
        # 将标签转换为向量
        project_labels_embedding = []
        for each_label in project_labels:
            if each_label in self.label_embed_cache:
                project_labels_embedding.append(self.label_embed_cache[each_label])

        dists = np.zeros((len(project_labels_embedding),))
        for i, each_embed in enumerate(project_labels_embedding):
            min_dist = None
            for each_pred in cluster_top_k:
                cur_vec = self.clustering_model.cluster_centers_[int(each_pred)]

                dist = np.linalg.norm(each_embed - cur_vec)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            dists[i] = min_dist

        distance_list = {project_labels[i]: dists[i] for i in range(len(project_labels_embedding))}

        # return {"instance": prediction, "all_labels": all_labels}
        return {"instance": prediction, "distance": distance_list}
