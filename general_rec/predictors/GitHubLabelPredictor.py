
from allennlp.common.util import JsonDict
from overrides import overrides
from typing import List
from allennlp.predictors import Predictor
import numpy as np

@Predictor.register('github-label-predictor')
class GitHubLabelPredictor(Predictor):
    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        title = json_dict['title']
        description = json_dict['description']
        text = title + ' ' + description
        instance = self._dataset_reader.text_to_instance(text)

        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        prediction = self.predict_instance(instance)
        prediction['top-5'] = []
        top_5_label_indice = np.array(prediction['logits']).argsort()[-5:][::-1].tolist()
        for each_label in top_5_label_indice:
            prediction['top-5'].append(label_dict[each_label])

        # return {"instance": prediction, "all_labels": all_labels}
        return {"instance": prediction}
