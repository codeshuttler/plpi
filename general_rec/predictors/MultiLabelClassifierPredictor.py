
from allennlp.common.util import JsonDict
from overrides import overrides
from typing import List
from allennlp.predictors import Predictor
import numpy as np

@Predictor.register('multi-label-classifier-predictor')
class MultiLabelClassifierPredictor(Predictor):
    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        text = json_dict['text']
        instance = self._dataset_reader.text_to_instance(text)

        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        prediction = self.predict_instance(instance)
        labels = []
        label_indice = np.array(prediction['logits']).argsort()[::-1]
        label_indice_list = label_indice.tolist()
        for each_label in label_indice_list:
            labels.append(label_dict[each_label])


        sorted_logits = np.array(prediction['logits'])[label_indice]

        # return {"instance": prediction, "all_labels": all_labels}
        return {"logits": sorted_logits, "labels": labels}

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        ret = []
        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        instances = []
        for each_input in inputs:
            text = each_input['text']
            instance = self._dataset_reader.text_to_instance(text)
            instances.append(instance)

        prediction = self.predict_batch_instance(instances)

        for each_predict in prediction:
            labels = []
            label_indice = np.array(each_predict['logits']).argsort()[::-1]
            label_indice_list = label_indice.tolist()
            for each_label in label_indice_list:
                labels.append(label_dict[each_label])

            sorted_logits = np.array(each_predict['logits'])[label_indice]
            ret.append({"logits": sorted_logits, "labels": labels})

        # return {"instance": prediction, "all_labels": all_labels}
        return ret
