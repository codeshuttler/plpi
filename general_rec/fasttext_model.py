import re
import tempfile
import fasttext
import numpy as np

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
        for each_label in item['labels']:
            count[label_to_id[each_label]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count)) * nclasses
    for i in range(nclasses):
        if float(count[i]) == 0:
            weight_per_class[i] = 0
            print(f'class {i} is zero count.')
        else:
            weight_per_class[i] = float(count[i])/N

    weight = [0] * len(data)
    for idx, val in enumerate(data):
        for each_label in item['labels']:
            weight[idx] += weight_per_class[label_to_id[each_label]]
        weight[idx] = 1.0 / weight[idx]

    return weight

class FasttextModel(object):
    def __init__(self, embedding_path=None, embed_size=50):
        self.embedding_path = embedding_path
        self.embed_size = embed_size
        self.model_path = None
        self.model = None

    def train(self, dataset):
        # 平衡数据集，FASTTEXT不需要，内部已经处理过了
        # 获取标签
        labels = set()
        for each_data in dataset:
            for each_label in each_data['labels']:
                labels.add(each_label)
        labels = list(labels)
        label_dict = {}
        for i, each_label in enumerate(labels):
            label_dict[i] = each_label
        # balance data
        weights = make_weights_for_balanced_classes(dataset, label_dict)
        weights = np.array(weights)
        weights_sum = weights.sum()
        weights /= weights_sum
        dataset = np.random.choice(dataset, size=2*len(dataset), p=weights)

        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8') as f_embed:
            if self.embedding_path is not None:
                with open(self.embedding_path, 'r', encoding='utf-8') as f_embed_in:
                    lines = f_embed_in.readlines()
                f_embed.write(f'{len(lines)} 300\n')
                f_embed.writelines(lines)
                f_embed.seek(0)
            with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8') as fp:
                dataset_to_fasttext_dataset(dataset, fp)
                if self.embedding_path is None:
                    self.model = fasttext.train_supervised(
                        input=fp.name,
                        dim=self.embed_size,
                        minn=3,
                        maxn=5,
                        neg=5,
                        bucket=1000000,
                        epoch=80,
                        lr=1,
                        wordNgrams=5,
                        minCount=100,
                        loss='ova')
                else:
                    self.model = fasttext.train_supervised(
                        input=fp.name,
                        dim=self.embed_size,
                        minn=3,
                        maxn=5,
                        neg=5,
                        bucket=1000000,
                        epoch=80,
                        lr=1,
                        wordNgrams=5,
                        minCount=100,
                        pretrainedVectors=f_embed.name,
                        loss='ova')
        if self.model_path is not None:
            self.model.save_model(self.model_path)

    def predict(self, text):
        prediction = self.model.predict(text, k=-1)
        return prediction

    def predict_batch(self, text_list):
        ret = []
        for each_text in text_list:
            ret.append(self.predict(each_text))
        return ret