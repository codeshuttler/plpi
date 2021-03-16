from collections import defaultdict

from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from nltk.stem import WordNetLemmatizer
import random
import tqdm
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn

import joblib
from numba import jit

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier

def fasttext_lines_to_dataset(lines):
    dataset = []
    for each_line in lines:
        splits_segs = each_line.split(sep=' ')
        text = ''
        labels = []
        for each_seg in splits_segs:
            if each_seg.startswith('__label__'):
                labels.append(each_seg[9:].replace('<s>', ' '))
            else:
                text += ' ' + each_seg

        dataset.append((text, labels))


wn = WordNetLemmatizer()
tag_map = defaultdict(lambda : NOUN)
tag_map['JJ'] = ADJ
tag_map['JJR'] = ADJ
tag_map['JJS'] = ADJ
tag_map['VB'] = VERB
tag_map['VBD'] = VERB
tag_map['VBP'] = VERB
tag_map['VBG'] = VERB
tag_map['VBN'] = VERB
tag_map['VBZ'] = VERB
tag_map['RB'] = ADV
tag_map['RBR'] = ADV
tag_map['RBS'] = ADV
tag_map['NN'] = NOUN
tag_map['NNS'] = NOUN
tag_map['NNP'] = NOUN
tag_map['NNPS'] = NOUN

selected_tags = set(['JJ', 'JJR','JJS','VB','VBD','VBP',\
                'VBG', 'VBN','VBZ','RB','RBR','RBS','NN',\
                'NNS','NNP', 'NNPS'])

lemma_cache = {}
stopwords_set = set(stopwords.words('english'))
# print('stopwords_set type:' + str(type(stopwords_set)))

def text_to_tokens(text):
    # 分词
    tokens = word_tokenize(text)

    new_tokens = []

    for word, tag in pos_tag(tokens):
        if tag in selected_tags and word not in stopwords_set and word.isalpha():
            if word in lemma_cache:
                word_final = lemma_cache[word]
            else:
                word_final = wn.lemmatize(word,tag_map[tag[0]])
                lemma_cache[word] = word_final
            new_tokens.append(word_final.lower())
            # 限制长度
            if len(new_tokens) > 256:
                break
    
    return new_tokens

def json_dataset_to_pair(dataset):
    ret = []
    for each_data in dataset:
        ret.append((each_data['title']+' '+each_data['description'], each_data['labels']))
    return ret

class SVMModel(object):
    def __init__(self):
        self.model = None
        self.Tfidf_vect = None

    def train(self, dataset):
        random.shuffle(dataset)
        split_index = int(0.9 * len(dataset))
        train_lines = dataset[:split_index]
        valid_lines = dataset[split_index:]

        train_dataset = json_dataset_to_pair(train_lines)
        valid_dataset = json_dataset_to_pair(valid_lines)

        # 获取标签
        labels = set()
        for each_data in train_dataset:
            for each_label in each_data[1]:
                labels.add(each_label)
        for each_data in valid_dataset:
            for each_label in each_data[1]:
                labels.add(each_label)
        
        self.labels = list(labels)
        self.label_to_id = {}
        self.id_to_label = {}
        for i, each_label in enumerate(labels):
            self.label_to_id[each_label] = i
            self.label_to_id[i] = each_label

        # 构建tfidf
        self.Tfidf_vect = TfidfVectorizer(max_features=5000)
        self.Tfidf_vect.fit([each_sent[0] for each_sent in train_dataset])
        
        n_labels = len(self.labels)
        n_feature = len(self.Tfidf_vect.get_feature_names())
        train_n_samples = len(train_dataset)
        valid_n_samples = len(valid_dataset)
        train_x = np.zeros((train_n_samples, n_feature))
        train_y = np.zeros((train_n_samples, n_labels))

        valid_x = np.zeros((valid_n_samples, n_feature))
        valid_y = np.zeros((valid_n_samples, n_labels))

        # 处理训练集
        print('preprocess svm train dataset')
        preprocessed_dataset = [None] * len(train_dataset)
        for i, each_data in enumerate(tqdm.tqdm(iterable=train_dataset)):
            tokens = text_to_tokens(each_data[0])
            preprocessed_dataset[i] = ' '.join(tokens)
            for each_label in each_data[1]:
                train_y[i, self.label_to_id[each_label]] = 1

        tfidf_features = self.Tfidf_vect.transform(preprocessed_dataset)
        tfidf_features.sort_indices()
        train_x = tfidf_features  # .todense()
        
        # 处理验证集
        print('preprocess svm valid dataset')
        preprocessed_dataset = [None] * len(valid_dataset)
        for i, each_data in enumerate(tqdm.tqdm(iterable=valid_dataset)):
            tokens = text_to_tokens(each_data[0])
            preprocessed_dataset[i] = ' '.join(tokens)
            for each_label in each_data[1]:
                valid_y[i, self.label_to_id[each_label]] = 1
        
        tfidf_features = self.Tfidf_vect.transform(preprocessed_dataset)
        tfidf_features.sort_indices()
        valid_x = tfidf_features  # .todense()
        
        self.svc = SVC(verbose=False, probability=True)
        # self.bagging = BaggingClassifier(self.svc, max_samples=0.6, max_features=0.7, verbose=1, oob_score=True)
        self.model = OneVsRestClassifier(self.svc, n_jobs=16)
        self.model.fit(train_x, train_y)

    def predict(self, text):
        tokens = text_to_tokens(text)
        preprocessed_text = ' '.join(tokens)
        tfidf_features = self.Tfidf_vect.transform([preprocessed_text])
        # input: (n_queries, n_features), output: (n_queries, n_outputs)
        result = self.model.predict_proba(tfidf_features)
        # print((tuple(self.labels), result[0]))
        return (tuple(self.labels), result[0])

    def predict_batch(self, text_list):
        ret = []
        for each_text in text_list:
            ret.append(self.predict(each_text))
        return ret

if __name__ == "__main__":
    tokens = text_to_tokens('Analyzer Feedback from IntelliJ <CODE> For additional log information, please append the contents of\nfile:///private/var/folders/26/bxvyy0q52wbggpksb6w5xtyh0000gn/T/report.txt. ')

    print(tokens)



    