# PLPI

This is the source code for the paper "Personalizing Label Prediction for GitHub Issues".

## Package Requirement

To run this code, some packages are needed as following:

```python
numba==0.54.0
joblib==1.0.1
overrides==3.1.0
torch==1.9.0
torchvision==0.10.0
numpy==1.19.5
tqdm==4.62.3
allennlp==2.7.0
pandas==1.2.2
nltk==3.6.3
fasttext==0.9.2
scikit_learn==1.0
sentence_transformers==2.1.0
```

## Download Embeddings

embedding/glove.6B: Embedding glove can be downloaded from the GloVe repository at [GloVe](https://github.com/stanfordnlp/GloVe) Github Repository. Download URL - [glove.6B.zip](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip).

embedding/word2vec/word2vec-google-news-300.txt: Word2vec Embedding is also available on GitHub [word2vec-GoogleNews-vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors).

Rename it from "GoogleNews-vectors-negative300.bin" to "word2vec-google-news-300".
Then we can use gensim to convert its model into txt data (Size 3.39G -> 10G):
```
from gensim.models.keyedvectors import KeyedVectors
 
model = KeyedVectors.load_word2vec_format('word2vec-google-news-300', binary=True)
model.save_word2vec_format('word2vec-google-news-300.txt', binary=False)
```

## Prepare dataset

**PLPI_GitHub_dataset.7z**

Under the data root is the issues data as the format of JSON LInes.
Each line of the file is a JSON object, which contains the following properties.

```json
{
    "project": "project_name",
    "title": "title",
    "description": "description", 
    "labels": ["bug", "question"]
}
```

## Research Questions

All experiment-related scripts are saved under the path `./scripts`.

### RQ1: Parameter Sensitivity

#### Parameter search

run `scripts/param_search.sh` and the result of experiments will be dumped to `param_search.json`.

#### Different Classifiers

run `scripts/classifiers.sh` and the result of experiments will be saved under the `result_classifiers` directory.

### RQ2: Effectiveness

run `scripts/effectiveness.sh` and the all metrics of RQ2.1 and RQ2.2 will be saved under the `result_effectiveness` directory.

### RQ3:  Effectiveness Comparision

run `scripts/comparison.sh` and the result will be saved under the `result_comparison` directory.

## References

For more details about data processing, please refer to the `code comments` and our paper.

For more flexible and specific parameter settings during training, please refer to the tutorial of *pytorch* and *pytorch_lightning*.

