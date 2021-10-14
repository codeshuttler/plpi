
from train_and_test import train
import json
import tqdm

if __name__ == "__main__":
    result = []

    n_cluster_list = [2, 4, 8, 16, 32, 64, 128]# , 256]
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    n_cluster_list = [2, 4, 8, 16, 32, 64, 128]
    n_cluster_list = [128, 230]
    threshold_list = [0.00005, 0.0005, 0.005, 0.05, 0.5]
    # threshold_list = [0.0005, 0.005, 0.05, 0.5]
    k_top_abstract_list = [2, 4, 8, 16]
    # threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    params = []
    for n_cluster in n_cluster_list:
        for threshold in threshold_list:
        # for k_top in k_top_abstract_list:
        #    if k_top >= n_cluster:
        #            continue
            param = {
                'device': 1,
                'n_experiment': 1,
                'dataset_path': 'data/golden_dataset_mini.txt',
                'project_info_path': 'data/projects_info.jsonl',
                'model_type': 'cnn',
                'seed': 0,
                'cluster_method': 'kmeans',
                'num_clusters': n_cluster,
                'top_k': 2,
                'n_fold': 3,
                'direct': False,
                'classic': False,
                'word_embed': None,
                'embed_size': 50,
                'result_dir': 'param_search_result',
                'threshold': threshold,
                'no_fold': True,
                'test_batch': 256
            }
            params.append(param)

    for each_param in tqdm.tqdm(iterable=params):
        print(each_param)
        each_result = train(each_param)
        result.append(
            {
                'n_cluster': each_param['num_clusters'],
                'k_top': each_param['top_k'],
                'threshold': each_param['threshold'],
                'metrics': each_result
            }
        )

        with open('param_search.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(result))