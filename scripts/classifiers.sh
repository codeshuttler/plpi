RESULT_PATH="./result_classifiers"
DEVICE=0
EMBEDDING="embedding/glove.6B/glove.6B.300d.txt"

# ex3
python train_and_test.py --model_type cnn --num_clusters 128 --threshold 0.05 --device $DEVICE --direct false --word_embed $EMBEDDING --result_dir $RESULT_PATH
python train_and_test.py --model_type bilstm --num_clusters 128 --threshold 0.05 --device $DEVICE --direct false --word_embed $EMBEDDING --result_dir $RESULT_PATH
python train_and_test.py --model_type bilstm_att --num_clusters 128 --threshold 0.05 --device $DEVICE --direct false --word_embed $EMBEDDING --result_dir $RESULT_PATH
python train_and_test.py --model_type fasttext --num_clusters 128 --threshold 0.05 --device $DEVICE --direct false --word_embed $EMBEDDING --result_dir $RESULT_PATH
python train_and_test.py --model_type knn --num_clusters 128 --threshold 0.05 --device $DEVICE --direct false --result_dir $RESULT_PATH
python train_and_test.py --model_type svm --num_clusters 128 --threshold 0.05 --device $DEVICE --direct false --result_dir $RESULT_PATH