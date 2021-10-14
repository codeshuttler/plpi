RESULT_PATH="./result_effectiveness"
DEVICE=0
# ex1
python train_and_test.py --model_type rcnn --test_batch 256 --num_clusters 128 --threshold 0.05 --device $DEVICE --direct false --word_embed "embedding/glove.6B/glove.6B.300d.txt" --result_dir $RESULT_PATH
python train_and_test.py --model_type rcnn --test_batch 256 --num_clusters 128 --threshold 0.05 --device $DEVICE --direct false --word_embed "embedding/word2vec/word2vec-google-news-300.txt" --result_dir $RESULT_PATH
python train_and_test.py --model_type rcnn --test_batch 256 --num_clusters 128 --threshold 0.05 --device $DEVICE --direct false --word_embed "" --result_dir $RESULT_PATH
