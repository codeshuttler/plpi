RESULT_PATH="./result_time"
DEVICE=0
# ex1
python train_and_test.py --dataset_path "./data/golden_dataset_mini_fined.txt" --model_type rcnn --test_batch 1 --num_clusters 32 --threshold 0.05 --device $DEVICE --direct false --word_embed "embedding/glove.6B/glove.6B.300d.txt" --result_dir $RESULT_PATH
python train_and_test.py --dataset_path "./data/golden_dataset_mini_fined.txt" --model_type rcnn --test_batch 1 --num_clusters 32 --threshold 0.05 --device $DEVICE --direct true --word_embed "embedding/glove.6B/glove.6B.300d.txt" --result_dir $RESULT_PATH