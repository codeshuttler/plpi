RESULT_PATH="./result_n"
DEVICE=0
NEXP=10
EMBEDDING="embedding/glove.6B/glove.6B.300d.txt"

# ex2
python train_and_test.py --n_experiment $NEXP --model_type rcnn --threshold 0.05 --device $DEVICE --direct false --word_embed $EMBEDDING --result_dir $RESULT_PATH
python train_and_test.py --n_experiment $NEXP --model_type rcnn --threshold 0.05 --device $DEVICE --direct true --word_embed $EMBEDDING --result_dir $RESULT_PATH
