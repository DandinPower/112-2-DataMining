TRAIN_CSV_PATH=datasets/train.csv
TEST_CSV_PATH=datasets/test.csv
ANALYZE_FOLDER=output/analyze
PREDICT_FOLDER=output/predict
TRAIN_RESULT_FOLDER=output/train
FILL_INVALID_STRATEGY=MEAN #ZERO, MEAN, MEDIAN
SCALER_STRATEGY=MIN_MAX # MIN_MAX, STANDARD, NON
TARGET_COLUMN_INDEX=9 # PM2.5
TEST_WINDOW_SIZE=9
SPLIT_SIZE=0.2
RANDOM_SEED=42
LEARNING_RATE=0.01
BATCH_SIZE=16
REGULARIZATION_RATE=0 # Set 0 = disable
EPOCH=3000
IS_BIAS=1

TRAIN_NAME=BATCH_SIZE_${BATCH_SIZE}_EPOCH_${EPOCH}_RANDOM_SEED_${RANDOM_SEED}_LEARNING_RATE_${LEARNING_RATE}

python main.py --train_csv_path $TRAIN_CSV_PATH --test_csv_path $TEST_CSV_PATH --fill_invalid_strategy $FILL_INVALID_STRATEGY --scaler_strategy $SCALER_STRATEGY \
    --target_column_index $TARGET_COLUMN_INDEX --test_window_size $TEST_WINDOW_SIZE --split_size $SPLIT_SIZE --random_seed $RANDOM_SEED --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE --regularization_rate $REGULARIZATION_RATE --epoch $EPOCH --is_bias $IS_BIAS --analyze_folder $ANALYZE_FOLDER --predict_folder $PREDICT_FOLDER \
    --train_result_folder $TRAIN_RESULT_FOLDER --train_name $TRAIN_NAME