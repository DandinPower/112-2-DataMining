from argparse import ArgumentParser
import numpy as np
from src.data_processing import FillInvalidStrategy, load_and_preprocess_train_test_dataset, get_fill_invalid_strategy_enum
from src.scaler import Scaler, get_scaler_by_strategy
from src.slide_window import create_train_dataset_by_sliding_window, create_test_dataset_by_sliding_window
from src.linear_regression import LinearRegression
from src.analyze import show_feature_correlation_to_target, show_train_valid_loss_progress
from sklearn.model_selection import train_test_split

def train(args: ArgumentParser) -> None:
    fill_invalid_strategy = get_fill_invalid_strategy_enum(args.fill_invalid_strategy)
    train_df, test_df = load_and_preprocess_train_test_dataset(args.train_csv_path, args.test_csv_path, fill_invalid_strategy)
    scaler = get_scaler_by_strategy(args.scaler_strategy)
    train_numpy_matrix = train_df.to_numpy()
    test_numpy_matrix = test_df.to_numpy()

    scaler.fit(train_numpy_matrix)
    train_scaled_matrix = scaler.transform(train_numpy_matrix)

    show_feature_correlation_to_target(train_scaled_matrix, args.target_column_index, args.analyze_folder)

    test_scaled_matrix = scaler.transform(test_numpy_matrix)

    # selected_features_index = list(range(train_numpy_matrix.shape[1]))
    selected_features_index = [1, 2, 3, 5, 6, 8, 9, 12, 13]

    train_X, train_Y = create_train_dataset_by_sliding_window(train_scaled_matrix, args.test_window_size, args.target_column_index, selected_features_index)
    test_X = create_test_dataset_by_sliding_window(test_scaled_matrix, args.test_window_size, selected_features_index)

    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=args.split_size, random_state=args.random_seed, shuffle=True)
    
    feature_num = train_X.shape[1]

    model = LinearRegression(feature_num, random_seed=args.random_seed, is_bias=args.is_bias)
    train_loss, valid_loss = model.fit(train_X, train_Y, valid_X, valid_Y, scaler, args.target_column_index, args.epoch, args.learning_rate, args.batch_size, args.regularization_rate)

    show_train_valid_loss_progress(train_loss, valid_loss, args.train_result_folder, args.train_name)

    test_Y_pred = model.predict(test_X)
    test_Y_de_scaled = scaler.de_transform(test_Y_pred, args.target_column_index)

    with open(f'{args.predict_folder}/{args.train_name}.csv', 'w+') as csv_file:
        csv_file.write('index,answer\n')
        for index, pred in enumerate(test_Y_de_scaled):
            csv_file.write(f'index_{index},{pred}\n')

def main(args: ArgumentParser) -> None:
    train(args)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_csv_path', type=str)
    parser.add_argument('--test_csv_path', type=str)
    parser.add_argument('--fill_invalid_strategy', type=str)
    parser.add_argument('--scaler_strategy', type=str)
    parser.add_argument('--target_column_index', type=int)
    parser.add_argument('--test_window_size', type=int)
    parser.add_argument('--split_size', type=float)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--regularization_rate', type=float)
    parser.add_argument('--is_bias', type=int)
    parser.add_argument('--analyze_folder', type=str)
    parser.add_argument('--predict_folder', type=str)
    parser.add_argument('--train_result_folder', type=str)
    parser.add_argument('--train_name', type=str)
    args = parser.parse_args()
    main(args)