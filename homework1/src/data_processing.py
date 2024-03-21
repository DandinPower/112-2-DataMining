import pandas as pd
import numpy as np
from enum import Enum 

class FillInvalidStrategy(Enum):
    ZERO = 0
    MEAN = 1
    MEDIAN = 2

def get_fill_invalid_strategy_enum(type: str) -> FillInvalidStrategy:
    if type == "ZERO":
        return FillInvalidStrategy.ZERO
    elif type == "MEAN":
        return FillInvalidStrategy.MEAN
    elif type == "MEDIAN":
        return FillInvalidStrategy.MEDIAN
    else:
        raise ValueError(f'Unknown fill_invalid_strategy: {type}')

def get_column_filled_invalid_value_by_strategy(train_df: pd.DataFrame, fill_invalid_strategy: FillInvalidStrategy) -> pd.Series | int:
    if fill_invalid_strategy == FillInvalidStrategy.ZERO:
        return 0
    elif fill_invalid_strategy == FillInvalidStrategy.MEAN:
        return train_df.mean(axis=0)
    elif fill_invalid_strategy == FillInvalidStrategy.MEDIAN:
        return train_df.median(axis=0)
    else:
        raise ValueError(f'Unknown fill_invalid_strategy: {fill_invalid_strategy}')

def load_and_preprocess_train_test_dataset(train_file_path: str, test_file_path: str, fill_invalid_strategy: FillInvalidStrategy) -> tuple[pd.DataFrame, pd.DataFrame]:
    # train part
    train_df = pd.read_csv(train_file_path)
    train_df['ItemName'] = train_df['ItemName'].str.strip()
    train_df.pop('Location')

    train_df['Date'] = train_df['Date'].str.strip()
    train_df['Date'] = train_df['Date'].str.strip('00:00')

    train_df = train_df.melt(id_vars=['Date', 'ItemName'], var_name='Hour', value_name='Value')

    train_df['Hour'] = train_df['Hour'].astype(int)


    train_df = train_df.pivot(index=['Date', 'Hour'], columns='ItemName', values='Value')
    train_df = train_df.apply(pd.to_numeric, errors='coerce')

    column_filled_values = get_column_filled_invalid_value_by_strategy(train_df, fill_invalid_strategy)
    train_df = train_df.fillna(column_filled_values)

    # test part
    test_df = pd.read_csv(test_file_path)
    test_df['Date'] = test_df['Date'].str.strip('index_')
    test_df['Date'] = test_df['Date'].astype(int)
    test_df['ItemName'] = test_df['ItemName'].str.strip()
    test_df = test_df.melt(id_vars=['Date', 'ItemName'], var_name='Hour', value_name='Value')
    test_df['Hour'] = test_df['Hour'].astype(int)
    test_df = test_df.pivot(index=['Date', 'Hour'], columns='ItemName', values='Value')
    test_df = test_df.apply(pd.to_numeric, errors='coerce')
    test_df = test_df.fillna(column_filled_values)

    return (train_df, test_df)