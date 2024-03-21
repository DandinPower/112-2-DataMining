import numpy as np

HOURS_PER_DAY=24

def create_train_dataset_by_sliding_window(train_data: np.ndarray, window_size: int, target_index: int, selected_features_index: list[int]) -> tuple[np.ndarray, np.ndarray]:
    X = []
    Y = []
    assert train_data.shape[0] % HOURS_PER_DAY == 0
    number_of_days = len(train_data) // HOURS_PER_DAY
    for day in range(number_of_days):
        day_data = train_data[day*HOURS_PER_DAY:(day+1)*HOURS_PER_DAY, :]
        for i in range(HOURS_PER_DAY - window_size):
            window = day_data[i:i+window_size, selected_features_index]
            features = window.flatten()
            target = day_data[i+window_size, target_index]
            X.append(features)
            Y.append(target)

    return np.array(X), np.array(Y)

def create_test_dataset_by_sliding_window(test_data: np.ndarray, window_size: int, selected_features_index: list[int]) -> np.ndarray:
    X = []
    assert len(test_data) % window_size == 0
    number_of_days = len(test_data) // window_size
    for day in range(number_of_days):
        window = test_data[day*window_size:(day+1)*window_size, selected_features_index]
        features = window.flatten()
        X.append(features)

    return np.array(X)