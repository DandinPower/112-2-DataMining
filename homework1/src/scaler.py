import numpy as np

class Scaler:
    def __init__(self):
        pass 

    def fit(self, X: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def de_transform(self, X: np.ndarray, column_index: int) -> np.ndarray:
        pass

class NonScaler(Scaler):
    def __init__(self):
        pass 

    def fit(self, X: np.ndarray) -> None:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def de_transform(self, X: np.ndarray, column_index: int) -> np.ndarray:
        return X

class MinMaxScaler(Scaler):
    def __init__(self):
        pass 

    def fit(self, X: np.ndarray) -> None:
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.min_ is not None 
        assert self.max_ is not None
        assert X.shape[1] == self.min_.shape[0]
        assert X.shape[1] == self.max_.shape[0]
        return (X - self.min_) / (self.max_ - self.min_)

    def de_transform(self, X: np.ndarray, column_index: int) -> np.ndarray:
        assert self.min_ is not None 
        assert self.max_ is not None
        assert column_index < self.min_.shape[0]
        assert column_index < self.max_.shape[0]
        return (X * (self.max_[column_index] - self.min_[column_index])) + self.min_[column_index]

class StandardScaler(Scaler):
    def __init__(self):
        pass 

    def fit(self, X: np.ndarray) -> None:
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None 
        assert self.std_ is not None
        assert X.shape[1] == self.mean_.shape[0]
        assert X.shape[1] == self.std_.shape[0]
        return (X - self.mean_) / self.std_
    
    def de_transform(self, X: np.ndarray, column_index: int) -> np.ndarray:
        assert self.mean_ is not None 
        assert self.std_ is not None
        assert column_index < self.mean_.shape[0]
        assert column_index < self.std_.shape[0]
        return (X * self.std_[column_index]) + self.mean_[column_index]

def get_scaler_by_strategy(type: str) -> Scaler:
    if type == "MIN_MAX":
        return MinMaxScaler()
    elif type == "STANDARD":
        return StandardScaler()
    elif type == "NON":
        return NonScaler()
    else:
        raise ValueError(f'Unknown scaler_type: {type}')