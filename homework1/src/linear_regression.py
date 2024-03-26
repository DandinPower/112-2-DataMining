import numpy as np
import math
from tqdm import tqdm
from .scaler import Scaler
import os

class LinearRegression:
    def __init__(self, feature_num: int, random_seed: int, is_bias: bool):
        scale = 1/max(1., (2+2)/2.)
        limit = math.sqrt(3.0 * scale)
        np.random.seed(random_seed)
        self.weights_ = np.random.uniform(-limit, limit, size=(feature_num))
        self.bias_ = None
        if is_bias:
            self.bias_ = np.random.uniform(-limit, limit, size=(1))
    
    def save(self, path: str) -> None:
        np.savez(path, weights=self.weights_, bias=self.bias_)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.weights_ = data['weights']
        if 'bias' in data:
            self.bias_ = data['bias']

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.shape[1] == self.weights_.shape[0]
        output = np.dot(X, self.weights_)
        if self.bias_:
            output += self.bias_
        return output

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray, valid_X: np.ndarray, valid_Y: np.ndarray, scaler: Scaler, target_column_index: int, epoch: int, learning_rate: float, batch_size: int, regularization_rate: float, train_name: str, model_folder: str) -> tuple[list[float], list[float]]:
        assert train_X.shape[0] == train_Y.shape[0]
        assert valid_X.shape[0] == valid_Y.shape[0]
        assert train_X.shape[1] == self.weights_.shape[0]
        assert valid_X.shape[1] == self.weights_.shape[0]
        assert train_X.shape[0] % batch_size == 0
        total_batch_num = train_X.shape[0] // batch_size
        
        train_loss = []
        valid_loss = []

        max_valid_loss = float('inf')
        
        for i in tqdm(range(epoch)):
            rmse_train_loss = 0.0
            for batch in range(total_batch_num):
                x = train_X[batch*batch_size:(batch+1)*batch_size]
                y = train_Y[batch*batch_size:(batch+1)*batch_size]
                y_pred = self.predict(x)
                loss = y - y_pred
                loss = np.reshape(loss, (-1, 1))

                l1_grad = -2 * loss * x
                l2_grad = 2 * regularization_rate * self.weights_
                grad = l1_grad + l2_grad
                grad = np.average(grad, axis=0)

                self.weights_ = self.weights_ - (learning_rate * grad)

                if self.bias_:
                    bias_grad = -2 * loss
                    bias_grad = np.average(bias_grad, axis=0)
                    self.bias_ = self.bias_ - (learning_rate * bias_grad)

            valid_Y_pred = self.predict(valid_X)
            rmse_valid_loss = valid_Y - valid_Y_pred 
            rmse_valid_loss = scaler.de_transform(rmse_valid_loss, target_column_index)
            rmse_valid_loss = rmse_valid_loss * rmse_valid_loss
            rmse_valid_loss = np.average(rmse_valid_loss, axis=0)
            rmse_valid_loss = math.sqrt(rmse_valid_loss)

            if rmse_valid_loss < max_valid_loss:
                print(f'epoch: {i}, valid_loss: {rmse_valid_loss} saved')
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
                self.save(f'{model_folder}/{train_name}.npz')
                max_valid_loss = rmse_valid_loss
                
            train_Y_pred = self.predict(train_X)
            rmse_train_loss = train_Y - train_Y_pred 
            rmse_train_loss = scaler.de_transform(rmse_train_loss, target_column_index)
            rmse_train_loss = rmse_train_loss * rmse_train_loss
            rmse_train_loss = np.average(rmse_train_loss, axis=0)
            rmse_train_loss = math.sqrt(rmse_train_loss)

            train_loss.append(rmse_train_loss)
            valid_loss.append(rmse_valid_loss)
        return train_loss, valid_loss