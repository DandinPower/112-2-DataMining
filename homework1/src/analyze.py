import matplotlib.pyplot as plt
import numpy as np
import os

def show_feature_correlation_to_target(data: np.ndarray, target_column_index: int, output_folder: str) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # draw every column (feature) with target feature correlation plot
    for i in range(data.shape[1]):
        plt.scatter(data[:, i], data[:, target_column_index])
        plt.xlabel(f'Feature {i}')
        plt.ylabel(f'Target Feature')
        plt.savefig(f'{output_folder}/feature_{i}.png')
        plt.clf()

def show_train_valid_loss_progress(train_loss: list[float], valid_loss: list[float], output_folder: str, train_name: str) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # show train and valid loss progress
    plt.plot(train_loss, label='train_loss')
    plt.plot(valid_loss, label='valid_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{output_folder}/{train_name}.png')
    plt.clf()