import numpy as np
from dataset_AE import LRADataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch


def get_test_data(batch_size, drop_last):
    X = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    target = np.array([0, 1, 1, 1, 1, 0, 0, 0])

    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(target)

    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
    return enumerate(loader)

if __name__ == "__main__":

    # task = "text"
    # int_size = 8
    # seq_len = 2048
    # ds_size = 100
    # batch_size = 16
    
    # ds_iter = {
    #     "train":enumerate(DataLoader(LRADataset(f"../datasets_AE/{task}_bin{int_size}_{seq_len}_{ds_size}.train.pickle", True), batch_size=batch_size, drop_last = True))
    # }

    # _, batch = next(ds_iter['train'])
    # print(batch)

    loader = get_test_data(batch_size=8, drop_last=True)
    
    _, batch = next(loader)
    print(batch[0])
    print(batch[1])


