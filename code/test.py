from dataset_AE import LRADataset
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch
import torch.nn as nn
import time
import os
import numpy as np
import argparse
import itertools
import config
import pandas as pd
import pickle

from SRNN_layers_AE.spike_dense import * #spike_dense,readout_integrator
from SRNN_layers_AE.spike_neuron import * #output_Neuron
from SRNN_layers_AE.spike_rnn import * # spike_rnn
from model_test import RNN_s

def create_test_data():
    X = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    target = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    
    ds_list = []
    for i, data_point in enumerate(X):
        ds_list.append({
            'input_bin': data_point,
            'target': target[i]
        })
    with open(f"../datasets_AE/train.pickle", "wb") as f:
        pickle.dump(ds_list, f)


if __name__ == "__main__":

    # labels = torch.Tensor([1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    # pred = torch.Tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
    # pred = torch.reshape(pred, (-1,))
    
    # accu = np.array((labels == pred), dtype=int).mean()
    # print(accu)

    task = "text"
    int_size = 8
    seq_len = 2048
    ds_size = 100
    batch_size = 16
    
    # ds_iter = {
    #     "train":enumerate(DataLoader(LRADataset(f"../datasets_AE/{task}_bin{int_size}_{seq_len}_{ds_size}.train.pickle", False), batch_size=batch_size, drop_last = True))
    # }

    # train_loader = DataLoader(LRADataset(f"../datasets_AE/{task}_bin{int_size}_{seq_len}_{ds_size}.train.pickle", False), batch_size=batch_size, drop_last = True)

    loss_fn = torch.nn.CrossEntropyLoss()
    input = torch.Tensor([[0.45, 0.23]])
    target = torch.Tensor([1]).long()

    # loss = loss_fn(input, target)
    # print(loss)

    # test = np.log(np.exp(0.23) / (np.exp(0.45) + np.exp(0.23)))
    # print(test)

    input = "abc"
    fw, bw = [], []

    for i, c in enumerate(input):
        print(i)
        print(-i-1)
        





    
    
    


    
    
    
    
    

    

    
    


        




    

    


    
    
    