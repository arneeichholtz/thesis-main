from dataset_AE import LRADataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch
import torch.nn as nn
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from SRNN_layers_AE.spike_dense import * #spike_dense,readout_integrator
from SRNN_layers_AE.spike_neuron import * #output_Neuron
from SRNN_layers_AE.spike_rnn import * # spike_rnn
from model import RNN_s

def print_sums():
        for name, p in model.named_parameters():
            print(f"Name: {name} - Sum: {torch.sum(p).data.item():.5f}")

def test(test_loader):
    print("Testing model...")
    test_lst = list(test_loader)
    acc_list = []
    for batch in test_lst:
        input = batch["input_bin"].to(device)
        labels = batch["target"].long().to(device)
        output, _ = model(input, labels)
        predictions = torch.argmax(output, dim=-1)
        labels = labels.cpu()
        predictions = predictions.cpu()

        batch_acc = np.array((predictions == labels), dtype=int).mean()
        acc_list.append(batch_acc)

    test_accu = np.mean(acc_list)
    return test_accu

def train(model, train_loader, valid_loader, optimizer, scheduler=None, num_epochs=10):
    print("Training model...")  
    train_lst = list(train_loader)
    model_path = "../logs/models/layers1/"
    summary_path = "../logs/stats/layers1/"
    acc_list = []
    loss_list = []
    best_acc = 0

    for epoch in range(num_epochs):
        # print_sums()
        t0 = time.time()
        epoch_loss_list, epoch_acc_list = [], [] # to store loss/accuracy for batches in an epoch
        pred_list = []
        for batch in tqdm(train_lst):
            input = batch["input_bin"].to(device)
            labels = batch["target"].long().to(device)

            # forward
            output, train_loss = model(input, labels) # train_loss is scalar because of reduction='mean'
            predictions = torch.argmax(output, dim=-1)
            
            # backward
            optimizer.zero_grad()
            train_loss.backward() # this computes the gradients
            optimizer.step() # updates params using grad attribute of params

            labels = labels.cpu() # necessary because labels and predictions are on GPU
            predictions = predictions.cpu()
            pred_list.append(predictions)

            batch_acc = np.array((predictions == labels), dtype=int).mean()
            epoch_acc_list.append(batch_acc)
            epoch_loss_list.append(train_loss.data.item())
            torch.cuda.empty_cache()
        
        if (epoch + 1) % 10 == 0:
            df = pd.DataFrame(summary)
            out_path = os.path.join(summary_path, f"stats_method{method}_sched{schedul_str}_initlr{learning_rate}_fact{fact}_step{step}_bs{batch_size}_ds{ds_size}_ep{epoch}-{num_epochs}.csv")
            df.to_csv(out_path)
            print(f"Saved summary to csv. Path: {out_path}")
        
        train_acc = np.mean(epoch_acc_list)
        valid_acc = test(valid_loader)
        train_loss = np.mean(epoch_loss_list)

        if scheduler is not None:
            if(isinstance(scheduler, ReduceLROnPlateau)):
                scheduler.step(train_loss)
            else:
                scheduler.step()
    
        if valid_acc>best_acc and train_acc>0.30:
            best_acc = valid_acc
            out_path = os.path.join(model_path, str(best_acc)[:7]+'-bi-srnn.pth')
            torch.save(model, out_path)
            print(f"Accu improved. Best accu: {best_acc} -- model saved to path {out_path} ")

        acc_list.append(train_acc)
        loss_list.append(train_loss)
        t_dataset = time.time() - t0

        print(f't_epoch: {t_dataset:.1f} - epoch: {epoch:3d} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Valid Acc: {valid_acc:.4f}', flush=True)

        summary["t_epoch"].append(t_dataset)
        summary["epoch"].append(epoch)
        summary["train_loss"].append(train_loss)
        summary["train_acc"].append(train_acc)
        summary["valid_acc"].append(valid_acc)
        summary["predictions"].append(pred_list)
    
    return acc_list, loss_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_size", type = int, help = "Size of dataset", dest = "ds_size", default = 100)
    parser.add_argument("--batch_size", type = int, help = "Batch size", dest = "batch_size", default = 16)
    parser.add_argument("--epochs", type = int, help = "Number of training epochs", dest = "epochs", default = 10)
    parser.add_argument("--scheduler", type = str, help = "Learning Rate Scheduler", dest = "scheduler", default = "step")
    args = parser.parse_args()

    torch.manual_seed(0)

    ds_size = args.ds_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    schedul_str = args.scheduler

    method = 2

    print(f"Epochs: {num_epochs}\nBatch size: {batch_size}\nDataset size: 1/{ds_size}\nScheduler: {schedul_str}\nMethod: {method}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("device name: ", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
    
    print("device:", device)

    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss(reduction = "mean") # reduction mean is default
    model = RNN_s(criterion=criterion, device=device, method=method, input_dim=8, output_dim=2)
    model.to(device)

    base_params = [model.rnn_fw1.dense.weight,model.rnn_fw1.dense.bias,
                  model.rnn_fw1.recurrent.weight,model.rnn_fw1.recurrent.bias,
                  model.rnn_bw1.dense.weight,model.rnn_bw1.dense.bias,
                  model.rnn_bw1.recurrent.weight,model.rnn_bw1.recurrent.bias,
                  model.dense_mean.dense.weight,model.dense_mean.dense.bias]
    
    learning_rate = 0.1
    optimizer = torch.optim.Adamax([{'params': base_params}], lr=learning_rate)
    # # optimizer = torch.optim.Adagrad([{'params': base_params},
    #                                     {'params': model.rnn_fw1.tau_adp, 'lr': learning_rate * 5},
    #                                     {'params': model.rnn_bw1.tau_adp, 'lr': learning_rate * 5},
    #                                     {'params': model.rnn_fw1.tau_m, 'lr': learning_rate * 2},
    #                                     {'params': model.rnn_bw1.tau_m, 'lr': learning_rate * 2},
    #                                     {'params': model.dense_mean.tau_m, 'lr': learning_rate * 2}],
    #                                     lr=learning_rate, eps=1e-5)

    fact = 0.1
    step = 5
    print(f"Factor: {fact}\nInitial learning rate: {learning_rate}\nUpdate step for Scheduler: {step}")

    if schedul_str == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=step, factor=fact, verbose=True)
    else:
        scheduler = StepLR(optimizer, step_size=step, gamma=fact)
    
    train_loader = DataLoader(LRADataset(f"../datasets_AE/text_bin8_2048_{ds_size}.train.pickle", False), batch_size=batch_size, drop_last = False)
    test_loader = DataLoader(LRADataset(f"../datasets_AE/text_bin8_2048_{ds_size}.test.pickle", False), batch_size=batch_size, drop_last = False)
    valid_loader = test_loader
    
    summary = {"t_epoch": [], "epoch": [], "predictions": [], "train_loss": [], "train_acc": [], "valid_acc": []}

    
    train_acc_list, train_loss_list = train(model, train_loader, valid_loader, optimizer, scheduler, num_epochs=num_epochs)
    print("Done training")

    
    test_acc = test(test_loader)
    df = pd.DataFrame([{"testing accuracy": test_acc}])
    test_acc_path = os.path.join("../logs/stats/layers1/", f"testaccu_method{method}_sched{schedul_str}_initlr{learning_rate}_fact{fact}_step{step}_bs{batch_size}_ds{ds_size}_ep{num_epochs}.csv")
    df.to_csv(test_acc_path)
    print(f"Model tested.\nTesting accuracy: {test_acc:.4f}")
    
    
    