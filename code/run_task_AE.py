# Author: Arne Eichholtz
# example usage: python3 run_task_AE.py --seq_len 2048 --int_size 8 --ds_size 100

from dataset_AE import LRADataset, TestDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch
import time
import os
import numpy as np
import argparse
import itertools
import config
import pandas as pd
import torch.nn as nn

from SRNN_layers_AE.spike_dense import * #spike_dense,readout_integrator
from SRNN_layers_AE.spike_neuron import * #output_Neuron
from SRNN_layers_AE.spike_rnn import * # spike_rnn

def step(component, step_idx, train_loss_sum):
    t0 = time.time()
    _, batch = next(ds_iter[component])
    # for key in batch:
    #     batch[key] = batch[key].cuda()
    
    if component == "train":
        input = batch["input_bin"].float().to(device)
        labels = batch["target"].long().to(device)
        # optimizer.zero_grad()
        # # forward
        # loss, accu = model(input, labels)
        # loss = loss.mean()
        # accu = accu.mean()
        # # backward
        # loss.backward()
        # optimizer.step()
        # scheduler.step()

        optimizer.zero_grad()
    
        predictions, loss = model(input, labels)
        _, predicted = torch.max(predictions.data, 2)
        
        train_loss_sum += loss
        loss.backward()
        optimizer.step()

        labels = labels.cpu()
        predicted = predicted.cpu().t()
        predicted = torch.reshape(predicted, (-1,))
        print(predicted)
        accu = np.array((labels == predicted), dtype=int).mean()
        
        torch.cuda.empty_cache()
        if scheduler is not None:
                scheduler.step()
    else:
        with torch.no_grad():
            input = batch["input_bin"].to(device)
            labels = batch["target"].to(device)

            predictions, loss = model(input, labels)
            _, predicted = torch.max(predictions.data, 2)

            labels = labels.cpu()
            predicted = predicted.cpu().t()
            predicted = torch.reshape(predicted, (-1,))
            accu = np.array((labels == predicted), dtype=int).mean()

            # loss, accu = model(input, labels) 
            # loss = loss.mean()
            # accu = accu.mean()
    
    t1 = time.time()
    t_batch = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    time_since_start = time.time() - start

    loss = loss.data.item()
    # accu = accu.data.item()
    
    print(f"step={step_idx}, t_total={time_since_start:.1f}, t_batch={t_batch:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.3f}, accu={accu:.3f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["step_idx"].append(step_idx)
    summary[component]["t_total"] += t_batch
    summary[component]["t_batch"].append(t_batch)
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)

def update_model(summary, train_step_idx):
    iters = num_valid_steps
    ds_accu = np.mean(summary["accu"][-iters:]) # mean of last num_valid_steps batches (ie, entire dataset)

    if ds_accu > summary["best_accu"]:
        summary["best_accu"] = ds_accu
        path = os.path.join(models_path, "srnn.model")
        torch.save({"model_state_dict": model.state_dict()}, path)
        print(f"Train step: {train_step_idx} -- Best accu: {ds_accu:.4f}. Saved best model", end = "\n")
    else:
        print(f"Dataset accuracy did not improve\nDataset accu: {ds_accu}\nBest accu: {summary['best_accu']}")

def stats_to_csv(component, step_idx, summary):
    if component == "test":
        path = os.path.join(stats_path, f"stats_{component}.csv")
    else:
        path = os.path.join(stats_path, f"stats_{component}_{step_idx}.csv")
    df = pd.DataFrame(summary[component])
    df.to_csv(path)
    print(f"Saved {component} to path: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", type = int, help = "Skip training", dest = "skip_train", default = 0)
    parser.add_argument("--skip_vt", type = int, help = "Skip validation and testing", dest = "skip_vt", default = 0)
    parser.add_argument("--seq_len", type = int, help = "Sequence length", dest = "seq_len", default = 4000)
    parser.add_argument("--int_size", type = int, help = "Size in bits of int in data", dest = "int_size", default = 32)
    parser.add_argument("--layers", type = int, help = "Number of layes in the network", dest = "layers", default = 1)
    parser.add_argument("--ds_size", type = int, help = "Size of dataset", dest = "ds_size", default = 1)
    args = parser.parse_args()

    # Making directories to store statistics and model
    log_path = os.path.join("../logs")
    stats_path = os.path.join(log_path, "stats", f"layers{args.layers}", "testing")
    models_path = os.path.join(log_path, "models", f"layers{args.layers}")
    try:
        os.mkdir(stats_path)
        os.mkdir(models_path)
    except FileExistsError:
        pass

    task = "text"
    seq_len = args.seq_len
    int_size = args.int_size
    ds_size = args.ds_size

    training_config = config.config["training"]
    num_train_steps = training_config["num_train_steps"]
    batch_size = training_config["batch_size"]

    train_acc = 0
    train_loss_sum = 0
    sum_samples = 0

    num_examples = int((1/ds_size) * 25000)
    num_valid_steps = int(num_examples / batch_size) # number of steps to validate entire dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Task: {task}\nSequence length: {seq_len}\nTraining steps: {num_train_steps}\nint size: {int_size}\nDataset size: 1/{ds_size}\nBatch size: {batch_size}\nDevice: {device}")

    if args.skip_vt == 1:
        input_dim = 1
        ds_iter = {
            "train": enumerate(DataLoader(TestDataset("../datasets_AE/train.pickle", True), batch_size=batch_size))
        }
        print("Test dataset loaded")
    else: # normal dataset is default
        input_dim = 8
        ds_iter = {
            "train":enumerate(DataLoader(LRADataset(f"../datasets_AE/{task}_bin{int_size}_{seq_len}_{ds_size}.train.pickle", True), batch_size=batch_size, drop_last = True)),
            "valid":enumerate(DataLoader(LRADataset(f"../datasets_AE/{task}_bin{int_size}_{seq_len}_{ds_size}.valid.pickle", True), batch_size=batch_size, drop_last = True)),
            "test":enumerate(DataLoader(LRADataset(f"../datasets_AE/{task}_bin{int_size}_{seq_len}_{ds_size}.test.pickle", False), batch_size = batch_size, drop_last = True)),
        }
    
    # Model, parameters, optimizer, learning-rate scheduler
    learning_rate = training_config["learning_rate"]
    if args.layers == 2:
        from model2_AE import RNN_s
        model = RNN_s(criterion=torch.nn.CrossEntropyLoss, device=device, input_dim=8, output_dim=2)
        model.to(device)

        base_params = [model.rnn_fw1.dense.weight, model.rnn_fw1.dense.bias,
                    model.rnn_fw1.recurrent.weight, model.rnn_fw1.recurrent.bias,
                    model.rnn_bw1.dense.weight, model.rnn_bw1.dense.bias,
                    model.rnn_bw1.recurrent.weight, model.rnn_bw1.recurrent.bias,
                    model.rnn_fw2.dense.weight, model.rnn_fw2.dense.bias,
                    model.rnn_fw2.recurrent.weight, model.rnn_fw2.recurrent.bias,
                    model.rnn_bw2.dense.weight, model.rnn_bw2.dense.bias,
                    model.rnn_bw2.recurrent.weight, model.rnn_bw2.recurrent.bias,
                    model.dense_mean.dense.weight, model.dense_mean.dense.bias]
        optimizer = torch.optim.Adagrad([{'params': base_params},
                                        {'params': model.rnn_fw1.tau_adp, 'lr': learning_rate * 5},
                                        {'params': model.rnn_bw1.tau_adp, 'lr': learning_rate * 5},
                                        {'params': model.rnn_fw1.tau_m, 'lr': learning_rate * 2},
                                        {'params': model.rnn_bw1.tau_m, 'lr': learning_rate * 2},
                                        {'params': model.rnn_fw2.tau_adp, 'lr': learning_rate * 5},
                                        {'params': model.rnn_bw2.tau_adp, 'lr': learning_rate * 5},
                                        {'params': model.rnn_fw2.tau_m, 'lr': learning_rate * 2},
                                        {'params': model.rnn_bw2.tau_m, 'lr': learning_rate * 2},
                                        {'params': model.dense_mean.tau_m, 'lr': learning_rate * 2}],
                                        lr=learning_rate, eps=1e-5)
    else: # Network with single layer is default
        from model_test import RNN_s
        model = RNN_s(criterion = nn.NLLLoss(), device=device, input_dim=8, output_dim=2)
        model.to(device)

        base_params = [model.rnn_fw1.dense.weight, model.rnn_fw1.dense.bias,
                    model.rnn_fw1.recurrent.weight, model.rnn_fw1.recurrent.bias,
                    model.rnn_bw1.dense.weight, model.rnn_bw1.dense.bias,
                    model.rnn_bw1.recurrent.weight, model.rnn_bw1.recurrent.bias,
                    model.dense_mean.dense.weight, model.dense_mean.dense.bias]
        optimizer = torch.optim.Adagrad([{'params': base_params},
                                        {'params': model.rnn_fw1.tau_adp, 'lr': learning_rate * 5},
                                        {'params': model.rnn_bw1.tau_adp, 'lr': learning_rate * 5},
                                        {'params': model.rnn_fw1.tau_m, 'lr': learning_rate * 2},
                                        {'params': model.rnn_bw1.tau_m, 'lr': learning_rate * 2},
                                        {'params': model.dense_mean.tau_m, 'lr': learning_rate * 2}],
                                        lr=learning_rate, eps=1e-5)
    scheduler = StepLR(optimizer, step_size=100, gamma=.5)
    optimizer = torch.optim.Adamax([{'params': base_params}], lr=learning_rate)

    # Summary statistics dictionary
    summary = {
        component:{"step_idx": [], "t_total":0, "t_batch": [], "loss":[], "accu":[], "best_accu":0, "component":component} for component in ["train", "valid", "test"]
    }

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # 22,000
    
    start = time.time()
    if args.skip_train == 0:
        try:
            model.train()
            for train_step_idx in range(num_train_steps):
                step("train", train_step_idx, train_loss_sum)
                if args.skip_vt == 0:
                    if (train_step_idx + 1) % training_config["eval_frequency"] == 0: 
                        print("\nEvaluating model...")
                        model.eval()
                        for valid_step_idx in range(num_valid_steps):
                            step("valid", valid_step_idx, train_loss_sum)
                        update_model(summary["valid"], train_step_idx)
                        for component in ["train", "valid"]:
                            stats_to_csv(component, train_step_idx, summary)
                        train_acc = 0
                        train_loss_sum = 0
                        sum_samples = 0
                        model.train()
                else:
                    stats_to_csv("train", train_step_idx, summary)

        except KeyboardInterrupt as e:
            print(e)

    if args.skip_vt == 1:
        print("Done training")
    else:
        print("\nTesting model...")
        train_acc = 0
        train_loss_sum = 0
        checkpoint = torch.load(os.path.join(models_path, "srnn.model"), map_location = "cpu")
        # checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location = "cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        try:
            for test_step_idx in itertools.count(): # itertools is used with try/except because then it runs until an error arises, ie, when the dataset has been traversed
                step("test", test_step_idx, train_loss_sum)
        except StopIteration:
            stats_to_csv("test", 0, summary)

    