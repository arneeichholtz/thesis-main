import torch
import torch.nn as nn
import math
from torch.utils.data.dataset import Dataset
import sys
import os
import random
import json
import pickle
import numpy as np

class LRADataset(Dataset):
    def __init__(self, file_path, endless):

        self.endless = endless
        with open(file_path, "rb") as f:
            self.examples = pickle.load(f)
            random.shuffle(self.examples)
            self.curr_idx = 0
            
        print(f"Loaded {file_path}... size={len(self.examples)}", flush = True)

    def __len__(self):
        if self.endless:
            return 1000000000
        else:
            return len(self.examples)

    def create_inst(self, inst):
        output = {}
        output["input_bin"] = torch.tensor(inst["input_bin"], dtype = torch.long)
        output["target"] = torch.tensor(inst["target"], dtype = torch.long)
        return output
    
    def __getitem__(self, i):
        if not self.endless:
            return self.create_inst(self.examples[i])
        
        if self.curr_idx >= len(self.examples):
            random.shuffle(self.examples)
            self.curr_idx = 0
        inst = self.examples[self.curr_idx]
        self.curr_idx += 1
        
        return self.create_inst(inst)