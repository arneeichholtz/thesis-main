import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from SRNN_layers_AE.spike_neuron import * #mem_update_adp
from SRNN_layers_AE.spike_dense import *

b_j0 = b_j0_value
class spike_rnn(nn.Module):
    def __init__(self,input_dim,output_dim, 
                 tauM = 20,tauAdp_inital =100, tau_initializer = 'normal',tauM_inital_std = 5,tauAdp_inital_std = 5,
                 is_adaptive=1,device='cpu',bias=True):
        super(spike_rnn, self).__init__()
        self.input_dim = input_dim # input_dim = 8
        self.output_dim = output_dim # output_dim = 100
        self.is_adaptive = is_adaptive # = 1
        self.device = device

        self.b_j0 = b_j0 # = 1.6
        self.dense = nn.Linear(input_dim, output_dim, bias=bias) # nn.Linear is linear dense layer mapping: input_dim -> output_dim = 8 -> 100
        self.recurrent = nn.Linear(output_dim, output_dim, bias=bias) # linear mapping: output_dim -> output_dim = 100 -> 100
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim)) # tau_m is decay time-constant -- initialized as tensor[0, 0, ..., 0] with length output_dim
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim)) # tau_adp is adaptive decay factor, decaying exponentially -- initialized as tensor[0, 0, ..., 0] with length output_dim
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
            nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m, tauM, tauM_inital_std) # tauM = [20,20,20,20], tauM_inital_std = [1,5,5,5]
            # Now tau_m is a tensor with 256 (output_dim) samples from normal distribution, so shape is unchanged
            self.tau_adp = multi_normal_initilization(self.tau_adp,tauAdp_inital,tauAdp_inital_std) # tauAdp_inital = [200,200,250,200], tauAdp_inital_std = [5,50,100,50]
            # Now tau_Adp is a tensor with 256 (output_dim) samples from normal distribution, so shape is unchanged
    
    def parameters(self):
        return [self.dense.weight, self.dense.bias, self.recurrent.weight, self.recurrent.bias, self.tau_m, self.tau_adp]
    
    def set_neuron_state(self, batch_size):
        self.mem = Variable(torch.zeros(batch_size, self.output_dim)*self.b_j0).to(self.device) # mem shape: (64, 100)
        self.spike = Variable(torch.zeros(batch_size, self.output_dim)).to(self.device) # spike shape: (batch size=64, 100)
        self.b = Variable(torch.ones(batch_size, self.output_dim)*self.b_j0).to(self.device) # b shape: (64, 100)

    def forward(self, input_spike): # input_spike shape: (64, 8)
        d_input = self.dense(input_spike) + self.recurrent(self.spike) # + does addition in-place, so d_input shape: (64, 100)
        self.mem, self.spike, theta, self.b = mem_update_adp(d_input, self.mem, self.spike, self.tau_adp, self.b, self.tau_m, device=self.device, isAdapt=self.is_adaptive)
        return self.mem, self.spike
    
    
