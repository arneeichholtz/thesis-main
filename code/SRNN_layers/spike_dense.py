import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
from SRNN_layers_AE.spike_neuron import * #mem_update_adp, output_Neuron

b_j0 = b_j0_value

class spike_dense(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tauM = 20,tauAdp_inital =200, tau_initializer = 'normal',tauM_inital_std = 5,tauAdp_inital_std = 5,
                 is_adaptive=1,device='cpu',bias=True):
        super(spike_dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device
        self.b_j0 = b_j0
        
        self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim))
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
            nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m,tauM,tauM_inital_std)
            self.tau_adp = multi_normal_initilization(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
    
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.tau_m,self.tau_adp]
    
    def set_neuron_state(self,batch_size):
        # self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0).to(self.device)
        self.mem = Variable(torch.zeros(batch_size,self.output_dim)*self.b_j0).to(self.device)
        self.spike = Variable(torch.zeros(batch_size,self.output_dim)).to(self.device)
        self.b = Variable(torch.ones(batch_size,self.output_dim)*self.b_j0).to(self.device)
    
    def forward(self,input_spike):
        d_input = self.dense(input_spike.float())
        self.mem,self.spike,theta,self.b = mem_update_adp(d_input,self.mem,self.spike,self.tau_adp,self.b,self.tau_m,device=self.device,isAdapt=self.is_adaptive)
        
        return self.mem,self.spike

class spike_Bidense(nn.Module):
    def __init__(self,input_dim1,input_dim2,output_dim,
                 tauM = 20,tauAdp_inital =100, tau_initializer = 'normal',tauM_inital_std = 5,tauAdp_inital_std = 5,
                 is_adaptive=1,device='cpu'):
        super(spike_Bidense, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device
        
        self.dense = nn.Bilinear(input_dim1,input_dim2,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim))
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
            nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m,tauM,tauM_inital_std)
            self.tau_adp = multi_normal_initilization(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
    
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.tau_m,self.tau_adp]
    
    def set_neuron_state(self,batch_size):
        self.mem = (torch.rand(batch_size,self.output_dim)*b_j0).to(self.device)
        self.spike = torch.zeros(batch_size,self.output_dim).to(self.device)
        self.b = (torch.ones(batch_size,self.output_dim)*b_j0).to(self.device)
    
    def forward(self,input_spike1,input_spike2):
        d_input = self.dense(input_spike1.float(),input_spike2.float())
        self.mem,self.spike,theta,self.b = mem_update_adp(d_input,self.mem,self.spike,self.tau_adp,self.b,self.tau_m,device=self.device,isAdapt=self.is_adaptive)
        
        return self.mem,self.spike

    
class readout_integrator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 tauM = 20, tau_initializer = 'normal', tauM_inital_std = 5, device='cpu', bias=True):
        super(readout_integrator, self).__init__()
        self.input_dim = input_dim # 200
        self.output_dim = output_dim # 2
        self.device = device
        
        self.dense = nn.Linear(input_dim, output_dim, bias=bias) # linear mapping 200 -> 2
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim)) # tau_m is placeholder for output_dim=2 values
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m, tauM, tauM_inital_std)
            
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.tau_m]
    
    def set_neuron_state(self, batch_size):
        self.mem = (torch.zeros(batch_size, self.output_dim)).to(self.device) # mem shape: (batch size, output dim)
    
    # This is called for self.dense_mean(mean_tensor/5)
    def forward(self, input_spike): # input_spike shape: (64, 200)
        d_input = self.dense(input_spike.float()) # d_input shape: output (batch size, 2)
        self.mem = output_Neuron(d_input, self.mem, self.tau_m, device=self.device) # self.mem is updated without spike
        return self.mem

def multi_normal_initilization(param, means=[10,200],stds = [5,20]):
    shape_list = param.shape # param is tensor[output_dim], so shape: output_dim
    if len(shape_list) == 1:
        num_total = shape_list[0] # num_total = 256
    elif len(shape_list) == 2:
        num_total = shape_list[0]*shape_list[1]

    num_per_group = int(num_total/len(means)) # int(256 / 4) = 64
    # if num_total%len(means) != 0: 
    num_last_group = num_total%len(means) # 256 % 4 = 0
    a = []
    for i in range(len(means)): # range(4)
        a = a + np.random.normal(means[i],stds[i],size=num_per_group).tolist() # sample 64 values from normal distribution
        # the lists are concatenated, so the result contains len(means) * num_per_group values (or num_total)
        
        if i == len(means): # looks like this code is never reached, i goes until len(means) - 1.
            a = a + np.random.normal(means[i],stds[i],size=num_per_group+num_last_group).tolist()
    p = np.array(a).reshape(shape_list) # p contains the 
    with torch.no_grad():
        param.copy_(torch.from_numpy(p).float())
    return param