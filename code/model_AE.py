import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from SRNN_layers_AE.spike_dense import * # spike_dense, readout_integrator
from SRNN_layers_AE.spike_neuron import * # output_Neuron
from SRNN_layers_AE.spike_rnn import * # spike_rnn

class RNN_s(nn.Module):
    def __init__(self, criterion, device, input_dim=8, output_dim=2, delay=0): 
        super(RNN_s, self).__init__()
        self.criterion = criterion
        self.delay = delay
        self.network = [input_dim, 100, 100, output_dim] # 8-dim binary input and 100-dim hidden state is as Towards paper
        
        self.rnn_fw1 = spike_rnn(self.network[0], self.network[1], # 8, 100
                               tau_initializer='multi_normal',
                               tauM=[20,20,20,20],tauM_inital_std=[1,5,5,5],
                               tauAdp_inital=[200,200,250,200],tauAdp_inital_std=[5,50,100,50],
                               device=device)
        
        self.rnn_bw1 = spike_rnn(self.network[0], self.network[2], # 8, 100
                                tau_initializer='multi_normal',
                                tauM=[20,20,20,20],tauM_inital_std=[5,5,5,5],
                                tauAdp_inital=[200,200,150,200],tauAdp_inital_std=[5,50,30,10],
                                device=device)
        
        self.dense_mean = readout_integrator(self.network[2]+self.network[1], self.network[3], # 200, 2
                                    tauM=3, tauM_inital_std=1, device=device)
        
        print("Spiking RNN with a single layer initialized")

    # Reduced function, ie, removed prediction at every time-step
    def forward(self, input_bin, target=None):
        b, s, c = input_bin.shape # b = 64, s = 2048, c = 8
        self.rnn_fw1.set_neuron_state(b)
        self.rnn_bw1.set_neuron_state(b)
        self.dense_mean.set_neuron_state(b)
        
        for l in range(s): 
            input_fw = input_bin[:, l, :]
            input_bw = input_bin[:, -l, :]
            mem_layer1, spike_layer1 = self.rnn_fw1.forward(input_fw) # updating for each step in the sequence
            mem_layer2, spike_layer2 = self.rnn_bw1.forward(input_bw)

        Y = torch.cat((spike_layer1, spike_layer2), -1) # concatenate the final spike output
        mem_layer3 = self.dense_mean(Y) # this maps (64, 200) -> (64, 2)

        output = F.log_softmax(mem_layer3, dim=-1) # bojian does log_softmax, what would be the difference?
        if target is not None:
            loss = self.criterion(reduction = "none")(output, target)
        accu = (output.argmax(dim = -1) == target).to(torch.float32)
        
        return loss, accu

    
    
