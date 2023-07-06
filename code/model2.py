import torch

from SRNN_layers_AE.spike_dense import * #spike_dense,readout_integrator
from SRNN_layers_AE.spike_neuron import * #output_Neuron
from SRNN_layers_AE.spike_rnn import * # spike_rnn

class RNN_s(nn.Module):
    def __init__(self, criterion, device, input_dim=8, output_dim=2, delay=0):
        super(RNN_s, self).__init__()
        self.criterion = criterion
        self.delay = delay
        
        # self.network = [input_dim,128,128,256,output_dim]
        # self.network = [input_dim, 100, 100, output_dim]
        self.network = [input_dim, 100, 100, 100, 100, 100, output_dim]

        self.rnn_fw1 = spike_rnn(self.network[0],self.network[1],
                               tau_initializer='multi_normal',
                               tauM=[20,20,20,20],tauM_inital_std=[1,5,5,5],
                               tauAdp_inital=[200,200,250,200],tauAdp_inital_std=[5,50,100,50],
                               device=device)
        
        self.rnn_bw1 = spike_rnn(self.network[0],self.network[2],
                                tau_initializer='multi_normal',
                                tauM=[20,20,20,20],tauM_inital_std=[5,5,5,5],
                                tauAdp_inital=[200,200,150,200],tauAdp_inital_std=[5,50,30,10],
                                device=device)
        
        self.rnn_fw2 = spike_rnn(self.network[3],self.network[4],
                               tau_initializer='multi_normal',
                               tauM=[20,20,20,20],tauM_inital_std=[1,5,5,5],
                               tauAdp_inital=[200,200,250,200],tauAdp_inital_std=[5,50,100,50],
                               device=device)
        
        self.rnn_bw2 = spike_rnn(self.network[3],self.network[5],
                                tau_initializer='multi_normal',
                                tauM=[20,20,20,20],tauM_inital_std=[5,5,5,5],
                                tauAdp_inital=[200,200,150,200],tauAdp_inital_std=[5,50,30,10],
                                device=device)
        
        self.dense_mean = readout_integrator(self.network[4]+self.network[5], self.network[6],
                                    tauM=3,tauM_inital_std=1,device=device)
    print("Model with 2 layers initialized")
    
    def forward(self, input, labels=None):
        b, s, _ = input.shape
        self.rnn_fw1.set_neuron_state(b)
        self.rnn_bw1.set_neuron_state(b)
        self.rnn_fw2.set_neuron_state(b)
        self.rnn_bw2.set_neuron_state(b)
        self.dense_mean.set_neuron_state(b)
        fw1_spikes = []
        bw1_spikes = []

        for l in range(s):
            input_fw=input[:, l, :].float()
            input_bw=input[:, -l-1, :].float() # -1 because backward iters should start at -1, not 0

            mem_fw1, spike_fw1 = self.rnn_fw1.forward(input_fw)
            mem_bw2, spike_bw1 = self.rnn_bw1.forward(input_bw)
            fw1_spikes.append(spike_fw1) # spike_layer shape: (batch size, 100)
            bw1_spikes.append(spike_bw1) # -l already traverses the input in the backward direction; inserting in beginning of list negates this
            # bw_spikes.insert(0, spike_bw)

            mem_fw2, spike_fw2 = self.rnn_fw2.forward(spike)

        
        for k in range(s):
            merge_spikes = torch.cat((fw_spikes[k], bw_spikes[k]), -1) # merge_spikes shape: (batch size, 200)
            mem_layer3 = self.dense_mean(merge_spikes) # mem_layer3 is not used here, it is used internally for the neuron
        
        # This is redundant, because the last value of mem_layer3 is already the membrane potential for the final output spikes
        # Y = torch.cat((spike_fw, spike_bw), -1) # after the for loop, these are the final two spikes
        # mem_layer3 = self.dense_mean(Y) # dense_mean maps (batch size, 200) -> (batch size 2)
        
        output = F.log_softmax(mem_layer3, dim=-1)
        if labels is not None:
            loss = self.criterion(output, labels)
    
        output = output.data.cpu()
        return output, loss
        
        