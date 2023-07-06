import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

surrograte_type = 'MG'
print('gradient type: ', surrograte_type)

gamma = 0.5
lens = 0.5
R_m = 1

beta_value = 0.184 # the parameter beta controls the size of adaptation of the threshold
b_j0_value = 1.6

# beta_value = 1.8
# b_j0_value = .1

# beta_value = .2#1.8
# b_j0_value = .1

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

# define approximate firing function

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential - threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ??? 
    
        # ctx is context
        # gt does greater than element-wise

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .4
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma
    
    
act_fun_adp = ActFun_adp.apply 
    
def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1, device=None):
    """
    This function updates the membrane potential and adaptation variable of a spiking neural network.
Inputs:
inputs: the input spikes to the neuron
mem: the current membrane potential of the neuron
spike: the current adaptation variable of the neuron
tau_adp: the time constant for the adaptation variable
b: a value used in the adaptation variable update equation
tau_m: the time constant for the membrane potential
dt: the time step used in the simulation
isAdapt: a boolean variable indicating whether or not to use the adaptation variable
device: a variable indicating which device (e.g. CPU or GPU) to use for the computation

Outputs:
mem: the updated membrane potential
spike: the updated adaptation variable
B: a value used in the adaptation variable update equation
b: the updated value of the adaptation variable

The function first computes the exponential decay factors alpha and ro using the time constants tau_m and tau_adp, respectively.
It then checks whether the isAdapt variable is True or False to determine the value of beta.
The adaptation variable b is then updated using the exponential decay rule, and B is computed using the value of beta and the initial value b_j0_value.
The function then updates the membrane potential mem using the input spikes, B, and the decay factor alpha, and computes the inputs_ variable as the difference between mem and B.
Finally, the adaptation variable spike is updated using the activation function defined in the act_fun_adp() function, and the updated values of mem, spike, B, and b are returned.
    """
    # inputs dense + recurrent on normalized input, which now is just the normalized input. So input shape: (128, 256)
    # mem = zeros(128, 256) -- spike = zeros(128, 256) -- b = ones(128, 256)
    # tau_m is tensor with 256 samples from normal distribution
    # tau_adp is a tensor with 256 samples from normal distribution

    alpha = torch.exp(-1. * dt / tau_m).to(device) # alpha shape: (256) -- alpha is used for updating membrane potential
    ro = torch.exp(-1. * dt / tau_adp).to(device) # ro shape: (256) -- ro is used for updating b
    if isAdapt:
        beta = beta_value # beta_value set in beginning
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike # b shape: (128, 256) -- b is adaptation variable 
    B = b_j0_value + beta * b # B shape: (128, 256) -- b_j0_value and beta are just scalars
    
    # alpha and ro have values a bit smaller than 1. (at least for mean=20; then exp(-1 * 1 / 20) is close to 1)
    
    mem = alpha * mem + (1 - alpha) * R_m * inputs - B * spike * dt
    # alpha has just floats, R_m is a scalar, inputs is between 0 and 1. So mem contains floats around 1.
    
    inputs_ = mem - B # inputs_ shape: (128, 256)
    # spike = F.relu(inputs_)
    spike = act_fun_adp(inputs_) # act_fun_adp returns greater than 0 for all elements in inputs_
    return mem, spike, B, b

def output_Neuron(inputs, mem, tau_m, dt=1, device=None):
    """
    The read out neuron is leaky integrator without spike
    """
    # inputs shape: (64, 2) -- inputs is the concatenated fw and bw spikes mapped to two output neurons
    # mem shape: (64, 2)
    # tau_m shape: (2)
    alpha = torch.exp(-1. * dt / tau_m).to(device) # alpha shape: (2)
    mem = alpha * mem + (1-alpha) * inputs
    return mem

    # mem is updated without spike. Not fully understanding this



