import torch

def linearRadial(x):
    return (1-x).view(-1,1)

def squareRoot(x):
    return (1-torch.sqrt(x)).view(-1,1)

def squaredRadial(x):
    return (1 - x*x).view(-1,1)

