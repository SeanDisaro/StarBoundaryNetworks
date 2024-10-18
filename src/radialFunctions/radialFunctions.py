import torch

def linearRadial(x):
    return 1-x

def squareRoot(x):
    return 1-torch.sqrt(x)

def squaredRadial(x):
    return 1 - x*x

def fraqLog(x):
    if(x> 1/torch.e ):
        return 1
    else:
        return 1/torch.log(x)
