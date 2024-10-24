import torch


def partialDerivative(outf, input):
    inputDim = input.shape[1]
    if inputDim == 1:
        pass
    else:
        raise Exception("Input shape for partialDerivative is larger than one.")
    batchSize = input.shape[0]
    outDim = outf.shape[1]
    derivative = torch.zeros((batchSize, outDim), dtype=float, requires_grad=True)
    for i in range(outDim):
        dx = torch.autograd.grad(outf[:,i].view(-1,1), input, torch.ones((batchSize, 1), requires_grad=True), allow_unused=True, create_graph=True)[0]
        mask = torch.zeros_like(derivative, dtype= float)
        if dx == None:
            dx = torch.zeros((batchSize,1))
        mask[:,i] = 1
        derivative =   dx.tile((1,outDim)) *mask  + derivative

    return derivative




def laplacian(outf, input):
    outputDim = outf.shape[1]
    if outputDim == 1:
        pass
    else:
        raise Exception("Laplace of vector valued function not defined.")
    dim = len(input)
    batchSize = input[0].shape[0]
    #outf = f(input)
    laplacian = torch.zeros((batchSize,1), dtype=float, requires_grad=True)

    for i in range(dim):
        dx = partialDerivative(outf, input[i])
        d2x = partialDerivative(dx, input[i])
        laplacian = laplacian + d2x
    return laplacian

def gradient(outf,input):
    outputDim = outf.shape[1]
    if outputDim == 1:
        pass
    else:
        raise Exception("Gradient of vector valued function not defined.")
    dim = len(input)
    batchSize = input[0].shape[0]
    gradient = torch.zeros((batchSize, dim), dtype=float, requires_grad=True)
    for i in range(dim):
        dx = partialDerivative(outf, input[i])
        mask = torch.zeros_like(gradient)
        mask[:,i] = 1
        gradient = gradient + mask *  dx.tile((1,dim))
    return gradient


def divergence(outf, input):
    dim = len(input)
    batchSize = input[0].shape[0]
    div = torch.torch.zeros((batchSize, 1), dtype=float, requires_grad=True)
    for i in range(dim):
        dx = partialDerivative(outf[:,i].view(-1,1), input[i])
        div = div + dx
    return div

