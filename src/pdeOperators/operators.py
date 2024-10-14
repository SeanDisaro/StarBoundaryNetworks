import torch



def laplacian(f, input):
    dim = input.shape[1]
    batchSize = input.shape[0]
    outf = f(input)
    laplacian = torch.zeros(batchSize, dtype=float, requires_grad=True)

    dx = torch.autograd.grad(outf, input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0]

    for i in range(dim):
        d2x = torch.autograd.grad((dx[:,i]).view(-1,1), input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0][:,i]
        laplacian = laplacian + d2x

    return laplacian.view(-1,1)

def laplacianOnlySpace(f, inputSpace, inputTime):
    '''
    Computes laplcian of f(x,t) wrt x.
    
    Args:
        f
        input: 
    '''
    dim = inputSpace.shape[1]
    batchSize = inputSpace.shape[0]
    outf = f(inputSpace, inputTime)
    laplacian = torch.zeros(batchSize, dtype=float, requires_grad=True)

    dx = torch.autograd.grad(outf, inputSpace, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0]

    for i in range(dim):
        d2x = torch.autograd.grad((dx[:,i]).view(-1,1), inputSpace, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0][:,i]
        laplacian = laplacian + d2x

    return laplacian.view(-1,1)

    

def partialDerivative(f, idx, input):
    batchSize = input.shape[0]
    outf = f(input)
    outDim = outf.shape[1]
    derivative = torch.zeros((batchSize, outDim), dtype=float, requires_grad=True)
    for i in range(outDim):
        dx = torch.autograd.grad(outf[:,i].view(-1,1), input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0]
        mask = torch.zeros_like(derivative, dtype= float)
        mask[:,i] = 1
        derivative =   dx[:, idx].tile((outDim,1)).transpose(0,1) *mask  + derivative

    return derivative

def partialTimeDerivative(f, xgrid, tgrid):
    batchSize = xgrid.shape[0]
    totalGrid = torch.cat((xgrid,tgrid), 1)
    outf = f(totalGrid[:,:-1], totalGrid[:,-1].view(-1,1))
    outDim = outf.shape[1]
    derivative = torch.zeros((batchSize, outDim), dtype=float, requires_grad=True)
    for i in range(outDim):
        dx = torch.autograd.grad(outf[:,i].view(-1,1), totalGrid, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0]
        mask = torch.zeros_like(derivative, dtype= float)
        mask[:,i] = 1
        
        derivative =   dx[:, -1].tile((outDim,1)).transpose(0,1) *mask  + derivative

    return derivative


def gradient(f, input):
    batchSize = input.shape[0]
    outf = f(input)
    
    gradient = torch.autograd.grad(outf, input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)

    return gradient

def gradientOnlySpace(f, xgrid, tgrid):
    batchSize = xgrid.shape[0]
    outf = f(xgrid, tgrid)
    
    gradient = torch.autograd.grad(outf, xgrid, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)

    return gradient

def divergence(f, input):
    dim = input.shape[1]
    batchSize = input.shape[0]
    outf = f(input)
    div = torch.torch.zeros((batchSize, 1), dtype=float, requires_grad=True)

    for i in range(dim): 
        dx = torch.autograd.grad(outf[:,i].view(-1,1), input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0]
        div = dx[:,i].view(-1,1) + div
    return div

def divergenceOnlySpace(f, xgrid, tgrid):
    dim = xgrid.shape[1]
    batchSize = xgrid.shape[0]
    outf = f(xgrid, tgrid)
    div = torch.torch.zeros((batchSize, 1), dtype=float, requires_grad=True)

    for i in range(dim):
        dx = torch.autograd.grad(outf[:,i].view(-1,1), xgrid, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0]
        div = dx[:,i].view(-1,1) + div
    return div

def gradientOfDivergenceOnlySpace(f, xgrid, tgrid):
    dim = xgrid.shape[1]
    batchSize = xgrid.shape[0]
    outf = f(xgrid, tgrid)
    div = torch.torch.zeros((batchSize, 1), dtype=float, requires_grad=True)

    for i in range(dim): 
        dx = torch.autograd.grad(outf[:,i].view(-1,1), xgrid, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0]
        div = dx[:,i].view(-1,1) + div
    
    gradient = torch.autograd.grad(div, xgrid, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)

    return gradient