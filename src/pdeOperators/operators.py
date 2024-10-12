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
    dim = input.shape[1]
    batchSize = input.shape[0]
    outf = f(inputSpace, inputTime)
    laplacian = torch.zeros(batchSize, dtype=float, requires_grad=True)

    dx = torch.autograd.grad(outf, input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0]

    for i in range(dim):
        d2x = torch.autograd.grad((dx[:,i]).view(-1,1), input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0][:,i]
        laplacian = laplacian + d2x

    return laplacian.view(-1,1)

    

def partialDerivative(f, idx, input):
    batchSize = input.shape[0]
    outf = f(input)

    dx = (torch.autograd.grad(outf, input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0][:,idx]).view(-1,1)

    return dx

def gradient(f, input):
    batchSize = input.shape[0]
    outf = f(input)
    
    gradient = torch.autograd.grad(outf, input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)

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

