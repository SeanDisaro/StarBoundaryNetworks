import torch
from src.pdeOperators.operators import *



def burgersEquationLoss(u, xgrid, tgrid, nu , cuda = False):
    #burgers eq: u_t + u* nabla(u) - nu* nabla(nabla(u)) = 0
    partialTime = partialTimeDerivative(u, xgrid, tgrid)
    udiv = divergenceOnlySpace(u, xgrid, tgrid)
    uGradOfDiv = gradientOfDivergenceOnlySpace(u, xgrid, tgrid)
    uout = u(xgrid, tgrid)
    return torch.mean(torch.norm(partialTime + uout * udiv - nu* uGradOfDiv, dim = 1))



def PoissonEquationLoss(u, xgrid, poissonData, cuda = False):
    #poisson eq: laplace(u)+ poissonDataFunction = 0
    laplacian_u = laplacian(u, xgrid)
    dataOut = poissonData(xgrid)
    return torch.mean(torch.norm(laplacian_u + dataOut, dim = 1))


