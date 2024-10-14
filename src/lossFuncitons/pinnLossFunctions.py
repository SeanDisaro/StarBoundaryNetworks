import torch
from src.pdeOperators.operators import *



#define PINN loss function for problem from above:
def burgersEquationLoss(u, xgrid, tgrid, nu , cuda = False):
    #burgers eq: u_t + u* nabla(u) - nu* nabla(nabla(u)) = 0
    partialTime = partialTimeDerivative(u, xgrid, tgrid)
    udiv = divergenceOnlySpace(u, xgrid, tgrid)
    uGradOfDiv = gradientOfDivergenceOnlySpace(u, xgrid, tgrid)
    uout = u(xgrid, tgrid)
    return torch.mean(torch.norm(partialTime + uout * udiv - nu* uGradOfDiv, dim = 1))

