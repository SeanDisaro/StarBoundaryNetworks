import torch
from src.pdeOperators.operators import *



def burgersEquationLoss(u, xgrid, tgrid, nu , cuda = False):
    #burgers eq: u_t + u* nabla(u) - nu* nabla(nabla(u)) = 0
    uout = u(xgrid, tgrid)
    partialTime = partialDerivative(uout, tgrid)
    udiv = divergence(uout, xgrid)
    uGradOfDiv = gradient(udiv, xgrid)

    return torch.mean(torch.norm(partialTime + uout * udiv - nu* uGradOfDiv, dim = 1))



def poissonEquationLoss(u, xgrid, poissonData, cuda = False):
    #poisson eq: laplace(u)+ poissonDataFunction = 0
    laplacian_u = laplacian(u, xgrid)
    dataOut = poissonData(xgrid)
    return torch.mean(torch.norm(laplacian_u + dataOut, dim = 1))

def darcyFlow(u, xgrid, diffCoefFunc, forcingFunc, cuda = False ):
    #darcyFlow: forcingFunc + divergence((diffCoef*grad(u)) = 0
    uOut = u(xgrid)
    diffCoefFuncOut = diffCoefFunc(xgrid)
    forcingFuncOut = forcingFunc(xgrid)
    uGrad = gradient(uOut, xgrid)
    divOfCoeffGrad = divergence(uGrad*diffCoefFuncOut)

    return torch.mean(torch.norm(forcingFuncOut + divOfCoeffGrad, dim = 1))


