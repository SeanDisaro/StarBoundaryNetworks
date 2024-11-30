import torch
from pdeOperators.operators import *


def laplaceEquationLossOperator(u, xgrid, boundaryDataFunction, boundaryPoints = None, alpha = 1, beta = 0.01 , device= "cpu"):
    uOut = u(xgrid, boundaryDataFunction).to(device)
    laplacian_u = laplacian(uOut, xgrid, device= device)
    if boundaryPoints is None:
        #return loss without bd condition
        return torch.mean(torch.norm(laplacian_u , dim = 1))
    else:
        #return loss with boundary condition
        boundaryEvalFunc = boundaryDataFunction(boundaryPoints)
        boundaryEvalu = u(boundaryPoints, boundaryDataFunction)
        return alpha* torch.mean(torch.norm(laplacian_u, dim = 1)) + beta * torch.mean(torch.norm(boundaryEvalFunc - boundaryEvalu, dim = 1))
    

def poissonEquationLossOperator(u, xgrid, domainFunction, fixBoundaryFunction = None, boundaryPoints = None, alpha = 1, beta = 0.01 , device= "cpu"):
    uOut = u(xgrid, domainFunction).to(device)
    laplacian_u = laplacian(uOut, xgrid, device= device)
    domainFuncOut = domainFunction(xgrid)
    if boundaryPoints is None:
        #return loss without bd condition
        return torch.mean(torch.norm(laplacian_u - domainFuncOut , dim = 1))
    else:
        #return loss with boundary condition
        boundaryEvalFunc = fixBoundaryFunction(boundaryPoints)
        boundaryEvalu = u(boundaryPoints, domainFunction)
        return alpha* torch.mean(torch.norm(laplacian_u - domainFuncOut, dim = 1)) + beta * torch.mean(torch.norm(boundaryEvalFunc - boundaryEvalu, dim = 1))