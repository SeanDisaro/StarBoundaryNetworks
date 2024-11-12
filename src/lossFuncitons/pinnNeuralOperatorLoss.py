import torch
from pdeOperators.operators import *


def laplaceEquationLossOperator(u, xgrid, boundaryDataFunction, device= "cpu"):
    uOut = u(xgrid, boundaryDataFunction).to(device)
    laplacian_u = laplacian(uOut, xgrid, device= device)
    return torch.mean(torch.norm(laplacian_u , dim = 1))