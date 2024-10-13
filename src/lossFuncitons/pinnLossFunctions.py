import torch
from src.pdeOperators.operators import *



#define PINN loss function for problem from above:
def burgersEquationLoss(u, xgrid, tgrid, nu , cuda = False):
    partialTime = partialTimeDerivative(u, xgrid, tgrid)
    udiv = divergenceOnlySpace(u, xgrid, tgrid)
    uGradOfDiv = gradientOnlySpace()
    uout = u(xgrid, tgrid)

    return 