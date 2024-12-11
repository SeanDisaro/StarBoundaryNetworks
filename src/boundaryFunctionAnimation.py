import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
import os
from pathlib import PurePath
from neuralOperators.imposedBCNeuralOperator import *
from domains.starDomain import *
from domains.timeStarSpaceDomain import *

def testFunc1(x,t, freq):
    out = torch.exp(-2*freq*freq*t)* torch.sin(freq* x[0])* torch.sin(freq* x[1])
    return out.view(-1,1)

def testFunc2(x,t, freq):
    out = torch.exp(-2*freq*freq*t)* torch.sin(freq* x[0])* torch.cos(freq* x[1])
    return out.view(-1,1)

def testFunc3(x,t, freq):
    out = torch.exp(-2*freq*freq*t)* torch.cos(freq* x[0])* torch.sin(freq* x[1])
    return out.view(-1,1)

def testFunc4(x,t, freq):
    out = torch.exp(-2*freq*freq*t)* torch.cos(freq* x[0])* torch.cos(freq* x[1])
    return out.view(-1,1)

def trainFunc(x,t, a,b,c,d, freq1,freq2,freq3,freq4):
    return a*testFunc1(x,t, freq1/10) + b*testFunc2(x,t, freq2/10) + c*testFunc3(x,t, freq3/10)+ d*testFunc4(x,t, freq4/10)

aRand,bRand, cRand, dRand =  torch.rand(4)*4 - 2
lam1Rand, lam2Rand, lam3Rand, lam4Rand = (1,1,1,1)

myTestFunc = lambda x,t : trainFunc(x,t, aRand,bRand, cRand, dRand, lam1Rand, lam2Rand, lam3Rand, lam4Rand)



#frames
frames = 1000
#domain of solution
domainSpaceOnly  = Sphere(2,torch.tensor([0.,0.]), torch.tensor(1.),"cpu")
spaceTimeDomain = timeStarSpaceDomain(domainSpaceOnly, endTime = torch.tensor(10.), device= "cpu")
pointsSpace = domainSpaceOnly.generateCartesianRandomPointsFullDomain(1000)
numPoints = pointsSpace[0].shape[0]
timePoints = torch.linspace(0,10, frames)



solOperatorImposed = ImposedBCDeepONetSphere2D_ParabDBDCondToSol(10, 10, 10, 10, spaceTimeDomain, 10)
solOperatorImposed.updateDevice("cpu")

zeroOnBoundaryFunc = solOperatorImposed.zeroOnBoundaryExtension
boundaryFunc = solOperatorImposed.DCBoundaryExtension




spaceXNP = pointsSpace[0].view(-1).detach().numpy()
spaceYNP = pointsSpace[1].view(-1).detach().numpy()

fig = plt.figure(figsize=(9, 5))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


ax1.set_title("zero on bd func")
ax2.set_title("boundary func condition")

# Create a meshgrid to interpolate over (choose grid resolution as needed)
grid_x, grid_y = np.mgrid[min(spaceXNP):max(spaceXNP):100j, 
                          min(spaceYNP):max(spaceYNP):100j]


zeroFuncOut = zeroOnBoundaryFunc(pointsSpace, torch.full((numPoints,1), 10)).view(-1).detach().numpy()
differenceBoundaryFuncInputFuncOut = (boundaryFunc(pointsSpace, torch.full((numPoints,1), 10.), myTestFunc) - myTestFunc(pointsSpace, torch.full((numPoints,1), 10.))).view(-1).detach().numpy()

#print(f"max zero func out = {max(zeroFuncOut)}")
# Interpolate the z values over the grid
grid_z_zero = griddata((spaceXNP, spaceYNP), zeroFuncOut, (grid_x, grid_y), method='cubic')
grid_z_boundFunc = griddata((spaceXNP, spaceYNP), differenceBoundaryFuncInputFuncOut, (grid_x, grid_y), method='cubic')


#pcm = axis.pcolormesh(uout, cmap=plt.cm.jet, vmin=0, vmax=100)
pcm1 = ax1.pcolormesh(grid_z_zero , cmap='coolwarm', shading='auto')
plt.colorbar(pcm1, ax=ax1)
pcm2 = ax2.pcolormesh(grid_z_boundFunc , cmap='coolwarm', shading='auto')
plt.colorbar(pcm2, ax=ax2)

# Simulating

counter = 0

while counter < frames :
    zeroFuncOut = zeroOnBoundaryFunc(pointsSpace, torch.full((numPoints,1), timePoints[counter])).view(-1).detach().numpy()
    differenceBoundaryFuncInputFuncOut = (boundaryFunc(pointsSpace, torch.full((numPoints,1), timePoints[counter]), myTestFunc) - myTestFunc(pointsSpace, torch.full((numPoints,1), timePoints[counter]))).view(-1).detach().numpy()

    grid_z_zero = griddata((spaceXNP, spaceYNP), zeroFuncOut, (grid_x, grid_y), method='cubic')
    grid_z_boundFunc = griddata((spaceXNP, spaceYNP), differenceBoundaryFuncInputFuncOut, (grid_x, grid_y), method='cubic')

    # Updating the plot
    pcm1.set_array( grid_z_zero)
    pcm2.set_array( grid_z_boundFunc)
    ax1.set_title(f"zero on bd func {counter}")
    ax2.set_title(f"boundary func condition {counter}")
    plt.tight_layout()
    plt.pause(0.1)

    counter += 1
plt.show()