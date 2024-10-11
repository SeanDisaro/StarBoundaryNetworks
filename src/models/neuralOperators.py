import torch
from torch import nn
from src.domains.starDomain import *
from src.radialFunctions.radialFunctions import *



class ImposedStarDeepONet(nn.Module):
  def __init__(self,  n_hidden_trunk, n_layers_trunk, n_hidden_branch, n_layers_branch, domain, nBoundaryPoints, nRandBoundaryPoints):
    super().__init__()

    #imposed BC stuff
    self.domain = domain
    self.radialDecayFunciton = linearRadial
    self.boundaryPoints = domain.generateSphericalRandomPointsOnBoundary(nBoundaryPoints)
    self.nRandBoundaryPoints = nRandBoundaryPoints
    nmax = max(n_hidden_branch, n_hidden_trunk)
    self.dim = domain.dim


    #specify TrunkNet
    self.in_layer_trunk = nn.Sequential(*[nn.Linear(self.dim,n_hidden_trunk)])
    self.hid_layers_trunk = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden_trunk, n_hidden_trunk),
            nn.Tanh()
            ])
        for i in range(n_layers_trunk)], nn.Linear(n_hidden_trunk, n_hidden_trunk))
    self.out_layer_trunk = nn.Linear(n_hidden_trunk,nmax)

    #specify BranchNet
    self.in_layer_branch = nn.Sequential(*[nn.Linear(self.dim + 1 ,n_hidden_branch)])
    self.hid_layers_branch = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden_branch, n_hidden_branch),
            nn.Tanh()
            ])
        for i in range(n_layers_branch)], nn.Linear(n_hidden_branch, n_hidden_branch))
    self.out_layer_branch = nn.Linear(n_hidden_branch,nmax)

    
    

  def boundaryCondition(self, input, boundaryFunction):
    return boundaryFunction(input)
  
  def boundaryConditionSpherical(self, angles,boundaryFunction):
    return self.boundaryCondition(self.domain.getCartesianCoordinates( self.domain.radiusDomainFunciton(angles) ,angles), boundaryFunction)
  
  def zeroOnBoundaryExtension(self, input):
    radius, angles = self.domain.getSphericalCoordinates(input)
    return self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles)).view(-1,1)
  
  def DCBoundaryExtension(self, input, boundaryFunction):
    radius, angles = self.domain.getSphericalCoordinates(input)
    return (self.boundaryConditionSpherical( angles) *  (1- self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles))), boundaryFunction).view(-1,1)
  
  def forward(self, domainPoints, boundaryFunction):
    numPoints = domainPoints.shape[0]
    trunk = domainPoints
   
    trunk = self.in_layer_trunk(trunk)
   
    trunk = self.hid_layers_trunk(trunk)
   
    trunk = self.out_layer_trunk(trunk)
    
    randPoints = self.domain.generateSphericalRandomPointsOnBoundary(self.nRandBoundaryPoints)
    allPoints = torch.cat((self.boundaryPoints, randPoints), 0)

    boundaryValues = boundaryFunction(allPoints)
    branch = torch.cat((allPoints , boundaryValues),1)
    branch = self.in_layer_branch(branch)
    branch = self.hid_layers_branch(branch)
    branch = self.out_layer_branch(branch)

    branchFeatures = torch.sum(branch, 0).tile((numPoints,1))
    x = torch.tensordot(trunk, branchFeatures, 1)

    return x*self.zeroOnBoundaryExtension(input) + self.DCBoundaryExtension(input, boundaryFunction)