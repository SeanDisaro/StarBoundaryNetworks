import torch
from torch import nn
from src.domains.starDomain import *
from src.radialFunctions.radialFunctions import *



class ImposedBCPINNSphere2D(nn.Module):
  def __init__(self,  n_hidden_trunk, n_layers_trunk, n_hidden_branch, n_layers_branch, domain, nBoundaryPoints, nViewBoundaryPoints):
    super().__init__()

    #imposed BC stuff
    self.domain = domain
    self.radialDecayFunciton = linearRadial
    self.boundaryPoints = domain.generateSphericalRandomPointsOnBoundary(nBoundaryPoints)
    self.nViewBoundaryPoints = nViewBoundaryPoints
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

    
    

  def boundaryCondition(self, input):
    return self.boundaryFunciton(input)
  
  def boundaryConditionSpherical(self, angles):
    return self.boundaryCondition(self.domain.getCartesianCoordinates( self.domain.radiusDomainFunciton(angles) ,angles))
  
  def zeroOnBoundaryExtension(self, input):
    radius, angles = self.domain.getSphericalCoordinates(input)
    return self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles)).view(-1,1)
  
  def DCBoundaryExtension(self, input):
    radius, angles = self.domain.getSphericalCoordinates(input)
    return (self.boundaryConditionSpherical( angles) *  (1- self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles)))).view(-1,1)
  
  def forward(self, domainPoints, boundaryFunction):
    trunk = domainPoints
   
    trunk = self.in_layer_trunk(trunk)
   
    trunk = self.hid_layers_trunk(trunk)
   
    trunk = self.out_layer_trunk(trunk)
    
    randPoints = self.domain.generateSphericalRandomPointsOnBoundary(self.nViewBoundaryPoints)
    allPoints = torch.cat((self.boundaryPoints, randPoints), 0)

    boundaryValue = boundaryFunction(allPoints)


    branch = self.in_layer_branch()

    return x*self.zeroOnBoundaryExtension(input) + self.DCBoundaryExtension(input)