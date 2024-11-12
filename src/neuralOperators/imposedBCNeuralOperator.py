import torch
from torch import nn
from domains.starDomain import *
from radialFunctions.radialFunctions import *



class ImposedBCDeepONetSphere2D(nn.Module):
  def __init__(self,  n_hidden_trunk, n_layers_trunk, n_hidden_branch, n_layers_branch, domain, nBoundaryPoints):
    super().__init__()

    #imposed BC stuff
    self.domain = domain
    self.radialDecayFunciton = squaredRadial
    self.nBoundaryPoints = nBoundaryPoints
    self.boundaryPoints = domain.generateSphericalRandomPointsOnBoundary(nBoundaryPoints)
    nmax = max(n_hidden_branch, n_hidden_trunk)

    #specify TrunkNet
    self.in_layer_trunk = nn.Sequential(*[nn.Linear(2,n_hidden_trunk)])
    self.hid_layers_trunk = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden_trunk, n_hidden_trunk),
            nn.Tanh()
            ])
        for i in range(n_layers_trunk)], nn.Linear(n_hidden_trunk, n_hidden_trunk))
    self.out_layer_trunk = nn.Linear(n_hidden_trunk,nmax)

    #specify BranchNet
    self.in_layer_branch = nn.Sequential(*[nn.Linear(nBoundaryPoints ,n_hidden_branch)])
    self.hid_layers_branch = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden_branch, n_hidden_branch),
            nn.Tanh()
            ])
        for i in range(n_layers_branch)], nn.Linear(n_hidden_branch, n_hidden_branch))
    self.out_layer_branch = nn.Linear(n_hidden_branch,nmax)

    
    
  def updateDevice(self, device):
    self.to(device)
    self.boundaryPoints[0].to(device)
    self.boundaryPoints[1].to(device)
    self.domain.updateDevice(device)


  def boundaryConditionSpherical(self, angles, boundaryFunction):
    return boundaryFunction(self.domain.getCartesianCoordinates( self.domain.radiusDomainFunciton(angles) ,angles))
  
  def zeroOnBoundaryExtension(self, input):
    radius, angles = self.domain.getSphericalCoordinates(input)
    return self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles)).view(-1,1)
  
  def DCBoundaryExtension(self, input, boundaryFunction):
    radius, angles = self.domain.getSphericalCoordinates(input)
    return self.boundaryConditionSpherical( angles, boundaryFunction) *  (1- self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles))).view(-1,1) 
  
  def forward(self, domainPoints, boundaryFunction):
    trunk = torch.cat(domainPoints, dim= 1)
    trunk = self.in_layer_trunk(trunk)
    trunk = self.hid_layers_trunk(trunk)
    trunk = self.out_layer_trunk(trunk)


    boundaryValue = boundaryFunction(self.boundaryPoints).view(1, self.nBoundaryPoints)

    batchSize = trunk.shape[0]

    branch = self.in_layer_branch(boundaryValue)
    branch = self.hid_layers_branch(branch)
    branch = self.out_layer_branch(branch)
    branchTiledFeatures = torch.tile(branch, (batchSize,1))
    outDeepONet = torch.sum(trunk*branchTiledFeatures, dim = 1).view(-1,1)


    return outDeepONet*self.zeroOnBoundaryExtension(domainPoints) + self.DCBoundaryExtension(domainPoints, boundaryFunction)