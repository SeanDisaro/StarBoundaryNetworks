import torch
from torch import nn
from src.domains.starDomain import *
from src.radialFunctions.radialFunctions import *



class ImposedStarDeepONet(nn.Module):
  def __init__(self,  n_hidden, n_layers, boundaryFunciton , domain):
    super().__init__()

    #imposed BC stuff
    self.domain = domain
    self.radialDecayFunciton = linearRadial
    self.dim = domain.dim
    self.boundaryFunciton = boundaryFunciton


    #specify TrunkNet
    self.in_layer = nn.Sequential(*[nn.Linear(self.dim,n_hidden)])
    self.hid_layers = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh()
            ])
        for i in range(n_layers)], nn.Linear(n_hidden, n_hidden))
    self.out_layer = nn.Linear(n_hidden,1)

    
    

  def boundaryCondition(self, input):
    return self.boundaryFunction(input)
  
  def boundaryConditionSpherical(self, angles):
    return self.boundaryCondition(self.domain.getCartesianCoordinates( self.domain.radiusDomainFunciton(angles) ,angles))
  
  def zeroOnBoundaryExtension(self, input):
    radius, angles = self.domain.getSphericalCoordinates(input)
    return self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles)).view(-1,1)
  
  def DCBoundaryExtension(self, input):
    radius, angles = self.domain.getSphericalCoordinates(input)
    return (self.boundaryConditionSpherical( angles) *  (1- self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles)))).view(-1,1)
  
  def forward(self, input):
    x = input
   
    x = self.in_layer(x)
   
    x = self.hid_layers(x)
   
    x = self.out_layer(x)


    return x*self.zeroOnBoundaryExtension(input) + self.DCBoundaryExtension(input)