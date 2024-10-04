import torch
from torch.autograd.functional import hessian
from torch import nn
from tqdm import tqdm

from domains.starDomain import Sphere
from radialFunctions.radialFunctions import linearRadial
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

def laplacian(f, input):
    dim = input.shape[1]
    batchSize = input.shape[0]
    outf = f(input)
    laplacian = torch.zeros(batchSize, dtype=float, requires_grad=True)

    dx = torch.autograd.grad(outf, input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0]

    for i in range(dim):
        d2x = torch.autograd.grad((dx[:,i]).view(-1,1), input, torch.ones((batchSize, 1), requires_grad=True), create_graph=True)[0][:,i]
        laplacian = laplacian + d2x
        
    return laplacian.view(-1,1)
    
#define PINN loss function for problem from above:
def pinnLossPoissonSin(u, xy_grid):#, device = 'cuda'):

  laplacian_u = laplacian(u, xy_grid)


  return torch.mean(laplacian_u - torch.sin(xy_grid[:,0] * xy_grid[:, 1]))



class ImposedBCPINNSphere2D(nn.Module):
  def __init__(self, n_hidden, n_layers):
    super().__init__()
    self.activation_func = nn.Tanh
    self.in_layer = nn.Sequential(*[nn.Linear(2,n_hidden) , self.activation_func()])
    self.hid_layers = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh()
            ])
        for i in range(n_layers)], nn.Linear(n_hidden, n_hidden))
    self.out_layer = nn.Linear(n_hidden,1)
    self.domain = Sphere(2, torch.tensor((0.,0.)), torch.tensor(1.))
    self.radialDecayFunciton = linearRadial
  

  def boundaryCondition(self, input):
    return torch.sin(input[:, 0]*input[:, 1])
  
  def boundaryConditionSpherical(self, angles):
    return self.boundaryCondition(self.domain.getCartesianCoordinates( self.domain.radiusDomainFunciton(angles) ,angles))
  
  def zeroOnBoundaryExtension(self, input):
    radius, angles = self.domain.getSphericalCoordinates(input)
    #print(radius.shape)
    #print(angles.shape)
    #print((self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles))).shape)
    return self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles)).view(-1,1)
  
  def DCBoundaryExtension(self, input):
    radius, angles = self.domain.getSphericalCoordinates(input)
    #print(radius.shape)
    #print(angles.shape)
    #print((self.boundaryConditionSpherical( angles) *  (1- self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles)))).shape)
    return (self.boundaryConditionSpherical( angles) *  (1- self.radialDecayFunciton( radius / self.domain.radiusDomainFunciton(angles)))).view(-1,1)
  
  def forward(self, input):
    x = input + 0
    x = self.in_layer(x)
    x = self.hid_layers(x)
    x = self.out_layer(x)

    return x*self.zeroOnBoundaryExtension(input) + self.DCBoundaryExtension(input)
  


sphereForPoints  = Sphere(2,torch.tensor([0.0,0.0]), 1.)
spherePoints = sphereForPoints.generateRandomPointsFullDomain(10000)



#define Trainingset on A:
#num_train_boundary = 16
xy_grid = spherePoints
xy_grid.requires_grad = True



#model:
solPDE = ImposedBCPINNSphere2D( 16, 4)

optimizer = torch.optim.Adam(solPDE.parameters(), lr = 1e-3)

print(solPDE(xy_grid))

for i in tqdm(range(2000)):
  optimizer.zero_grad()
 
  loss = pinnLossPoissonSin(solPDE, xy_grid )

  print(loss.item())
  loss.backward(retain_graph=True)
  optimizer.step()
