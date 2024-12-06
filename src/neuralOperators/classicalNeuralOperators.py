import torch
from torch import nn


class DeepONet2D_DBCtoSol(nn.Module):
  def __init__(self,  n_hidden_trunk, n_layers_trunk, n_hidden_branch, n_layers_branch, boundaryPoints):
    super().__init__()

    self.boundaryPoints = boundaryPoints
    self.nBoundaryPoints = boundaryPoints[0].shape[0]
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
    self.in_layer_branch = nn.Sequential(*[nn.Linear(self.nBoundaryPoints ,n_hidden_branch)])
    self.hid_layers_branch = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden_branch, n_hidden_branch),
            nn.Tanh()
            ])
        for i in range(n_layers_branch)], nn.Linear(n_hidden_branch, n_hidden_branch))
    self.out_layer_branch = nn.Linear(n_hidden_branch,nmax)

    
    
  def updateDevice(self, device):
    self.to(device)
    self.boundaryPoints[0] = self.boundaryPoints[0].to(device)
    self.boundaryPoints[1] = self.boundaryPoints[1].to(device)


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

    return outDeepONet



class DeepONetSphere2D_DomainFuncToSol(nn.Module):
  def __init__(self,  n_hidden_trunk, n_layers_trunk, n_hidden_branch, n_layers_branch, fixDomPoints):
    super().__init__()


    self.fixDomainPoints = fixDomPoints
    self.nFixDomPoints = fixDomPoints[0].shape[0]
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
    self.in_layer_branch = nn.Sequential(*[nn.Linear(self.nFixDomPoints ,n_hidden_branch)])
    self.hid_layers_branch = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden_branch, n_hidden_branch),
            nn.Tanh()
            ])
        for i in range(n_layers_branch)], nn.Linear(n_hidden_branch, n_hidden_branch))
    self.out_layer_branch = nn.Linear(n_hidden_branch,nmax)

    
    
  def updateDevice(self, device):
    self.to(device)
    self.fixDomainPoints[0] = self.fixDomainPoints[0].to(device)
    self.fixDomainPoints[1] = self.fixDomainPoints[1].to(device)


  
  def forward(self, domainPoints, domainFunc):
    trunk = torch.cat(domainPoints, dim= 1)
    trunk = self.in_layer_trunk(trunk)
    trunk = self.hid_layers_trunk(trunk)
    trunk = self.out_layer_trunk(trunk)

    batchSize = trunk.shape[0]

    branch = domainFunc(self.fixDomainPoints).view(1, self.nFixDomPoints)
    branch = self.in_layer_branch(branch)
    branch = self.hid_layers_branch(branch)
    branch = self.out_layer_branch(branch)
    branchTiledFeatures = torch.tile(branch, (batchSize,1))
    outDeepONet = torch.sum(trunk*branchTiledFeatures, dim = 1).view(-1,1)


    return outDeepONet
  



class DeepONetSphere2D_ParabDBDCondToSol(nn.Module):
  def __init__(self,  n_hidden_trunk, n_layers_trunk, n_hidden_branch, n_layers_branch, fixDomPoints, boundaryPoints, timeGrid):
    super().__init__()

    self.fixDomainPoints = fixDomPoints
    self.nFixDomPoints = fixDomPoints[0].shape[0]
    
    self.boundaryPoints = boundaryPoints
    self.nboundaryPoints = boundaryPoints[0].shape[0]

    self.timeGrid = timeGrid
    self.nTimeGrid = timeGrid.shape[0]

    nmax = max(n_hidden_branch, n_hidden_trunk)

    #specify TrunkNet
    self.in_layer_trunk = nn.Sequential(*[nn.Linear(3,n_hidden_trunk)])
    self.hid_layers_trunk = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden_trunk, n_hidden_trunk),
            nn.Tanh()
            ])
        for i in range(n_layers_trunk)], nn.Linear(n_hidden_trunk, n_hidden_trunk))
    self.out_layer_trunk = nn.Linear(n_hidden_trunk,nmax)

    #specify BranchNet
    self.in_layer_branch = nn.Sequential(*[nn.Linear(self.nFixDomPoints + self.nboundaryPoints *self.nTimeGrid  ,n_hidden_branch)])
    self.hid_layers_branch = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden_branch, n_hidden_branch),
            nn.Tanh()
            ])
        for i in range(n_layers_branch)], nn.Linear(n_hidden_branch, n_hidden_branch))
    self.out_layer_branch = nn.Linear(n_hidden_branch,nmax)

    
    
  def updateDevice(self, device):
    self.to(device)
    self.fixDomainPoints[0] = self.fixDomainPoints[0].to(device)
    self.fixDomainPoints[1] = self.fixDomainPoints[1].to(device)
    self.timeGrid = self.timeGrid.to(device)
    self.boundaryPoints[0] = self.boundaryPoints[0].to(device)
    self.boundaryPoints[1] = self.boundaryPoints[1].to(device)

  
  def forward(self, domainPoints,timePoints,  funcOnDomain, funcOnBoundarytimesTime):
    trunk = torch.cat((domainPoints[0],domainPoints[1], timePoints), dim= 1)
    trunk = self.in_layer_trunk(trunk)
    trunk = self.hid_layers_trunk(trunk)
    trunk = self.out_layer_trunk(trunk)

    batchSize = trunk.shape[0]
    outFuncOnBoundarytimesTime
    branch = torch.cat((funcOnDomain(self.fixDomainPoints),funcOnBoundarytimesTime(self.boundaryPoints, self.timeGrid))).view(1,-1)
    branch = self.in_layer_branch(branch)
    branch = self.hid_layers_branch(branch)
    branch = self.out_layer_branch(branch)
    branchTiledFeatures = torch.tile(branch, (batchSize,1))
    outDeepONet = torch.sum(trunk*branchTiledFeatures, dim = 1).view(-1,1)


    return outDeepONet