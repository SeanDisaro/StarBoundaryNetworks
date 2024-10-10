import typing
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import math
import numpy as np

class StarDomain:


    def __init__(self, dim, center):
        self.dim = dim
        self.center = center

    @abstractmethod
    def radiusDomainFunciton(self, angles):
        pass


    def isInDomainCartesian(self, x):
        radius, angles = self.getSphericalCoordinates(x)
        return self.isInDomainSpherical(radius, angles)

    def isInDomainSpherical(self, radius, angles):
        if self.radiusDomainFunciton(angles) <= 0:
            pass
        if radius <= self.radiusDomainFunciton(angles):
            return True
        else:
            return False

    def getSphericalCoordinates(self, x):
        #this function has a bug
        batchSize = x.shape[0]
        for i in range(self.dim):
            mask = torch.zeros_like(x)
            mask[:,i] = 1 
            x = x - mask*self.center[i]

        r = torch.linalg.vector_norm(x, axis = 1)

        angles = torch.empty(batchSize, self.dim - 1)

        auxAngles = torch.pow(r,2) - torch.pow(x[:,0],2)

        for i in range(self.dim-2):
            angles[:,i] = torch.atan2(torch.sqrt(auxAngles), x[:,i])
            auxAngles = auxAngles - torch.pow(x[:,i+1],2)

        angles[:,-1] = torch.atan2(x[:,-1], x[:,-2])


        return r, angles

    def getCartesianCoordinates(self, radius, angles):
        batchSize = radius.shape[0]
        x = torch.ones(batchSize, self.dim)
        
        for i in range(self.dim):
            for j in range(i):
                mask = torch.zeros_like(x)
                mask[: , i] = 1
                x = x * (mask == 0) + x*mask * torch.sin(angles[:,j]).view(-1, 1).tile(self.dim)
                #x[:,i] = x[:,i]* torch.sin(angles[:,j])

            if i == self.dim -1:
                mask = torch.zeros_like(x)
                mask[: , i] = 1
                x = x * (mask == 0) + x*mask * radius.view(-1, 1).tile(self.dim)
                #x[:,i] = x[:,i]*radius
            else:
                mask = torch.zeros_like(x)
                mask[: , i] = 1
                x = x * (mask == 0) + x*mask * radius.view(-1, 1).tile(self.dim) * torch.cos(angles[:,i]).view(-1, 1).tile(self.dim)
                #x[:,i] = x[:,i]*radius * torch.cos(angles[:,i])

        for i in range(self.dim):
            mask = torch.zeros_like(x)
            mask[:,i] = 1 
            x = x - mask*self.center[i]

        return x
    
    def generateSphericalRandomPointsFullDomain(self, numPoints ):
        randomSpherical = torch.rand((numPoints, self.dim))
        randomSpherical[:,-1] = randomSpherical[:,-1] *2*torch.pi
        for i in range(self.dim -2):
            randomSpherical[:, 1+i] = randomSpherical[:, 1+i]* torch.pi
        angles = randomSpherical[:,1:]
        max_radius = self.radiusDomainFunciton(angles)
        radius = randomSpherical[:,0] * max_radius
        return self.getCartesianCoordinates(radius, angles)
    
    def generateCartesianRandomPointsFullDomain(self, numPoints):
        randomUnitQube = torch.rand((numPoints, self.dim))
        randomCoverQube = randomUnitQube - torch.full((numPoints, self.dim), 0.5)
        randomCoverQube = randomCoverQube * 2*self.maxRadius
        randomCoverQube = randomCoverQube + self.center.tile((numPoints, 1))
        maskKeep = torch.zeros((numPoints), dtype= bool)
        for i in range(numPoints):
            if self.isInDomainCartesian(randomCoverQube[i].view(1,-1)):
                maskKeep[i] = True

        return randomCoverQube[maskKeep.nonzero().view(-1)]

    
    def generateSphericalRandomPointsOnBoundary(self, numPoints ):
        randomSpherical = torch.rand((numPoints, self.dim))
        randomSpherical[:,-1] = randomSpherical[:,-1] *2*torch.pi
        for i in range(self.dim -2):
            randomSpherical[:, 1+i] = randomSpherical[:, 1+i]* torch.pi
        angles = randomSpherical[:,1:]
        max_radius = self.radiusDomainFunciton(angles)
        return self.getCartesianCoordinates(max_radius, angles)






class Sphere(StarDomain):


    def __init__(self, dim, center, radius):
        super().__init__( dim, center)
        self.radius = radius
        self.maxRadius = radius


    def radiusDomainFunciton(self, angles):
        return torch.full((angles.shape[0],1), self.radius).view(angles.shape[0])



    
class HyperCuboid(StarDomain):
    def __init__(self, dim, center, sideLengths):
        super().__init__(dim, center)
        self.sideLengths = sideLengths
        self.maxRadius = torch.norm(sideLengths*0.5)


    def radiusDomainFunciton(self, angles):
        alpha = torch.arctan(self.sideLengths[1]/ self.sideLengths[0] ) 
        twoPiAngle = angles[:, -1]

        biggerThanPiMask = twoPiAngle > torch.pi
        twoPiAngle = twoPiAngle - torch.pi * biggerThanPiMask

        biggerThanPiHalfMask = twoPiAngle > torch.pi/2
        twoPiAngle = torch.pi * biggerThanPiHalfMask - twoPiAngle*biggerThanPiHalfMask + twoPiAngle*(biggerThanPiHalfMask ==0)

        smallerAlphaMask = twoPiAngle <= alpha
        r_n = smallerAlphaMask*(self.sideLengths[0]/2) / torch.cos(twoPiAngle * smallerAlphaMask)

        biggerAlphaMask = smallerAlphaMask == 0
        r_n = r_n + biggerAlphaMask*(self.sideLengths[1]/2) / torch.cos((torch.pi/2 - twoPiAngle )* biggerAlphaMask)

        for i in range(self.dim - 2):
            newAngle = angles[:,i]
            newSide = self.sideLengths[2+i]
            newRad = torch.zeros_like(r_n)
            auxAngles = torch.arctan(2* r_n /newSide)

            biggerThanPiHalfMask = newAngle > torch.pi/2
            newAngle = torch.pi * biggerThanPiHalfMask - newAngle*biggerThanPiHalfMask + newAngle*(biggerThanPiHalfMask == 0)

            
            smallerAuxAngleMask = newAngle <= auxAngles
            biggerAuxAngleMask = smallerAuxAngleMask == 0

            newRad = smallerAuxAngleMask*(newSide/2)/torch.cos(newAngle*smallerAuxAngleMask)
            newRad = newRad + biggerAuxAngleMask*(r_n / torch.cos((torch.pi /2 - newAngle)* biggerAuxAngleMask))

            r_n = newRad

        return r_n

        




#testSphere3D = Sphere(3, torch.tensor([0.,0.,0.]),torch.tensor(1.))
#testSphere2D = Sphere(2, torch.tensor([0.,0.]),torch.tensor(1.))

#fig = plt.figure()

#qube2D = HyperCuboid(2, torch.tensor([0.,0.]), torch.tensor([1.,1.]))
#qube3D = HyperCuboid(3, torch.tensor([0.,0.,0.]), torch.tensor([1.,1.,1.]))

#spherePoints3D = testSphere3D.generateCartesianRandomPointsFullDomain(10000) #generateRandomPointsSphere(2,100000, 1., 0. )
#spherePoints2D = testSphere2D.generateRandomPointsFullDomain(10000)
#spherePoints2D = testSphere2D.generateCartesianRandomPointsFullDomain(10000)
#qubePoints2D = qube2D.generateSphericalRandomPointsOnBoundary(1000)
#qubePoints3D, realQube = qube3D.generateCartesianRandomPointsFullDomain(10000)


#qubePointsNP = torch.Tensor.numpy(qubePoints3D)
#ax = fig.add_subplot(projection='3d')
#ax.scatter(qubePointsNP[:,0], qubePointsNP[:,1], qubePointsNP[:,2],c = 'b')
#realQube = np.random.uniform(-0.5,0.5,(10000,3))
#ax.scatter(realQube[:,0], realQube[:,1], realQube[:,2],c = 'r')


#spherePointsNP = torch.Tensor.numpy(spherePoints3D)
#ax = fig.add_subplot(projection='3d')
#ax.plot3D(spherePointsNP[:,0], spherePointsNP[:,1], spherePointsNP[:,2])


#plt.scatter(qubePoints2D[:,0],qubePoints2D[:,1])
#plt.scatter(spherePoints2D[:,0],spherePoints2D[:,1])

#plt.show()