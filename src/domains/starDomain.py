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
        randomSpherical[:,-1] = randomSpherical[:,-1]-0.5
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
        randomSpherical[:,-1] = randomSpherical[:,-1]-0.5
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
        alpha = torch.arctan(self.sideLengths[-1]/ self.sideLengths[-2] ) 
        minPitoPiAngle = angles[:, -1]

        minPitoPiAngle = torch.abs(minPitoPiAngle)

        maskBiggerPiHalf = minPitoPiAngle > torch.pi/2

        minPitoPiAngle = (maskBiggerPiHalf ==0) *minPitoPiAngle + torch.pi *maskBiggerPiHalf - minPitoPiAngle * maskBiggerPiHalf

        biggerAlphaMask = minPitoPiAngle > alpha
        smallerAlphaMask = biggerAlphaMask == 0
 
        r_n = smallerAlphaMask * (self.sideLengths[-2]*0.5 / torch.cos(smallerAlphaMask * minPitoPiAngle )) + biggerAlphaMask*(self.sideLengths[-1]*0.5 / torch.cos(biggerAlphaMask * (torch.pi/2 - minPitoPiAngle) ))
        


        for i in range(self.dim - 2):
            newAngle = angles[:,self.dim-3-i]
            newSide = self.sideLengths[-(3+i)]
            newRad = torch.zeros_like(r_n)
            auxAngles = torch.arctan(2* r_n /newSide)

            biggerThanPiHalfMask = newAngle > torch.pi/2
            newAngle = torch.pi * biggerThanPiHalfMask - newAngle*biggerThanPiHalfMask + newAngle*(biggerThanPiHalfMask == 0)

            
            smallerAuxAngleMask = newAngle <= auxAngles
            biggerAuxAngleMask = smallerAuxAngleMask == 0

            newRad = smallerAuxAngleMask*(newSide *0.5/torch.cos(smallerAuxAngleMask*newAngle)) + biggerAuxAngleMask *(r_n/torch.cos(biggerAuxAngleMask *(torch.pi/2 - newAngle)))

            r_n = newRad

        return r_n

        


