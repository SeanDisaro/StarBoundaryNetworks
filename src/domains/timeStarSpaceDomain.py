import typing
import torch
from abc import ABC, abstractmethod
import numpy as np
from domains.starDomain import *

class timeStarSpaceDomain:
    def __init__(self, spaceDom:StarDomain, endTime = torch.tensor(1.), device = "cpu")->object:
        self.spaceDom = spaceDom
        self.device = device
        self.endTime= endTime


    def updateDevice(self, device:str):
        '''
        This function changes the device of the domain, i.e. points are generated in ram or vram.

        args:
            device: the new device you want to have
        '''
        self.spaceDom.updateDevice(device)
        self.endTime.to(device)
        self.device = device


    def radiusDomainFunciton(self, timeAngles:torch.tensor,spaceAngles:torch.tensor)->torch.tensor:
        spaceRadius = self.spaceDom.radiusDomainFunciton(spaceAngles)
        maxTimeAngles = torch.arctan(spaceRadius/self.endTime)
        smallerEqMaxTimeAngles =  timeAngles <= maxTimeAngles
        out = smallerEqMaxTimeAngles * self.endTime / torch.cos(timeAngles) + (smallerEqMaxTimeAngles == 0) * spaceRadius / torch.sin(timeAngles)
        return out


    def getCartesianCoordinates(self, radius:torch.tensor, timeAngles:torch.tensor, spaceAngles:torch.tensor) ->typing.Union[torch.tensor, torch.tensor]:
        spaceRadius = radius*torch.sin(timeAngles)
        times = self.endTime - radius*torch.cos(timeAngles)
        spacePoints = self.spaceDom.getCartesianCoordinates(spaceRadius, spaceAngles)
        return spacePoints, times

    def getSphericalCoordinates(self, pointsSpace, times ) ->typing.Union[torch.tensor, torch.tensor, torch.tensor]:
        timeLat = self.endTime - times
        radiusSpace, spaceAngles = self.spaceDom.getSphericalCoordinates(pointsSpace)
        timeLatisZeroMask = timeLat == 0
        timeAngles = (timeLatisZeroMask== 0) *torch.arctan(radiusSpace / (timeLat + timeLatisZeroMask)) + timeLatisZeroMask * torch.pi/2
        totalRadius = torch.sqrt(timeLat **2 + radiusSpace **2)
        return  totalRadius, spaceAngles, timeAngles

    def generateCartesianRandomPointsFullDomain(self, numPoints:int)->typing.Union[torch.tensor, torch.tensor]:
        space = self.spaceDom.generateCartesianRandomPointsFullDomain(numPoints)
        numSpace = space[0].shape[0]
        time = torch.rand((numSpace,1),device= self.device )* self.endTime
        return space, time


    def generateSphericalRandomPointsFullDomain(self, numPoints:int)->typing.Union[torch.tensor, torch.tensor]:
        return self.spaceDom.generateSphericalRandomPointsFullDomain(numPoints), torch.rand((numPoints,1),device= self.device )* self.endTime
    
    def generateParabolicBoundarySphericalFromCartesian(self, numPoints:int)->typing.Union[torch.tensor, torch.tensor]:
        domainPonits, times = self.generateCartesianRandomPointsFullDomain(numPoints)
        _, spaceAngles, timeAngles = self.getSphericalCoordinates( domainPonits, times )
        maxRadius = self.radiusDomainFunciton(timeAngles, spaceAngles)
        outSpace, outTime = self.getCartesianCoordinates(maxRadius, timeAngles, spaceAngles)
        maskNotTimeEnd = outTime != self.endTime
        outTime = outTime*maskNotTimeEnd
        return outSpace, outTime

