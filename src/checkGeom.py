"""This module can be used as a geometry checker before running the analysis"""

from time import time
from numpy import ones, copy
from scipy.linalg import norm
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse.linalg import aslinearoperator
from os import path as pth
import sys
import cPickle

import ConFEM2D_Basics
reload(ConFEM2D_Basics)
from ConFEM2D_Basics import *

import lib_Mat
reload(lib_Mat)
from lib_Mat import *

import lib_Elem
reload(lib_Elem)
from lib_Elem import *

import ConFEM2D_Steps
reload(ConFEM2D_Steps)
from ConFEM2D_Steps import *

import ConFEM2D_InOut
reload(ConFEM2D_InOut)
from ConFEM2D_InOut import *

def PlotGeom(NodeList, ElList):
    """
    :param NodeList:
    :param ElList:
    :return: plot undeformed mesh
    """

    colorsmap = ["blue", "green", "black", "red", "yellow"]
    d2, d3 = False, False
    for i in xrange(len(ElList)):
        Elem = ElList[i]
        if Elem.Type not in ['']: d2 = True  # ['SB3','CPE4','T1D2','S1D2','CPE3','B23E']: d2 = True
        if Elem.Type in ['SH4', 'SB3']: d3 = True
    if d2:
        LX, LY, LZ, LU = [], [], [], []
        for i in xrange(len(NodeList)):
            LX.append(NodeList[i].XCo)
            LY.append(NodeList[i].YCo)
            LZ.append(NodeList[i].ZCo)
        Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = min(LX), max(LX), min(LY), max(LY), min(LZ), max(LZ)
        D = max(Xmax - Xmin, Ymax - Ymin, Zmax - Zmin)
        figure()
        ElSets = {}
        colI = -1  # color index
        for i in xrange(len(ElList)):
            xS, yS, xP, yP, xN, yN = [], [], [], [], [], []
            Elem = ElList[i]
            Set = Elem.Set
            if Set not in ElSets:
                ElSets[Set] = colI
                colI = colI + 1
            if Elem.Type == 'SB3':
                #            d3=True
                xN = [NodeList[Elem.Inzi[0]].XCo, NodeList[Elem.Inzi[1]].XCo, NodeList[Elem.Inzi[2]].XCo,
                      NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo, NodeList[Elem.Inzi[1]].YCo, NodeList[Elem.Inzi[2]].YCo,
                      NodeList[Elem.Inzi[0]].YCo]
            elif Elem.Type == 'CPE4' or Elem.Type == 'CPS4':
                xN = [NodeList[Elem.Inzi[0]].XCo, NodeList[Elem.Inzi[1]].XCo, NodeList[Elem.Inzi[2]].XCo,
                      NodeList[Elem.Inzi[3]].XCo, NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo, NodeList[Elem.Inzi[1]].YCo, NodeList[Elem.Inzi[2]].YCo,
                      NodeList[Elem.Inzi[3]].YCo, NodeList[Elem.Inzi[0]].YCo]

            elif Elem.Type == 'CPE3' or Elem.Type == 'CPS3':
                xN = [NodeList[Elem.Inzi[0]].XCo, NodeList[Elem.Inzi[1]].XCo, NodeList[Elem.Inzi[2]].XCo,
                      NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo, NodeList[Elem.Inzi[1]].YCo, NodeList[Elem.Inzi[2]].YCo,
                      NodeList[Elem.Inzi[0]].YCo]

            elif Elem.Type == 'SH4':
                xN = [NodeList[Elem.Inzi[0]].XCo, NodeList[Elem.Inzi[1]].XCo, NodeList[Elem.Inzi[2]].XCo,
                      NodeList[Elem.Inzi[3]].XCo, NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo, NodeList[Elem.Inzi[1]].YCo, NodeList[Elem.Inzi[2]].YCo,
                      NodeList[Elem.Inzi[3]].YCo, NodeList[Elem.Inzi[0]].YCo]

            elif Elem.Type == 'SH3':
                xN = [NodeList[Elem.Inzi[0]].XCo, NodeList[Elem.Inzi[1]].XCo, NodeList[Elem.Inzi[2]].XCo,
                      NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo, NodeList[Elem.Inzi[1]].YCo, NodeList[Elem.Inzi[2]].YCo,
                      NodeList[Elem.Inzi[0]].YCo]

            elif Elem.Type == 'T1D2':
                xN = [NodeList[Elem.Inzi[0]].XCo, NodeList[Elem.Inzi[1]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo, NodeList[Elem.Inzi[1]].YCo]

            elif Elem.Type == 'S1D2':
                j = 0
                k0 = Elem.DofI[j, 0]
                k1 = Elem.DofI[j + 1, 0]

            elif Elem.Type == 'B23E':
                xN += [NodeList[Elem.Inzi[0]].XCo, NodeList[Elem.Inzi[2]].XCo]
                yN += [NodeList[Elem.Inzi[0]].YCo, NodeList[Elem.Inzi[2]].YCo]

            elif Elem.Type not in ['SB3', 'CPE4', 'CPS4', 'SH4', 'SH3', 'T1D2', 'S1D2', 'CPE3', 'CPS3', 'B23E']:
                for j in xrange(Elem.nNod):
                    k0 = Elem.DofI[j, 0]
                    k1 = Elem.DofI[j, 1]
                    xN += [NodeList[Elem.Inzi[j]].XCo]  # undeformed 2D geometry
                    yN += [NodeList[Elem.Inzi[j]].YCo]  # undeformed 2D geometry

            plot(xN, yN, colorsmap[colI])
        #grid()
        show()

def SortElemSets(ElList, targSet=None):
    """
    :param ElList: all data
    :param targSet: format=list, desired element set
    :return: all elements that have the targSet
    """
    ElSets = {}
    for Elem in ElList:
        Set = Elem.Set
        if not Set in ElSets: ElSets[Set] = list()
        ElSets[Set] += [Elem]
    if targSet == None: return ElList
    else:
        ElList_ = []
        for set in targSet:
            ElList_ += ElSets[set]
        return ElList_


if __name__ == "__main__":
    Name = "C:/Users/regga/Desktop/testcode/E6-03"
    f1 = open(Name+".in.txt", 'r')
    NodeList, ElList, MatList, StepList = ReadInputFile(f1, False)
    f1.close()
    N, Mask, Skyline, SDiag, SLen = AssignGlobalDof(NodeList, ElList,
                                                    MatList)  # assign degrees of freedom (dof) to nodes and elements -> see above
    ElList_ = SortElemSets(ElList, targSet=["EL1","SIREBAR","ELA"])
    PlotGeom(NodeList, ElList_)
