# ConPostPlot -- module for result plotting.

from time import *
from numpy import *
#from scipy.linalg import *
#from scipy import sparse
#from scipy.sparse.linalg.dsolve import linsolve
#from scipy.sparse.linalg import aslinearoperator
import matplotlib.pyplot as plt
from os import path as pth
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

class ConPostPlot:
    def __init__(self):
        pass
    def Run(self, Name):
        f1=open( Name+".in.txt", 'r')
        NodeList, ElList, MatList, StepList = ReadInputFile(f1, False) # read input file and create node, element, material and step lists -> in ConFEM2D_InOut.py
        f1.close()
        # if LinAlgFlag:
        CuthillMckee(NodeList, ElList)  # to reduce the skyline
        NodeList.sort(key=lambda t: t.CMLabel_)
        N, Mask, Skyline, SDiag, SLen = AssignGlobalDof( NodeList, ElList, MatList)          # assign degrees of freedom (dof) to nodes and elements -> see above
    
        if pth.isfile(Name+".opt.txt"):                     # read options file if there is any
            f4=open( Name+".opt.txt", 'r')
            WrNodes, Lines, ReDes, MaxType = ReadOptionsFile(f4, NodeList)
            f4.close()
    
        if pth.isfile(Name+".pkl"):                 # read restart file if there is one
            fd = open(Name+'.pkl', 'r')                     #
            uuu = cPickle.Unpickler(fd)
            NodeList,ElList,MatList,StepList,N,WrNodes,LineS,Flag,\
                      VecU,VecC,VecI,VecP,VecP0,VecP0old,VecT,VecS,VeaU,VevU,VeaC,VevC,VecY,BCIn,Time,TimeOld,TimeEl,TimeNo,TimeS,i,Mask,Skyline,SDiag,SLen,SymSys\
                      = uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load(),uuu.load()
            fd.close()
        else: raise NameError ("PostFem: cannot read data")
    
        f2=open( Name+".elemout.txt", 'rb')                  #
        #RC = FinishAllStuff(True, Name, ElList, NodeList, MatList, f2, VecU, WrNodes, None)
        # Rewrite explicitly FinishAllStuff function with PlotF=True
        SE, SN, SP, PE2DFlag, mod_Z_limit, PE3DFlag = 1.0, 1.0, 1.0, True, 1.0, False
        if path.isfile(Name + ".plt.txt"):  # read plot options file if there is any
            f1 = open(Name + ".plt.txt", 'r')
            SE, SN, SP, PE2DFlag, mod_Z_limit, PE3DFlag = ReadPlotOptionsFile(f1)  # to modify plot scaling factors
        else:
            PE3DFlag = False
        Scales, Scales1 = PostScales(ElList, NodeList, f2)
        f2.seek(0)
        #if PE2DFlag: PostElem2D(ElList, NodeList, MatList, f2, Scales, Scales1, SE, SP)
        #if PE3DFlag: PostElem3D(ElList, NodeList, VecU, SN, mod_Z_limit)
        f2.seek(0)
        PostElem1D(f2)
        PostNode(ElList, NodeList, VecU, SN, mod_Z_limit, PE3DFlag)  # plot deformed mesh
        if WrNodes <> None and len(WrNodes) > 0:
            f5_name = Name + ".timeout.txt"
            try:
                PlotNodes(f5_name, WrNodes, PlotOpt=2)  # plot Displ-Reaction curve based on WrNode file
            except NameError:
                print NameError("ConFemInOut:PlotNodes: unknown key")
            except:
                print TypeError("ConFemInOut:PlotNodes: too much variables in .timeout file")
        show()


if __name__ == "__main__":
    Case = 0
    if Case == 0:
        Name = "C:/Users/regga/Desktop/ConFEM2D/examples/ex5/ex51"                                 # input data name
        PostFem_ = ConPostPlot()
        PostFem_.Run(Name)
    elif Case == 1:
        LogName="../LogFiles"                             # to log temporary data
        PlotResiduals( LogName, "log", 33, 24)
    elif Case == 2:
        LogName="../LogFiles"                             # to log temporary data
        WriteResiduals( LogName, 1, 8, 100)
    print 'finish'
