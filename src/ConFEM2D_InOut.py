# ConFemInOut - module for read input, output files.
#
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, grid, title, text, contour, clabel, show, axis, xticks, yticks, ylim, annotate
import matplotlib.colors as colors
import mpl_toolkits.mplot3d as a3
import pylab as pl
from pylab import arange, griddata
import numpy as np
from numpy import meshgrid, pi, arcsin, fabs, sqrt
from scipy.linalg import norm
from os import path, makedirs

from ConFEM2D_Basics import ResultTypes, PrinC
from lib_Mat import *
from lib_Elem import *
from ConFEM2D_Steps import *
from ConFemRandom import RandomField_Routines

Colors = ['indianred','red','magenta','darkviolet','blue','green','yellow']
FonSizTi='x-large'   # fontsize title x-large
FonSizAx='x-large'     # fontsize axis
#FonSizAx='medium'     # fontsize axis
LiWiCrv=3            # line width curve in pt
LiWiCrvMid=2            # line width curve in pt
LiWiCrvThin=0.5        # line width curve in pt

def ReadPlotOptionsFile(f1):                            # read plot options
    SE, SN, SP, PE2DFlag, mod_Z_limit, PE3DFlag  = 1.0, 1.0, 1.0, True, 1.0, False
    z1 = f1.readline()                                  # 1st input line
    z2 = z1.split(",")
    zLen = len(z2)
    for i in xrange(zLen): z2[i] = z2[i].strip()
    while z1<>"":
        for i in xrange(zLen): z2[i] = z2[i].strip()
        if z2[0]=="ScaleElementData": SE=float(z2[1])                         # 
        elif z2[0]=="ScaleNodalData": SN=float(z2[1])                         # 
        elif z2[0]=="ScalePstrainData": SP=float(z2[1])                         #
        elif z2[0]=="ScaleZ3D": mod_Z_limit=float(z2[1])                         #
        elif z2[0]=="PlotElements2D":
             if int(z2[1]) == 0: PE2DFlag = False
        elif z2[0]=="PlotElements3D":
             if int(z2[1]) == 1: PE3DFlag = True
        z1 = f1.readline()                              # read next line
        z2 = z1.split(",")
        zLen = len(z2)
    return SE, SN, SP, PE2DFlag, mod_Z_limit, PE3DFlag

def ReadOptionsFile(f1, NodeList):                      # read options
    z1 = f1.readline()                                  # 1st input line
    z2 = z1.split(",")
    zLen = len(z2)
    LS = [1,1.]                                         # default values for line search
    R, L, MaxType = [], [], []                                        
    for i in xrange(zLen): z2[i] = z2[i].strip()
    while z1<>"":
        for i in xrange(zLen): z2[i] = z2[i].strip()
        if z2[0]=="writeNodes":                         # write single nodal displacement, force during whole course of computation
            n = (len(z2)-1)/3                           # three data - node, dof, outtype - expected per item  
            for j in xrange(n):
                iP = int(z2[3*j+1])                     # node
                iD = int(z2[3*j+2])                   # dof   
                ty = z2[3*j+3]                          # outtypes p-u, p-t, u-t, b-u, b-t
                jj = FindIndexByLabel( NodeList, iP)    # find node index
                if jj<0: 
                    print iP
                    raise NameError ("no such node available in current dataset")
                DofTL = list(NodeList[jj].DofT)         # DofT is set of markers for dof types
                iD_ = -1
                for i in xrange(len(DofTL)):
                    if DofTL[i]==iD: iD_ = i
                if iD_==-1: raise NameError ("ConFemInOut::ReadOptionsFile: invalid dof type indicator ")
                ii = NodeList[jj].GlobDofStart+iD_      # global index history plot
                L += [[iP,iD_,ii,ty]]
        elif z2[0]=="LineSearch":                       # parameter for line search iteration method
            if int(z2[1])>1:
                LS[0] = int(z2[1])
                LS[1] = float(z2[2])
        elif z2[0]=="reinforcementDesign":              # parameter reinforcement design - ConPlad
            if int(z2[1])>1:
                R += [float(z2[1])]                     # design value reinforcement strength
                R += [float(z2[2])]                     # design value concrete compressive strength
                R += [float(z2[3])]                     # geometric height
                R += [float(z2[4])]                     # structural height
        elif z2[0]=="MaxValues": MaxType = z2[1:]
        z1 = f1.readline()                              # read next line
        z2 = z1.split(",")
        zLen = len(z2)
    return L, LS, R, MaxType

def WriteNodes( f5, WrNodes, Time, VecU, VecB, VecP):   # write single nodal displacement, force during whole course of computation into file
    for j in xrange(len(WrNodes)):
        ii = WrNodes[j][2]
        if WrNodes[j][3]=='u-t': f5.write('%8.4f%12.6f'%(Time,VecU[ii]))
        elif WrNodes[j][3]=='t-u': f5.write('%12.6f%8.4f'%(fabs(VecU[ii]),Time,))
        elif WrNodes[j][3]=='b-u': f5.write('%14.8f%12.6f'%(VecU[ii],VecB[ii]))
        elif WrNodes[j][3]=='b-t': f5.write('%8.4f%12.6f'%(Time,VecB[ii]))
        elif WrNodes[j][3]=='p-u': f5.write('%14.8f%14.8f'%(VecU[ii],VecP[ii]))
#        elif WrNodes[j][3]=='p-u': f5.write('%12.6f%12.6f'%(-VecU[ii],Time))
        elif WrNodes[j][3]=='p-t': f5.write('%8.4f%12.6f'%(Time,VecP[ii]))
        else: raise NameError ("no type defined for writeNodes in *.opt.txt")
    f5.write('\n')
    return 0

def PlotNodes( filename, WrNodes, PlotOpt ):
    """
    :param filename:
    :param WrNodes: dtype=list.
    :param PlotOpt: option 1 = subplots in figure. option 2= all curves in one figure
    :return: plotting figure
    """
    Colors = ['red', 'magenta', 'green', 'yellow', 'blue', 'cyan']
    FonSizTi = 'x-large'  # fontsize title x-large
    FonSizAx = 'x-large'  # fontsize axis
    # FonSizAx='medium'     # fontsize axis
    LiWiCrv = 3  # line width curve in pt
    LiWiCrvMid = 2  # line width curve in pt
    LiWiCrvThin = 0.5  # line width curve in pt

    nfig = len(WrNodes)
    if PlotOpt == 1:           # option 1 : figure arrangement by subplot
        fig, ax = plt.subplots(nrows=nfig)
        for i in xrange(nfig):
            with open(filename, 'rb') as f5:
                if WrNodes[i][3] == 'p-u': label = 'load-displacement node=%s dof=%s' %(WrNodes[i][0], WrNodes[i][1])
                elif WrNodes[i][3] == 'p-t': label = 'load-time node=%s dof=%s' %(WrNodes[i][0], WrNodes[i][1])
                elif WrNodes[i][3] == 'u-t': label = 'displacement-time node=%s dof=%s' %(WrNodes[i][0], WrNodes[i][1])
                elif WrNodes[i][3] == 'b-t': label = 'reaction force-time node=%s dof=%s' %(WrNodes[i][0], WrNodes[i][1])
                elif WrNodes[i][3] == 'b-u': label = 'reaction force-displacement node=%s dof=%s' %(WrNodes[i][0], WrNodes[i][1])
                else: raise NameError("ConFemInOut:PlotNodes: unknown key")
                sc1, sc2 = 1., 1.
                L1, L2 = [0.], [0.]
                z1 = f5.readline()
                z2 = z1.split()
                L1 += [sc1 * float(z2[2 * i])]
                L2 += [sc2 * float(z2[2 * i + 1])]
                while z1 <> '':     # retrieve data
                    z1 = f5.readline()
                    z2 = z1.split()
                    if len(z2) > 0:
                        L1 += [sc1 * float(z2[2 * i])]
                        L2 += [sc2 * float(z2[2 * i + 1])]
                    else:
                        continue
            P0 = ax[i].plot(L1, L2, c=Colors[i], lw=LiWiCrv)
            ax[i].set_title(label, fontsize=FonSizTi)
            ax[i].grid()
        plt.subplots_adjust(hspace=0.5)
    if PlotOpt == 2:    # option 2 : all curves in one figure
        PP = plt.figure()
        P0 = PP.add_subplot(111)
        with open(filename,'rb') as f5:
            L1, L2 = list(), list()
            label = []
            for i in xrange(nfig):
                if WrNodes[i][3] == 'p-u': label += ['load-displacement node=%s dof=%s' % (WrNodes[i][0], WrNodes[i][1]) ]
                elif WrNodes[i][3] == 'p-t': label += ['load-time node=%s dof=%s' % (WrNodes[i][0], WrNodes[i][1]) ]
                elif WrNodes[i][3] == 'u-t': label += ['displacement-time node=%s dof=%s' % (WrNodes[i][0], WrNodes[i][1])]
                elif WrNodes[i][3] == 'b-t': label += ['reaction force-time node=%s dof=%s' % (WrNodes[i][0], WrNodes[i][1])]
                elif WrNodes[i][3] == 'b-u': label += ['reaction force-displacement node=%s dof=%s' % (WrNodes[i][0], WrNodes[i][1])]
                else: raise NameError("ConFemInOut:PlotNodes: unknown key")
                L1 += [[0.]]
                L2 += [[0.]]
            sc1, sc2 = 1., 1.
            z1 = f5.readline()
            z2 = z1.split()
            for i in xrange(nfig):
                L1[i] += [sc1*float(z2[2*i])]
                L2[i] += [sc2*float(z2[2*i+1])]
            while z1 <> '':     # retrieve data
                z1 = f5.readline()
                z2 = z1.split()
                if len(z2)>0:
                    for i in xrange(nfig):
                        L1[i] += [sc1 * float(z2[2 * i])]
                        L2[i] += [sc2 * float(z2[2 * i + 1])]
                else : continue
            for i in xrange(nfig):
                P0.plot(L1[i], L2[i], linewidth=LiWiCrvMid, c=Colors[i], ls='-')
                P0.legend(label, loc='best')
        P0.grid()
        PP.autofmt_xdate()

    plt.show()
    return 0

def ReadInputFile(f1, Restart):                         # read data
    NodeList = []
    ElList = []
    StepList = []
    MatList = {}                                        # dictionary for materials
    ElList_ = []                                        # list for elements, local use only
    SecDic = {}                                         # dictionary for sections, local use only
    BreiDic = {}                                        # dictionary for beam reinforcement, local use only, key is a label, data is list of lists
    BreiDic["DEFAULT"] = []                             # default value empty dic / no reinforcement
    z1 = f1.readline()                                  # 1st input line
    z2 = z1.split(",")
    z2 = z2[0].strip()
    IType = ""
    ITyp2 = ""
    # single input line processing
    while z1<>"":
        if z2[0]=="*NODE":                              # key NODE
            IType = "node"
        ##################################################################
        elif z2[0]=="*BEAM REINF":                      # key BEAM REINFORCEMENT - another option for input of reinforcement geometry presumably if beam section is not RECT
            IType = "beamreinforcement"
            for i in xrange(1,zLen):                    # loop starts with 1
                if z2[i].find("NAME")>-1:
                    breiName=z2[i].split("=")[1]         # set current reinforcement name
                    if BreiDic.get(breiName)<>None: raise NameError ("ConFemInOut::ReadInputFile: beam reinf with same name")
            linecounter = 0
        ##################################################################
        elif z2[0]=="*BEAM SECTION":                    # key BEAM SECTION
            IType = "beamsection"
            reinf = "DEFAULT"
            for i in xrange(1,zLen):                    # loop starts with 1
                if z2[i].find("SECTION")>-1: beamSecType=z2[i].split("=")[1]
                elif z2[i].find("ELSET")>-1: elset=z2[i].split("=")[1]
                elif z2[i].find("MATERIAL")>-1: mat=z2[i].split("=")[1]
                elif z2[i].find("REINF")>-1: reinf=z2[i].split("=")[1]
            linecounter = 0
        ###################################################################
        elif z2[0]=="*SOLID SECTION":                   # key SOLID SECTION
            IType = "solidsection"
            for i in xrange(1,zLen):                    # loop starts with 1
                if z2[i].find("ELSET")>-1: elset=z2[i].split("=")[1]
                elif z2[i].find("MATERIAL")>-1: mat=z2[i].split("=")[1]
            linecounter = 0
        ###################################################################
        elif z2[0]=="*SHELL SECTION":                   # key SOLID SECTION
            IType = "shellsection"
            for i in xrange(1,zLen):                    # loop starts with 1
                if z2[i].find("ELSET")>-1: elset=z2[i].split("=")[1]
                elif z2[i].find("MATERIAL")>-1: mat=z2[i].split("=")[1]
            linecounter = 0
        ##############################################################
        elif z2[0]=="*MATERIAL":                        # key MATERIAL
            IType = "material"
            mass = 0.                                   # default value specific mass
            RegType = 0                                 # default no regularization
            for i in xrange(1,zLen):
                if z2[i].find("NAME")>-1:
                    matName=z2[i].split("=")[1]         # set current matname
                    if MatList.get(matName)<>None: raise NameError ("material with same name already exists")
            linecounter = 0
        #############################################################
        elif z2[0]=="*ELEMENT":                         # key ELEMENT
            IType = "element"
            for i in xrange(1,zLen):
                if z2[i].find("TYPE")>-1: eltype=z2[i].split("=")[1]# element type
                elif z2[i].find("ELSET")>-1: elset=z2[i].split("=")[1]# element set
            mat = SecDic[elset].Mat                     # assigned material name
            MatList[mat].Used = True
            if MatList.get(mat)==None: raise NameError ("no such material available in current dataset")
            if (eltype=='B23E' or eltype=='B23'):
                if isinstance( MatList[mat], RCBeam): ResultTypes[elset]=('longitudinal strain','curvature','normal force','moment','max reinf strain','min conc strain')
                elif isinstance( MatList[mat], Elastic): ResultTypes[elset]=('longitudinal strain','curvature','normal force','moment')
            elif (eltype=='B21' or eltype=='B21E'):
                if isinstance( MatList[mat], RCBeam):    ResultTypes[elset]=('longitudinal strain','curvature','shear def','normal force','moment','shear force','max reinf strain','min conc strain')
                elif isinstance( MatList[mat], Elastic): ResultTypes[elset]=('longitudinal strain','curvature','shear def','normal force','moment','shear force')
            elif eltype=='T1D2' or eltype=='T2D2' or eltype=='T3D2':
                if isinstance( MatList[mat], Mises): mat = mat+'1'                  # becomes uniaxial Mises 
                ResultTypes[elset]=('strain','stress')
            elif eltype=='S1D2': ResultTypes[elset]=('displ','stress')
            elif eltype=='CPE4' or eltype=='CPE4R' or eltype=='CPE3': ResultTypes[elset]=('strain','stress')
            elif eltype=='CPS4' or eltype=='CPS4R' or eltype=='CPS3': ResultTypes[elset]=('strain','stress')
#            elif eltype=='T2D2': ResultTypes[elset]=('strain','stress')
#            elif eltype=='T3D2': ResultTypes[elset]=('strain','stress')
            elif eltype=='SB3':  ResultTypes[elset]=('curv x','curv y','curv xy','mom x','mom y','mom xy')
            elif eltype=='SH4':  ResultTypes[elset]=('n_x','n_y','n_xy','m_x','m_y','m_xy','q_x','q_y')
            elif eltype=='SH3':  ResultTypes[elset]=('n_x','n_y','n_xy','m_x','m_y','m_xy','q_x','q_y')
            elif eltype=='C3D8': ResultTypes[elset]=('strain','stress')
            elif eltype=='S2D6': ResultTypes[elset]=('displ','force')
            else: raise NameError ("no result types for this element defined")
        ##########################################################
        elif z2[0]=="*STEP":                            # key STEP
            ### build final list for elements #############################################################
            if len(ElList)==0:
                for i in range(len(ElList_)):
                    eltype = ElList_[i][0]
                    if   eltype=='B23':  ElList +=[B23( ElList_[i][1],ElList_[i][2],ElList_[i][3],ElList_[i][4],ElList_[i][5],ElList_[i][6],BreiDic[ElList_[i][7]],MatList[ElList_[i][4]].StateVar, MatList[ElList_[i][4]].NData)]
                    elif eltype=='B23E': ElList +=[B23E(ElList_[i][1],ElList_[i][2],ElList_[i][3],ElList_[i][4],ElList_[i][5],ElList_[i][6],BreiDic[ElList_[i][7]],MatList[ElList_[i][4]].StateVar, MatList[ElList_[i][4]].NData,ElList_[i][8])]
                    elif eltype=='B21':  ElList +=[B21( ElList_[i][1],ElList_[i][2],ElList_[i][3],ElList_[i][4],ElList_[i][5],ElList_[i][6],BreiDic[ElList_[i][7]],MatList[ElList_[i][4]].StateVar, MatList[ElList_[i][4]].NData)]
                    elif eltype=='B21E': ElList +=[B21E(ElList_[i][1],ElList_[i][2],ElList_[i][3],ElList_[i][4],ElList_[i][5],ElList_[i][6],BreiDic[ElList_[i][7]],MatList[ElList_[i][4]].StateVar, MatList[ElList_[i][4]].NData)]
#                   elif eltype=='S2D6': ElList +=[S2D6(              Label,        SetLabel,     InzList,                 MatName,       NoList,        SolSecDic,             StateV,                           NData)]
#                       elif eltype=='S2D6': ElList_+=[[eltype,       int(z2[0]),   elset,        [int(z2[1]),int(z2[2])], mat,           NodeList,      SecDic[elset]]]
                    elif eltype=='S2D6': ElList +=[S2D6(              ElList_[i][1],ElList_[i][2],ElList_[i][3],           ElList_[i][4], ElList_[i][5], SecDic[ElList_[i][2]], MatList[ElList_[i][4]].StateVar, MatList[ElList_[i][4]].NData)]
            ##############################################################################################
            IType = "step"
            StepList += [Step()]
            for i in xrange(1,zLen):                    # loop starts with 1
                if z2[i].find("NLGEOM")>-1: 
                    if z2[i].split("=")[1]=="YES": StepList[len(StepList)-1].NLGeom = True
            BList = []                                  # temp list for boundary conditions
            DList = []                                  # temp list for distributed / element loads
            CList = []                                  # temp list for nodal loads
            TList = []                                  # temp list for nodal temperatures
            ElFiList = False                            # temp flag for element output
            ElNoList = False                            # temp flag for node output
            Dalph, Dbeta = 0., 0.
        elif z2[0]=="" or z2[0].find("**")>-1:          # blank line / comment line
            pass
        elif z2[0].find("*")>-1 and IType<>"material" and IType<>"step":# unknown key, comment, connected line
            IType = ""
        ## bulk data ##############################################
        elif IType<>"":
            if IType=="node":                           # data NODE
                if zLen<3: raise NameError ("not enough nodal data")
                if zLen==3:   NodeList += [Node( int(z2[0]), float(z2[1]), float(z2[2]), 0., 0.,0.,0. )]
                elif zLen==4: NodeList += [Node( int(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), 0.,0.,0. )]
                elif zLen==7: NodeList += [Node( int(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), float(z2[6]) )]
            ################################################################
            elif IType=="beamreinforcement":                 # data SHELL SECTION
                if zLen<4: raise NameError ("too less reinforcement data")
                if zLen>4 and z2[4][0:2]=='TC': RType='TC' # textile or other reinforcement than ordinary rebars
                else:                           RType='RC'
                if linecounter==0:
                    BreiDic[breiName] = [[float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),RType]] # reinforcement cross section and tension stiffening data
                    linecounter = linecounter+1
                else:
                    BreiDic[breiName] += [[float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),RType]] # reinforcement cross section and tension stiffening data
            ###############################################################
            elif IType=="beamsection":                  # data BEAM SECTION - these data are transferred to beam element constructor data and do not leave this method
                if beamSecType=="RECT":
                    if linecounter==0:
                        SecDic[elset] = BeamSectionRect( beamSecType, elset, mat, float(z2[0]), float(z2[1]), []) # concrete cross section data
                        linecounter = linecounter+1
                    else:
                        if zLen<4: raise NameError ("too less reinforcement data")
                        if zLen>4 and z2[4][0:2]=='TC': RType='TC' # textile or other reinforcement than ordinary rebars
                        else:                           RType='RC'
                        SecDic[elset].Reinf += [[float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),RType]] # reinforcement cross section and tension stiffening data
                else:
                    if linecounter==0:
                        SecDic[elset] = BeamSectionPoly( beamSecType, elset, mat, []) # concrete cross section data
                        SecDic[elset].AddPoint(float(z2[0]), float(z2[1]))
                        linecounter = linecounter+1
                    else:
                        SecDic[elset].AddPoint(float(z2[0]), float(z2[1]))
            ################################################################
            elif IType=="shellsection":                 # data SHELL SECTION
                if linecounter==0:
                    SecDic[elset] = ShellSection( elset, mat, float(z2[0]), [] )
                    linecounter = linecounter+1
                else:
                    SecDic[elset].Reinf += [[float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4])]] # reinforcement cross section and tension stiffening data, see also ConFemEelem::SH4.__init__
            ################################################################
            elif IType=="solidsection":                 # data SOLID SECTION
                SecDic[elset] = SolidSection( elset, mat, float(z2[0]))
            ########################################################################
            elif IType=="material":       
                ############################################################
                if ITyp2=="density":                    # data MATERIAL
                    mass = float(z2[0])
                    ITyp2 = ""
                elif ITyp2=="rcBeam":                   # data MATERIAL rcBeam -- also notice tcbeam 
                    if linecounter==0:                  # material data concrete
                        if    zLen<9: raise NameError ("too less concrete data")
                        elif zLen==9: L1 = [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),int(z2[5]),float(z2[6]),float(z2[7]),float(z2[8]),mass] # material data concrete
                        else:         L1 = [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),int(z2[5]),float(z2[6]),float(z2[7]),float(z2[8]),mass,float(z2[9])] # material data concrete
                        linecounter = linecounter+1
                    else:                               # material data reinforcement of mises type
                        if   zLen==6: val = 0.
                        elif zLen==7: val = float(z2[6])
                        else: raise NameError ("ConFemInOut::ReadInputFile: number of mises data")
                        MatList[matName] = RCBeam( [L1, [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),float(z2[5]), val, mass]])  # material data reinforcement
                        ITyp2 = ""
                # presumably obsolete, covered by RCSHELL
#                elif ITyp2=="elasticLTRCS":             # data MATERIAL elasticLT for reinforced shell
#                    if linecounter==0:
#                        if zLen<9: raise NameError ("too less data for elasticlt with reinforced shell")
#                        L1 = [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),float(z2[5]),float(z2[6]),int(z2[7]),float(z2[8]), mass]
#                        linecounter = linecounter+1
#                    else:
#                        if   zLen==6: val = 0.
#                        elif zLen==7: val = float(z2[6])
#                        else: raise NameError ("ConFemInOut::ReadInputFile: number of reinforcement data for shell")
#                        L2=[float(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), val, mass]
#                        MatList[matName] = WraElasticLTReShell([L1, L2])
#                        ITyp2 = ""
                elif ITyp2=="rcshell":                  # data MATERIAL wrapper for reinforced shell
                    if   zLen==7: val, matn, matT = 0., z2[6], None
                    elif zLen==8: val, matn, matT = float(z2[6]), z2[7], None
                    elif zLen==9: val, matn, matT = float(z2[6]), z2[7], z2[8]   # to allow for a selection of reinforcement type: TR -> TextileR, otherwise -> MisesUniaxial
                    if matn in MatList: L1 = MatList[matn].PropMat   
                    else: raise NameError ("ConFemInOut::ReadInputFile: unknown bulk for RC shell wrapper - maybe wrong sequence of materials in input")
                    L2 = [float(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), val, mass] # see mises:__init__ for meaning of parameters
                    MatList[matName] = WraRCShell([L1, L2], MatList[matn], matT)
                    ITyp2 = ""
                elif ITyp2=="tcBeam":                   # data MATERIAL tcBeam -- also notice rcbeam
                    if linecounter==0:                  # material data concrete
                        if zLen<9:    raise NameError ("too less concrete data")
                        elif zLen==9: L1 = [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),int(z2[5]),float(z2[6]),float(z2[7]),float(z2[8]),mass] # material data concrete
                        else:         L1 = [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),int(z2[5]),float(z2[6]),float(z2[7]),float(z2[8]),mass,float(z2[9])] # material data concrete
                        linecounter = linecounter+1
                    elif linecounter==1:                # material data reinforcement of mises type
                        if   zLen==6: val = 0.
                        elif zLen==7: val = float(z2[6])# additional value to smooth elastic / plastic transition 
                        else: raise NameError ("ConFemInOut::ReadInputFile: number of reinforcement data for beam")
                        L2 = [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),float(z2[5]), val, mass] # material data reinforcement
                        linecounter = linecounter+1
                    else:                               # complete previous with TC (or so) reinforcement
                        MatList[matName] = WraTCBeam( [L1, L2, [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3])]]) # material data reinforcement
                        ITyp2 = ""
                elif ITyp2=="tcBar":
                    MatList[matName] = TextileR( [float(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), mass])
                    ITyp2 = ""
                elif ITyp2=="elastic":                  # data MATERIAL elastic
                    MatList[matName] = Elastic( [float(z2[0]), float(z2[1]), mass])
                    ITyp2 = ""
                elif ITyp2=="elasticOR":                # data MATERIAL elastic orthogonal
                    MatList[matName] = ElasticOrtho( [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),float(z2[5]),float(z2[6]),float(z2[7]),float(z2[8]), mass])
                    ITyp2 = ""
                elif ITyp2=="isodamage":                # data MATERIAL isotropic damage
                    MatList[matName] = IsoDamage([float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),int(z2[4]),float(z2[5]),float(z2[6]),float(z2[7]),int(z2[8]),float(z2[9]),float(z2[10]), mass])
                    RegType = int(z2[8])
                    ITyp2 = ""
                elif ITyp2=="elasticLT":                # data MATERIAL elasticLT
                    if zLen<9: raise NameError ("too less elasticlt data")
                    MatList[matName] = ElasticLT([float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),float(z2[5]),float(z2[6]),int(z2[7]),float(z2[8]), mass] )
                    ITyp2 = ""
                elif ITyp2=="spring":                   # data MATERIAL spring
                    MatList[matName] = Spring( [float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), float(z2[6])] )
                    ITyp2 = ""
                elif ITyp2=="mises":
                    if   zLen==6: val = 0.
                    elif zLen==7: val = float(z2[6])
                    else: raise NameError ("ConFemInOut::ReadInputFile: number of mises data")
                    MatList[matName] =     Mises( [float(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), val, mass], [0, 0] )
                    MatList[matName+'1'] = MisesUniaxial( [float(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), val, mass], [0, 0] ) # for uniaxial behavior
                    ITyp2 = ""
                elif ITyp2=="template":
                    if   zLen==6: pass
                    else: raise NameError ("ConFemInOut::ReadInputFile: number of template data")
                    MatList[matName] = Template( [float(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), mass])
                    ITyp2 = ""
                elif ITyp2=="lubliner":
                    if   zLen==11: pass
                    else: raise NameError ("ConFemInOut::ReadInputFile: number of Lubliner data")
                    MatList[matName] = Lubliner( [float(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), float(z2[6]), float(z2[7]), float(z2[8]), float(z2[9]), float(z2[10]), mass])
                    ITyp2 = ""
                elif ITyp2=="misesReMem":               # data MATERIAL ansisotropic reinforcement membrane with mises
                    if   zLen==6: val = 0.
                    elif zLen==7: val = float(z2[6])
                    else: raise NameError ("ConFemInOut::ReadInputFile: number of misesReMem data")
                    MatList[matName] = WraMisesReMem( [float(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), val, mass], [0, 0] )
                    ITyp2 = ""
                elif ITyp2=="nlslab":
                    MatList[matName] = NLSlab( [float(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), float(z2[6])] )
                    ITyp2 = ""
                elif ITyp2=="mp_dam1":
                    MatList[matName] = MicroPlane( [float(z2[0]), float(z2[1]), int(z2[2]), float(z2[3]), float(z2[4]), float(z2[5]), float(z2[6]), float(z2[7]), float(z2[8]), float(z2[9]), mass])
                    ITyp2 = ""
                ########################################################
                if z2[0]=="*DENSITY":     ITyp2="density" # must precede all other material specifications, if defined
                ########################################################
                elif z2[0]=="*RCBEAM":                  # sub-items MATERIAL
                    ITyp2="rcBeam"
                    linecounter = 0                     # more than one material data line
                elif z2[0]=="*TCBEAM":
                    ITyp2="tcBeam"
                    linecounter = 0                     # more than one material data line
                elif z2[0]=="*TCTBAR":    ITyp2="tcBar" # TRC Tensile Bar 1D
                elif z2[0]=="*ELASTIC":   ITyp2="elastic"
                elif z2[0]=="*ELASTICORTHO":      ITyp2="elasticOR"
                elif z2[0]=="*ELASTICLT": ITyp2="elasticLT"
#                elif z2[0]=="*ELASTICLT_RCSHELL": ITyp2="elasticLTRCS"  # obsolete, covered by RCSHELL 
                elif z2[0]=="*RCSHELL": ITyp2="rcshell"
                elif z2[0]=="*ISODAMAGE": ITyp2="isodamage"
                elif z2[0]=="*SPRING":    ITyp2="spring"
                elif z2[0]=="*MISES":     ITyp2="mises"
                elif z2[0]=="*TEMPLATE":     ITyp2="template"
                elif z2[0]=="*LUBLINER":     ITyp2="lubliner"
                elif z2[0]=="*MISESREMEM":ITyp2="misesReMem"
                elif z2[0]=="*NLSLAB":    ITyp2="nlslab"
                elif z2[0]=="*MICROPLANE_DAMAGE1":    ITyp2="mp_dam1"
            ##########################################################
            elif IType=="element":                      # data ELEMENT
                # what follows refers to ElList_
                if eltype=='B23':    ElList_+=[[eltype, int(z2[0]),elset,[int(z2[1]),int(z2[2])],mat,NodeList,SecDic[elset],reinf]]
                elif eltype=='B23E': 
                    if len(z2)==6:          
                        '''accept 2 randomly distributed variables'''
                        RandomField_Routines().add_elset(elset)
                        RandomField_Routines().add_propertyEntry(float(z2[4]),elset,int(z2[0]))
                        RandomField_Routines().add_propertyEntry(float(z2[5]),elset,int(z2[0]))
                    ElList_+=[[eltype, int(z2[0]),elset,[int(z2[1]),int(z2[2]),int(z2[3])],                 mat,NodeList,SecDic[elset],reinf,True]]
                elif eltype=='B21':  ElList_+=[[eltype, int(z2[0]),elset,[int(z2[1]),int(z2[2])],           mat,NodeList,SecDic[elset],reinf]]
                elif eltype=='B21E': ElList_+=[[eltype, int(z2[0]),elset,[int(z2[1]),int(z2[2]),int(z2[3])],mat,NodeList,SecDic[elset],reinf]]
                elif eltype=='S2D6': ElList_+=[[eltype, int(z2[0]),elset,[int(z2[1]),int(z2[2])],           mat,NodeList,SecDic[elset]]]
                # what follows refers to ElList
                elif eltype=='T1D2': 
                    if len(z2)==4:
                        '''accept 1 randomly distributed variables'''
                        RandomField_Routines().add_elset(elset)
                        RandomField_Routines().add_propertyEntry(float(z2[3]),elset,int(z2[0]))
                    ElList += [T1D2( int(z2[0]), elset, [int(z2[1]),int(z2[2])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData, MatList[mat].RType)]
                elif eltype=='S1D2': 
                    ElList += [S1D2( int(z2[0]), elset, [int(z2[1]),int(z2[2])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData)]
                elif eltype=='CPS3': 
                    ElList += [CPE3( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData, True, 1)]
                elif eltype=='CPE3': 
                    ElList += [CPE3( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData, False, 1)]
                elif eltype=='CPS4': 
                    ElList += [CPE4( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3]),int(z2[4])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData, True, 2, MatList[mat].RType)]
                elif eltype=='CPS4R':
                    ElList+= [CPE4( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3]),int(z2[4])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData, True, 1, MatList[mat].RType)]
                elif eltype=='CPE4': 
                    ElList += [CPE4( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3]),int(z2[4])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData, False, 2, MatList[mat].RType)]
                elif eltype=='CPE4R':
                    ElList+= [CPE4( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3]),int(z2[4])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData, False, 1)]
                elif eltype=='T2D2':
                    if zLen==3: Val = 1.0
                    else:       Val = float(z2[3])
                    ElList += [T2D2( int(z2[0]), elset, [int(z2[1]),int(z2[2])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData,Val)]
                elif eltype=='T3D2':
                    if zLen==3: Val = 1.0
                    else:       Val = float(z2[3])
                    ElList += [T3D2( int(z2[0]), elset, [int(z2[1]),int(z2[2])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData,Val)]
                elif eltype=='SB3':
                    ElList+= [SB3( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3])], mat, NodeList, SecDic[elset], MatList[mat].StateVar, MatList[mat].NData)]
                elif eltype=='SH4':
                    NData = MatList[mat].NData
                    RCFlag = False
#                    if isinstance( MatList[mat], WraElasticLTReShell) or isinstance( MatList[mat], WraRCShell): 
                    if  isinstance( MatList[mat], WraRCShell): 
                        NData = NData+6
                        RCFlag = True
                        if MatList[mat].StateVar==None: NStateVar = 5                               # for mises reinforcement, TR included (-> 1)
                        else:                           NStateVar = max(5,MatList[mat].StateVar)    # 1st for mises reinforcement, 2nd for bulk material -- both use different ip points, see SH4.__init__  
                    else:                               NStateVar = MatList[mat].StateVar
                    ElList+= [SH4( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3]),int(z2[4])], mat, NodeList, SecDic[elset], NStateVar, NData, RCFlag)]
                elif eltype=='SH3':
                    NData = MatList[mat].NData
                    NStateVar = MatList[mat].StateVar
                    ElList+= [SH3( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3])],            mat, NodeList, SecDic[elset], NStateVar, NData, False)]
                elif eltype=='C3D8':
                    ElList += [C3D8( int(z2[0]), elset, [int(z2[1]),int(z2[2]),int(z2[3]),int(z2[4]),int(z2[5]),int(z2[6]),int(z2[7]),int(z2[8])] , mat, NodeList, MatList[mat].StateVar, MatList[mat].NData, MatList[mat].RType)]
                else: raise NameError ("unknown element type")
            ###########################################################
            elif IType=="step":                         # subitems STEP
                if ITyp2=="stepControls0":              # data STEP CONTROLS
                    StepList[len(StepList)-1].IterTol = float(z2[0])
                    ITyp2 = ""
                elif ITyp2=="stepControls1":            # data STEP CONTROLS
                    StepList[len(StepList)-1].IterNum = int(z2[0])
                    ITyp2 = ""
                elif ITyp2=="stepStatic" or ITyp2=="stepDynamic": # data STEP STATIC/DYNAMIC
                    if StepList[len(StepList)-1].varTimeSteps: 
                        ITyp2 = "stepStaticVar"
                        StepList[len(StepList)-1].TimeStepVar = [float(z2[0])]
                        StepList[len(StepList)-1].TimeTargVar = [float(z2[1])]
                    else:
                        StepList[len(StepList)-1].TimeStep = float(z2[0])
                        StepList[len(StepList)-1].TimeTarg = float(z2[1])
                        if StepList[len(StepList)-1].ArcLen: StepList[len(StepList)-1].ArcLenV = float(z2[2])
                        if ITyp2=="stepDynamic": StepList[len(StepList)-1].Dyn = True
                        ITyp2 = ""
                elif ITyp2=="stepStaticVar":
                    try:
                        StepList[len(StepList)-1].TimeStepVar.append(float(z2[0]))
                        StepList[len(StepList)-1].TimeTargVar.append(float(z2[1]))
                    except ValueError:
                        ITyp2 = ""
                elif ITyp2=="stepDamping":
                    StepList[len(StepList)-1].Damp = True
                    StepList[len(StepList)-1].RaAlph = Dalph
                    StepList[len(StepList)-1].RaBeta = Dbeta
                    ITyp2 = ""
                elif ITyp2=="stepBuckling":
                    StepList[len(StepList)-1].Buckl = True
                    ITyp2 = ""
                elif ITyp2=="stepBoundary":             # data STEP BOUNDARY
                    if z2[0].find("*")>-1:
                        StepList[len(StepList)-1].BoundList = BList
                        ITyp2 = ""
                    else:
                        for i in range(int(z2[2])-int(z2[1])+1): BList += [Boundary( int(z2[0]), int(z2[1])+i, float(z2[3]), NodeList, AmpLbl, AddVal)]
                elif ITyp2=="stepDLoad":                # data STEP DLOAD
                    if z2[0].find("*")>-1:
                        StepList[len(StepList)-1].DLoadList = DList
                        ITyp2 = ""
                    else:
                        DList += [DLoad( z2[0], int(z2[1]), float(z2[2]), AmpLbl)]
                elif ITyp2=="stepCLoad":                # data STEP CLOAD
                    if z2[0].find("*")>-1:
                        StepList[len(StepList)-1].CLoadList = CList
                        ITyp2 = ""
                    else:
                        CList += [CLoad( int(z2[0]), int(z2[1]), float(z2[2]), NodeList, AmpLbl)]
                elif ITyp2=="stepTemp":                 # data STEP TEMPERATURE
                    if z2[0].find("*")>-1:
                        StepList[len(StepList)-1].TempList = TList
                        ITyp2 = ""
                    else:
                        TList += [Temperature( int(z2[0]), [float(z2[i]) for i in xrange(1,len(z2))], NodeList, AmpLbl)]
                elif ITyp2=="stepPrestress":            # data STEP PRESTRESS
                    if z2[0].find("*")>-1:
                        if PreName not in Step.PreSDict: Step.PreSDict[PreName] = PreStress(PreType, L1, PList, AmpLbl)
                        StepList[len(StepList)-1].PrestList += [PreStreSt(PreName, AmpLbl)]
                        ITyp2 = ""
                    else:
                        if linecounter == 0:
                            if   zLen==7: val = 0.
                            elif zLen==8: val = float(z2[7])
                            else: raise NameError ("ConFemInOut::ReadInputFile: number of reinforcement data")
                            L1 = [float(z2[0]),float(z2[1]),float(z2[2]),float(z2[3]),float(z2[4]),float(z2[5]),float(z2[6]),val,mass] # prestressing parameter
                            linecounter = linecounter +1
                        else: PList += [[ FindIndexByLabel(ElList, int(z2[0])), [float(z2[i]) for i in xrange(1,len(z2))] ]]
                elif ITyp2=="stepElFile":               # data STEP EL File
                    if z2[0].find("*")>-1 and ElFiList:
                        StepList[len(StepList)-1].ElFilList += [ElFile(OutFr)]
                        ITyp2 = ""
                    else:
                        if set(z2)<=set(["E","S"]):
                            ElFiList = True
                elif ITyp2=="stepNoFile":               # data STEP NODE File
                    if z2[0].find("*")>-1 and ElNoList:
                        StepList[len(StepList)-1].NoFilList += [NoFile(OutFr)]
                        ITyp2 = ""
                    else:
                        if set(z2)<=set(["U"]):
                            ElNoList = True
                elif ITyp2=="stepAmp":                  # data STEP AMPLITUDE
                    AList = []
                    for i in xrange(0,zLen,2): 
                        try:
                            AList += [[float(z2[i]),float(z2[i+1])]]
                        except:
                            break 
                    StepList[len(StepList)-1].AmpDict[AmpName] = AList
                    ITyp2 = ""
                # if no data in current line, check upon key
                if z2[0]=="*CONTROLS" and z2[1]=="PARAMETERS=FIELD":# key STEP CONTROLS FIELD
                    ITyp2 = "stepControls0"
                elif z2[0]=="*CONTROLS" and z2[1]=="PARAMETERS=TIME INCREMENTATION":# key STEP CONTROLS TIME INCREMENTATION
                    ITyp2 = "stepControls1"
                elif z2[0]=="*STATIC":                  # key STEP STATIC
                    ITyp2 = "stepStatic"
                    for i in xrange(1,zLen):
                        if z2[i].find("RIKS")>-1: StepList[len(StepList)-1].ArcLen = True
                        if z2[i].find("VAR")>-1: StepList[len(StepList)-1].varTimeSteps = True 
                elif z2[0]=="*DYNAMIC":                 # key STEP DYNAMIC
                    ITyp2 = "stepDynamic"
                elif z2[0]=="*DAMPING":                 # key STEP DAMPING
                    ITyp2 = "stepDamping"
                    for i in xrange(1,zLen):
                        if z2[i].find("ALPHA")>-1: Dalph=float(z2[i].split("=")[1])
                    for i in xrange(1,zLen):
                        if z2[i].find("BETA")>-1:  Dbeta=float(z2[i].split("=")[1])
                elif z2[0]=="*BUCKLING":                # key STEP BUCKLE
                    ITyp2 = "stepBuckling"
                elif z2[0]=="*BOUNDARY":                # key STEP BOUNDARY
                    ITyp2 = "stepBoundary"
                    AmpLbl, AddVal = "Default", False
                    for i in xrange(1,zLen):
                        if z2[i].find("AMPLITUDE")>-1:     AmpLbl=z2[i].split("=")[1]
                        if z2[i].find("OP")>-1: 
                            if z2[i].split("=")[1]=="ADD": AddVal=True # add given value to final value of last step to get actual prescribed value
                elif z2[0]=="*DLOAD":                   # key STEP DLOAD
                    ITyp2 = "stepDLoad"
                    AmpLbl="Default"
                    for i in xrange(1,zLen):
                        if z2[i].find("AMPLITUDE")>-1: AmpLbl=z2[i].split("=")[1]
                elif z2[0]=="*CLOAD":                   # key STEP CLOAD
                    ITyp2 = "stepCLoad"
                    AmpLbl="Default"
                    for i in xrange(1,zLen):
                        if z2[i].find("AMPLITUDE")>-1: AmpLbl=z2[i].split("=")[1]
                elif z2[0]=="*TEMPERATURE":             # key STEP TEMPERATURE
                    ITyp2 = "stepTemp"
                    AmpLbl="Default"
                    for i in xrange(1,zLen):
                        if z2[i].find("AMPLITUDE")>-1: AmpLbl=z2[i].split("=")[1]
                    if AmpLbl not in StepList[len(StepList)-1].AmpDict: raise NameError ("Amplitude not known")
                elif z2[0]=="*PRESTRESS":              # key STEP PRESTRESS
                    ITyp2 = "stepPrestress"
                    AmpLbl="Default"
                    for i in xrange(1,zLen):
                        if z2[i].find("AMPLITUDE")>-1: AmpLbl=z2[i].split("=")[1]
                        if z2[i].find("TYPE")>-1: PreType=(z2[i].split("=")[1])
                        if z2[i].find("NAME")>-1: PreName=z2[i].split("=")[1]
                    if AmpLbl not in StepList[len(StepList)-1].AmpDict: raise NameError ("Amplitude not known")
                    PList = []
                    linecounter = 0
                elif z2[0]=="*EL FILE":                 # key STEP EL FILE
                    ITyp2= "stepElFile"
                    for i in xrange(1,zLen):
                        if z2[i].find("FREQUENCY")>-1: OutFr=float(z2[i].split("=")[1])
                elif z2[0]=="*NODE FILE":               # key STEP NODE FILE
                    ITyp2= "stepNoFile"
                    for i in xrange(1,zLen):
                        if z2[i].find("FREQUENCY")>-1: OutFr=float(z2[i].split("=")[1])
                elif z2[0]=="*SOLUTION TECHNIQUE":      # key STEP SOLUTION TECHNIQUE
                    for i in xrange(1,zLen):
                        if z2[i].find("TYPE")>-1: SolType=z2[i].split("=")[1]
                    if SolType=="QUASI-NEWTON": StepList[len(StepList)-1].SolType = 'BFGS'
                    elif SolType=="MODIFIED-NR": StepList[len(StepList)-1].SolType = 'MNR'
                elif z2[0]=="*AMPLITUDE":               # key STEP AMPLITUDE
                    ITyp2 = "stepAmp"
                    for i in xrange(1,zLen):
                        if z2[i].find("NAME")>-1: AmpName=z2[i].split("=")[1]
                    if AmpName in StepList[len(StepList)-1].AmpDict: raise NameError ("amplitude with same name already exists")
                elif z2[0]=="*END STEP":                # key END STEP
                    IType = ""
                    ITyp2 = ""
                    if StepList[len(StepList)-1].Damp and StepList[len(StepList)-1].RaBeta<>0. and StepList[len(StepList)-1].SolType<>'NR':
                        raise NameError ("ConFemInOut::ReadInputFile: Rayleigh-beta damping only compatible with Newton-Raphson") 
        ################################################################
        z1 = f1.readline()                              # read next line
        z2 = z1.split(",")
        zLen = len(z2)
        for i in xrange(zLen): z2[i] = z2[i].strip()
    if len(NodeList)<1 or len(ElList)<1 or len(MatList)<1 or len(StepList)<1: raise NameError ("Basic data missing")
    if Restart: return MatList, StepList
    else:       return NodeList, ElList, MatList, StepList

# restructuring nodal results for postprocessing
def ReadNodeResults( f1, f2, time ):
    NodeResultList = []
    z1 = f1.readline()
    IType = ""
    # single input line processing
    while z1<>"":
        z2 = z1.split()
        zLen = len(z2)
        for i in z2: i = i.strip()
        if zLen==1:
            if float(z2[0])==time: IType = "node"                       # does not control time!
            else:                  IType = ""
        elif IType=="node":
            # should be SH4
            if   zLen==14: NodeResultList += [NodeResult( int(z2[0]), float(z2[4]), float(z2[5]), float(z2[6]), float(z2[7]), float(z2[8]),          None,\
                                                                      float(z2[9]), float(z2[10]),float(z2[11]),float(z2[12]),float(z2[13]),         None )] 
            elif zLen==16: NodeResultList += [NodeResult( int(z2[0]), float(z2[4]), float(z2[5]), float(z2[6]), float(z2[7]), float(z2[8]),  float(z2[9]),\
                                                                      float(z2[10]),float(z2[11]),float(z2[12]),float(z2[13]),float(z2[14]),float(z2[15]) )]
            # all other elements
            else:          NodeResultList += [NodeResult( int(z2[0]), float(z2[4]), float(z2[5]), float(z2[6]), 0., 0., 0., \
                                                                      float(z2[7]), float(z2[8]), float(z2[9]), 0., 0., 0.)]
        z1 = f1.readline()
    #
    if f2<>None:
        z1 = f2.readline()
        z2 = z1.split()
        while z1<>"":
            nL, x, y, z, rr  = int(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), float(z2[-1])
            i = FindIndexByLabel(NodeResultList, nL)
            NodeResultList[i].ResNorm = rr
            z1 = f2.readline()
            z2 = z1.split()
    #    
    return NodeResultList
# restructuring element results for postprocessing -- used by PostFemShell
def ReadElemResults( f1, time, ElList, NodeList ):
    ElemResultList = []
    z1 = f1.readline()
    IType, elType, CCounter, RCounter, LL = "", "", 0, 0, []
    # single input line processing
    while z1<>"":
        z2 = z1.split()
        zLen = len(z2)
        for i in z2: i = i.strip()
        if zLen==2:
            if len(LL)>0:                            ElemResultList += [ElemResult( elType, elLabel, elIndex, nn, LL, LLClo, LLCup, LLR )]
            if z2[0]=='Time' and float(z2[1])==time: IType, LL = 'el', []
            else:                                    IType, elType, LL = "", "x", []
        else:
            if IType=='el':
                if z2[0]=='El':
                    if len(LL)>0:                                                   # to complete last entry 
                        ElemResultList += [ElemResult( elType, elLabel, elIndex, nn, LL, LLClo, LLCup, LLR, LLDlo, LLDup )]
                        CCounter, RCounter = 0, 0
                    LL, LLClo, LLCup, LLR, LLDlo, LLDup = [], [], [], [], [], []
                    elType  = z2[2]
                    elLabel = int(z2[1])
                    elIndex = FindIndexByLabel( ElList, elLabel)
                    if elType =='SH4': 
                        nn, nc, nr = [ int(z2[4]), int(z2[5]), int(z2[6]), int(z2[7])], int(z2[8]), int(z2[9])  # indices of nodes, number solid layers, number of reinforcement layers
                        CCpLow  = [ 1, 1+  nc, 1+2*nc, 1+3*nc]                      # to find the right lower (ip over height) solid-layers
                        CCpHigh = [nc,   2*nc,   3*nc,   4*nc]                      # to find the right upper (ip over height) solid-layers
                        for i in range(nr): LLR += [[]]
                    elif elType =='SH3':  nn, nc, nr = [ int(z2[4]), int(z2[5]), int(z2[6]) ], 0, 0 
                    elif elType =='C3D8': nn, nc, nr = [], 0, 0
                elif elType=='SH4':                                            
                    if z2[4] == 'C':                                                # concrete principal stresses per integration point   
                        CCounter += 1
                        if   CCounter in CCpLow:  
                            LLClo += [[float(z2[13]),float(z2[21])]]                # minimum and maximum principal stress for lowest ip over cross section
                            LLDlo += [[float(z2[25]),float(z2[26])]]                # damage and other data, e.g. viscous stress measure
                        elif CCounter in CCpHigh:                           
                            LLCup += [[float(z2[13]),float(z2[21])]]                # minimum and maximum principal stress for highest ip over cross section
                            LLDup += [[float(z2[25]),float(z2[26])]]                # damage and other data, e.g. viscous stress measure
                    elif z2[4]=='R':
                        v1, v2 = float(z2[11]), float(z2[19])
                        if abs(v1)>abs(v2): LLR[RCounter] += [[v1]]
                        else:               LLR[RCounter] += [[v2]]
                        RCounter += 1
                    elif z2[3]=='Z':
                        RCounter = 0                                                # integrated internal forces per integration point
                        LL += [[float(z2[0]),float(z2[1]),float(z2[2]), float(z2[4]),float(z2[5]),float(z2[6]),float(z2[7]),float(z2[8]),float(z2[8]),float(z2[10]),float(z2[11]) ]]
                elif elType =='SH3':
                    if z2[3]=='Z':
                        RCounter = 0
                        LL += [[float(z2[0]),float(z2[1]),float(z2[2]), float(z2[4]),float(z2[5]),float(z2[6]),float(z2[7]),float(z2[8]),float(z2[8]),float(z2[10]),float(z2[11]) ]]
                elif elType =='C3D8':
                    LL += [1.]                                                  # dummies
                elif elType =='T3D2':
                    LL += [1.]                                                  # dummies
                else: raise NameError("ConFemInOut::ReadElemResults: Element type not yet implemented")
        z1 = f1.readline()
    if len(LL)>0: ElemResultList += [ElemResult( elType, elLabel, elIndex, nn, LL, LLClo, LLCup, LLR, LLDlo, LLDup )]
    return ElemResultList

def WriteElemData( f2, f7, Time, ElList, NodeList, MatList, MaxType):
    # coefficients of linear approximation
    AI = array([[53./12.,-11./6.,-11./6.],[-11./6.,53./12.,-11./6.],[-11./6.,-11./6.,53./12.]])
    XX = array([[1./3.,1./3.,1./3.],[0.2,0.6,0.2],[0.2,0.2,0.6],[0.6,0.2,0.2]])
    f2.write('%s%8.4f\n'%('Time ',Time))
    MaxOut = False
    if f7<>None and len(MaxType)>0:                         # to extract maximum and minimum values of element data
        MaxOut = True 
        f7.write('%s%8.4f\n'%('Time ',Time))
        MaxVal, MinVal = {}, {}
        for i in MaxType: MaxVal[i], MinVal[i] = [0, 0.,0.,0., 0.], [0, 0.,0.,0., 0.] # element number, coordinates, value
    for i in xrange(len(ElList)):   # loop over all elements
        Elem = ElList[i]
        if Elem.Type=='SB3':        # only for SB3 element type
            for j in xrange(len(Elem.Inzi)): NodeList[Elem.Inzi[j]].mx, NodeList[Elem.Inzi[j]].my, NodeList[Elem.Inzi[j]].mxy, NodeList[Elem.Inzi[j]].c = 0,0,0,0   # why reset values mx my mxy ?
    for i in xrange(len(ElList)):  # loop over all elements
        xN, yN, zN = [], [], []
        Elem = ElList[i]
        f2.write('%s%7i%6s%40s'%('El',Elem.Label,Elem.Type, Elem.Set))
        for i in Elem.InzList: f2.write('%7i'%(i))
        if Elem.Type<>'SH4' and Elem.Type<>'SH3': f2.write('\n')
        for j in xrange(len(Elem.Inzi)):
            xN += [NodeList[Elem.Inzi[j]].XCo]
            yN += [NodeList[Elem.Inzi[j]].YCo]
            zN += [NodeList[Elem.Inzi[j]].ZCo]
        if Elem.Type=='SB3':                                # postprocessing kirchhoff slab for shear forces
            for j in xrange(len(Elem.Inzi)): NodeList[Elem.Inzi[j]].c += 1 #count assigned elements
            mx = [Elem.Data[0,3],Elem.Data[1,3],Elem.Data[2,3],Elem.Data[3,3]]
            aa = dot(AI, dot(transpose(XX),mx))             # coefficients of linear approximation
            for j in xrange(len(Elem.Inzi)): NodeList[Elem.Inzi[j]].mx += aa[j]#extrapolation to nodes
            mx_x = aa[0]*Elem.b1+aa[1]*Elem.b2+aa[2]*Elem.b3 # x derivative
            mx_y = aa[0]*Elem.c1+aa[1]*Elem.c2+aa[2]*Elem.c3 # y derivative
            my = [Elem.Data[0,4],Elem.Data[1,4],Elem.Data[2,4],Elem.Data[3,4]]
            aa = dot(AI, dot(transpose(XX),my))
            for j in xrange(len(Elem.Inzi)): NodeList[Elem.Inzi[j]].my += aa[j]#extrapolation to nodes
            my_x = aa[0]*Elem.b1+aa[1]*Elem.b2+aa[2]*Elem.b3
            my_y = aa[0]*Elem.c1+aa[1]*Elem.c2+aa[2]*Elem.c3
            mxy = [Elem.Data[0,5],Elem.Data[1,5],Elem.Data[2,5],Elem.Data[3,5]]
            aa = dot(AI, dot(transpose(XX),mxy))
            for j in xrange(len(Elem.Inzi)): NodeList[Elem.Inzi[j]].mxy += aa[j]#extrapolation to nodes
            mxy_x = aa[0]*Elem.b1+aa[1]*Elem.b2+aa[2]*Elem.b3
            mxy_y = aa[0]*Elem.c1+aa[1]*Elem.c2+aa[2]*Elem.c3
            for j in xrange(Elem.nIntL): Elem.Data[j,6] = -mx_x -mxy_y # x shear force
            for j in xrange(Elem.nIntL): Elem.Data[j,7] = -my_y -mxy_x # y shear force
        if Elem.Type=='SH4' or Elem.Type=='SH3':            # post processing continuum based shell
            IntF = {}                                       # same initialization for all elements ?!
            if Elem.ShellRCFlag: nRe, marker = Elem.Geom.shape[0]-2, 'C'     # number of reinforcement layers
            else:           nRe, marker = 0, 'U'
            if   Elem.nInt==2: nC = 4                       # number of concrete layers
            elif Elem.nInt==5: nC = 5
            else:              nC = 0
            f2.write('     %7i%7i\n'%(nC,nRe))
            Lis = Elem.Lists1()                             # indices for first integration points in base area
            if   Elem.Type=='SH4': Corr, CorM =0.5, 0.5     # to compensate for local isoparametric coordinate system and base integration 4*([-1..1] --> 8) --> 0.5
            elif Elem.Type=='SH3': Corr, CorM =3.0, 0.5     #                                                                              3*([ 0..1] --> 1) --> 3.0
            for j in Lis:                                   # loop over base plane
                Lis2, Lis3 = Elem.Lists2( nRe, j)           # integration point indices specific for base point
                r = SamplePoints[Elem.IntT,Elem.nInt-1,j][0]
                s = SamplePoints[Elem.IntT,Elem.nInt-1,j][1]
                xI = dot( Elem.FormX(r,s,0), xN)
                yI = dot( Elem.FormX(r,s,0), yN)
                zI = dot( Elem.FormX(r,s,0), zN)
                aa = dot( Elem.FormX(r,s,0), Elem.a)        # interpolated shell thickness from node thicknesses
                for k in ResultTypes[Elem.Set]: IntF[k] = 0. 
                for k in Lis2:                              # loop over cross section height
                    jj = j+k
                    if (k in Lis3): 
                        t  =         SamplePointsRCShell[Elem.Set,Elem.IntT,Elem.nInt-1,jj][2]
                        ff = Corr*aa*SampleWeightRCShell[Elem.Set,Elem.IntT,Elem.nInt-1,jj]  # 0.5 seems to compensate for SampleWeight local coordinates
                    else:           
                        t  =                SamplePoints[         Elem.IntT,Elem.nInt-1,jj][2]
                        ff = Corr*aa*       SampleWeight[         Elem.IntT,Elem.nInt-1,jj]  # 0.5 seems to compensate for SampleWeight local coordinates
                    ppp = Elem.Data[jj]                     #  content ruled through the material methods returning Data for Elem.dim==21
                    IntF['n_x']  = IntF['n_x']  + ff*ppp[0]  
                    IntF['n_y']  = IntF['n_y']  + ff*ppp[1]
                    IntF['q_y']  = IntF['q_y']  + ff*ppp[3]
                    IntF['q_x']  = IntF['q_x']  + ff*ppp[4]
                    IntF['n_xy'] = IntF['n_xy'] + ff*ppp[5]
                    IntF['m_x']  = IntF['m_x']  + CorM*aa*t*ff*ppp[0]
                    IntF['m_y']  = IntF['m_y']  + CorM*aa*t*ff*ppp[1]
                    IntF['m_xy'] = IntF['m_xy'] + CorM*aa*t*ff*ppp[5]
                    la_,v_ = eigh( [[ppp[0],ppp[5],ppp[4]] , [ppp[5],ppp[1],ppp[3]] , [ppp[4],ppp[3],ppp[2]]])       # principal values/eigenvalues -- in ascending order -- and eigenvectors of stress tensor
                    D =   ppp[-2]                           # for damage of isodam
                    kap = ppp[-1]                           # for equivalent damage strain of isodam
                    # case R reinforcement data
                    if (k in Lis3):  # R mode
                        f2.write('%10.4f%10.4f%10.4f%7.3f R%13.4e%13.4e%13.4e%13.4e%13.4e%13.4e                          '\
                                       %(xI,yI,zI,t,ppp[0],ppp[1],ppp[2],ppp[3],ppp[4],ppp[5])) 
                        f2.write('%9.2f  %7.3f%7.3f%7.3f%9.2f  %7.3f%7.3f%7.3f%9.2f  %7.3f%7.3f%7.3f'%(la_[0],v_[0,0],v_[1,0],v_[2,0],la_[1],v_[0,1],v_[1,1],v_[2,1],la_[2],v_[0,2],v_[1,2],v_[2,2]))
                        f2.write('\n')
                    elif marker=='C': 
                        f2.write('%10.4f%10.4f%10.4f%7.3f C%13.4e%13.4e%13.4e%13.4e%13.4e%13.4e%13.4e%13.4e'\
                                       %(xI,yI,zI,t,ppp[0+6],ppp[1+6],ppp[2+6],ppp[3+6],ppp[4+6],ppp[5+6],ppp[6+6],ppp[7+6])) # concrete data
                        f2.write('%9.2f  %7.3f%7.3f%7.3f%9.2f  %7.3f%7.3f%7.3f%9.2f  %7.3f%7.3f%7.3f'%(la_[0],v_[0,0],v_[1,0],v_[2,0],la_[1],v_[0,1],v_[1,1],v_[2,1],la_[2],v_[0,2],v_[1,2],v_[2,2]))
                        f2.write('%8.5f  %8.5f'%(D,kap))        # superfluous, already in ppp[6+6],ppp[7+6]
                        f2.write('\n')
                N, br, bs, JJ, JI, vv = Elem.Basics( r, s, 0.)   # vv: orientation of local coordinate system 1st aligned to first edge, 2nd column perp in-plane, 3rd column director, see also SH4:Basics
                # case Z
                f2.write('%10.4f%10.4f%10.4f        Z%13.4e%13.4e%13.4e%13.4e%13.4e%13.4e%13.4e%13.4e   %8.4f%8.4f%8.4f %8.4f%8.4f%8.4f\n'\
                                       %(xI,yI,zI,IntF['n_x'],IntF['n_y'],IntF['n_xy'],IntF['m_x'],IntF['m_y'],IntF['m_xy'],IntF['q_x'],IntF['q_y'],vv[0,0],vv[1,0],vv[2,0],vv[0,1],vv[1,1],vv[2,1])) # integrated internal forces
                if MaxOut:
                    for k in MaxType:
                        if IntF[k]>MaxVal[k][4]: MaxVal[k][0], MaxVal[k][1],MaxVal[k][2],MaxVal[k][3], MaxVal[k][4] = Elem.Label, xI,yI,zI, IntF[k]                
                        if IntF[k]<MinVal[k][4]: MinVal[k][0], MinVal[k][1],MinVal[k][2],MinVal[k][3], MinVal[k][4] = Elem.Label, xI,yI,zI, IntF[k]                
        elif Elem.Type=='T3D2':
            for j in xrange(Elem.nIntL):                    # write sample point coordinates and element data into file
                r = SamplePoints[Elem.IntT,Elem.nInt-1,j][0]
                xI = dot( Elem.FormX(r,0,0), xN)
                yI = dot( Elem.FormX(r,0,0), yN)
                zI = dot( Elem.FormX(r,0,0), zN)
                f2.write('%9.4f%9.4f%9.4f'%(xI,yI,zI))
                for k in xrange(Elem.Data.shape[1]): f2.write('%13.4e'%(Elem.Data[j,k]))
                f2.write('\n')
        elif Elem.Type=='C3D8':
            for j in xrange(Elem.nIntL):                    # write sample point coordinates and element data into file
                r = SamplePoints[Elem.IntT,Elem.nInt-1,j][0]
                s = SamplePoints[Elem.IntT,Elem.nInt-1,j][1]
                t = SamplePoints[Elem.IntT,Elem.nInt-1,j][2]
                xI = dot( Elem.FormX(r,s,t), xN)
                yI = dot( Elem.FormX(r,s,t), yN)
                zI = dot( Elem.FormX(r,s,t), zN)
                f2.write('%9.4f%9.4f%9.4f'%(xI,yI,zI))
                for k in xrange(Elem.Data.shape[1]): f2.write('%13.4e'%(Elem.Data[j,k]))
                f2.write('\n')
        else:
            for j in xrange(Elem.nIntL):                    # write sample point coordinates and element data into file
                r = SamplePoints[Elem.IntT,Elem.nInt-1,j][0]
                s = SamplePoints[Elem.IntT,Elem.nInt-1,j][1]
                t = SamplePoints[Elem.IntT,Elem.nInt-1,j][2]
                xI = dot( Elem.FormX(r,s,t), xN)
                yI = dot( Elem.FormX(r,s,t), yN)
                f2.write('%9.4f%9.4f'%(xI,yI))
                for k in xrange(Elem.Data.shape[1]): f2.write('%13.4e'%(Elem.Data[j,k]))
                f2.write('\n')
    if MaxOut: 
        for k in MaxType:
            f7.write('max%6s%8i%10.4f%10.4f%10.4f%13.4e\n'%(k,MaxVal[k][0],MaxVal[k][1],MaxVal[k][2],MaxVal[k][3],MaxVal[k][4]))
            f7.write('min%6s%8i%10.4f%10.4f%10.4f%13.4e\n'%(k,MinVal[k][0],MinVal[k][1],MinVal[k][2],MinVal[k][3],MinVal[k][4]))
    return 0

def WriteNodalData( f3, Time, NodeList, VecU, VecB):
    f3.write('Time %8.4f\n'%(Time))
    for i in xrange(len(NodeList)):
        Node = NodeList[i]
        iS = Node.GlobDofStart
        if len(Node.DofT)>0:
            f3.write('%5i%9.4f%9.4f%9.4f'%(Node.Label,Node.XCo,Node.YCo,Node.ZCo))
            for j in xrange(len(Node.DofT)): f3.write('%12.4e'%(VecU[iS+j]))   # DofT might be 5 or 6 for SH$, the latter for presumably folded edges / unfavorable director
            if Node.c>0:
                f3.write('  %12.4e%12.4e%12.4e  '%(Node.mx/Node.c,Node.my/Node.c,Node.mxy/Node.c)) # only for element type SB3
            for j in xrange(len(Node.DofT)): f3.write('%12.4e'%(VecB[iS+j]))
            f3.write('\n')
    return 0

def PostElem1D( f2 ):
    dataXe = []                                         # list for x per element
    dataYe = []                                         # list for y per element
    dataAe = []                                         # list for A per element
    dataXt = []                                         # list for x per time step
    dataYt = []                                         # list for y per time step
    dataAt = []                                         # list for A per time step
    dataSt = []                                         # list for sets per time step
    dataX = []                                          # list for all x
    dataY = []                                          # list for all y
    dataA = []                                          # list for all A
    dataS = []                                          # list for all sets
    Times = []                                          # list for time values
    z1 = f2.readline()
    z2 = z1.split()
    def Rotate( dataXe, dataYe):
        n = len(dataXe)
        if n>1:
            LL = ((dataXe[n-1]-dataXe[0])**2+(dataYe[n-1]-dataYe[0])**2)**0.5
            cosA = (dataXe[n-1]-dataXe[0])/LL
            sinA = (dataYe[n-1]-dataYe[0])/LL
            return cosA, sinA
        else: return 1, 0
    Flag_ = False
    while z1<>"":
        if z2[0]=="Time":                               # time key found
            ElSetI = 0
            ElSets = {}                                 # dictionary: element set label -> index
            ElSetElType = {}                            # dictionary: element set label -> element type
            Times += [float(z2[1])]
            if len(dataXe)>0:
                dataXt += [dataXe]                      # append last element
                dataYt += [dataYe]                      # append last element
                dataAt += [dataAe]
                dataX += [dataXt]                       # append to list of all x
                dataY += [dataYt]                       # append to list of all y
                dataA += [dataAt]
                dataS += [dataSt]
                dataXt, dataYt, dataAt, dataXe, dataYe, dataAe = [], [], [], [], [], [] # reset lists for elements
        elif z2[0]=="El":                               # element key found
            ElType = z2[2]
            if ElType in ['CPS4','CPS4R','CPS3','CPE4','CPE4R','CPE3','SB3','SH4', 'SH3','T2D2','T3D2']: Flag = False
            else:
                Flag_= True
                Flag = True
                ElSet  = z2[3]
                if ElSet not in ElSets:                 # new elset
                    ElSetI = ElSetI+1
                    ElSets[ElSet] = ElSetI              # index for new elset
                    ElSetElType[ElSet] = ElType         # element type of new elset
                dataSt += [ElSetI]                      # assign elset index to current data
                if len(dataXe)>0:
                    dataXt += [dataXe]                  # append to list of elements per time step
                    dataYt += [dataYe]                  # append to list of elements per time step
                    dataAt += [dataAe]
                    dataXe = []                         # reset list for element
                    dataYe = []                         # reset list for element
                    dataAe = []
        elif Flag:                                      # data line
            dataXe += [float(z2[0])]                    # append x-coordinate to element list
            dataYe += [float(z2[1])]                    # append y-coordinate to element list
            dataAe += [[float(z2[pos]) for pos in xrange(2,len(z2))]] #
        z1 = f2.readline()
        z2 = z1.split()
    dataXt += [dataXe]                                  # append last element
    dataYt += [dataYe]                                  # append last element
    dataAt += [dataAe]
    dataX += [dataXt]                                   #
    dataY += [dataYt]                                   #
    dataA += [dataAt]
    dataS += [dataSt]
    if len(dataX)==0 or not Flag_: return 0

    # plot whole stuff
    scal = [ 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
#    if raw_input('Separate element sets for plot? (y/n):')=='y':
    if len(ElSets)>0:
        SFlag=True
        nSets = len(list(ElSets.values()))
    else:
        SFlag=False
        nSets = 1
    for K in xrange(nSets):                             # loop over element sets
        sI =  list(ElSets.values())[K]
        ElS = list(ElSets.keys())[K]
        titleList = ResultTypes[ElS]
        for I in xrange(len(titleList)):                # loop over result types
            Ymin, Ymax = 0,0
            if titleList[I]==None: continue
            figure()                                # new figure
            for i in xrange(len(Times)):                # loop over time steps
                X_, Y_ = [],[]
                for j in xrange(len(dataA[i])):         # loop over fe elements
                    if dataS[i][j]==sI or not SFlag:
                        neD = len(dataA[i][j])          # number of data per element
                        cosA, sinA = Rotate( dataX[i][j], dataY[i][j])
                        X = [ dataX[i][j][k] - sinA*(scal[I]*dataA[i][j][k][I]) for k in xrange(neD)]
                        X_ += [ dataX[i][j][k] - sinA*(scal[I]*dataA[i][j][k][I]) for k in xrange(neD)]
                        if fabs(sinA)>ZeroD:
                            plot( dataX[i][j], dataY[i][j], '-k')
                            Y = [ dataY[i][j][k] + cosA*(scal[I]*dataA[i][j][k][I]) for k in xrange(neD)]
                            Y_+=[ dataY[i][j][k] + cosA*(scal[I]*dataA[i][j][k][I]) for k in xrange(neD)]
                        else: 
                            Y = [                     (scal[I]*dataA[i][j][k][I]) for k in xrange(neD)]
                            Y_+=[                     (scal[I]*dataA[i][j][k][I]) for k in xrange(neD)]
                        Ymin=min(Ymin,min(Y))
                        Ymax=max(Ymax,max(Y))
                        plot( X, Y, '-', color=Colors[i % len(Colors)],linewidth=LiWiCrv)
                        plot( X, Y, 'o', color=Colors[i % len(Colors)])
                        plot( X_, Y_, '-', color=Colors[i % len(Colors)],linewidth=LiWiCrvThin)
#                        print titleList[I], X, Y
#                        plot( X, Y, '-', color=Colors[i % len(Colors)])
            ylim(1.1*Ymin,1.1*Ymax)
#            ylim(-1.20,0.2)
            yticks(fontsize=FonSizAx) 
            xticks(fontsize=FonSizAx) 
            if SFlag: title(ElS+': '+titleList[I],fontsize=FonSizTi)
            else:     title(titleList[I],fontsize=FonSizTi)
            grid()
    return 0

def PostScales( ElList, NodeList, f2 ):
    def Diff(stressX, stressY, stressXY):
        diffX,diffY,diffXY = 0,0,0
        maxX, maxY, maxXY, minX, minY, minXY = 0,0,0,0,0,0
        if len(stressX)>0: 
            maxX, minX = max(stressX), min(stressX)
            diffX=maxX-minX
        if len(stressY)>0:
            maxY, minY = max(stressY), min(stressY)
            diffY=maxY-minY
        if len(stressXY)>0:
            maxXY, minXY = max(stressXY), min(stressXY)
            diffXY=maxXY-minXY
        diff = max(diffX,diffY,diffXY,fabs(maxX),fabs(maxY),fabs(maxXY),fabs(minX),fabs(minY),fabs(minXY))
        if diff>ZeroD: return diff
        else: return 1
    def ScaleF(stressX, stressY, stressXY, stressX1, stressY1, stressXY1, CharLen):
        ScFactor, ScFactor1 = 1.0, 1.0 # 3 , 3,  1.5, 1.5                                     #
        diff = Diff(stressX, stressY, stressXY)
        sc = min(CharLen)*ScFactor/diff
        diff1 = Diff(stressX1, stressY1, stressXY1)
        sc1 = min(CharLen)*ScFactor1/diff1
        return sc, sc1
    ESet, Tim = ' ', ' '
    stressX, stressY, stressXY, stressX1, stressY1, stressXY1 =[0], [0], [0], [0], [0], [0] 
    Sc_, Sc1_ = {}, {}
    CharLen = []                                        # characteristic length
    z1 = f2.readline()
    z2 = None
    while z1<>"":
        z2 = z1.split()
        if z2[0]=="Time": Time = z2[1]
        elif z2[0]=="El":                               # element key found
            ElType = z2[2]                              # key for element type
            ElSet = z2[3]                               # key for element set
            Elem = ElList[FindIndexByLabel( ElList, z2[1] )]
            CharLen.append(Elem.Lch_)
            if ElSet<>ESet or Time<>Tim:                # new time or element set found
                sc, sc1 = ScaleF(stressX, stressY, stressXY, stressX1, stressY1, stressXY1, CharLen)
                Sc_[ESet+Tim] = sc
                Sc1_[ESet+Tim] = sc1
#                stressX, stressY, stressXY, stressX1, stressY1, stressXY1 = [0], [0], [0], [0], [0], [0] 
                stressX, stressY, stressXY, stressX1, stressY1, stressXY1 = [], [], [], [], [], [] 
                ESet, Tim = ElSet, Time
        else:                                           # data found
            if ElType=='CPE4' or ElType=='CPS4' or ElType=='CPS4R' or ElType=='CPE4R' or ElType=='SB3' or ElType=='CPE3'  or ElType=='CPS3':
                stressX.append(float(z2[5]))
                stressY.append(float(z2[6]))
                stressXY.append(float(z2[7]))
            elif ElType=='T1D2' or ElType=='T2D2' or ElType=='T3D2' or ElType=='S1D2':
                stressX.append(float(z2[3]))
            elif ElType=='S2D6':
                stressX.append(float(z2[5]))       #  ??????????????
            elif ElType=='B23' or ElType=='B23E' or ElType=='B21' or ElType=='B21E':  
                stressX.append(float(z2[5]))
            elif ElType=='SH4' or ElType=='SH3':
                if z2[3]=='Z': 
                    stressX.append(float(z2[4]))
                    stressY.append(float(z2[5]))
                    stressXY.append(float(z2[6]))
                    stressX1.append(float(z2[7]))
                    stressY1.append(float(z2[8]))
                    stressXY1.append(float(z2[9]))
            elif ElType=='C3D8':
                stressX.append(float(z2[3]))
                stressY.append(float(z2[4]))
                stressXY.append(float(z2[5]))
                stressX1.append(float(z2[6]))
                stressY1.append(float(z2[7]))
                stressXY1.append(float(z2[8]))
            else:
                print 'XXX', ElType 
                raise NameError ("PostScales: unknown element type")
        z1 = f2.readline()
    if z2<>None:
        sc, sc1 = ScaleF(stressX, stressY, stressXY, stressX1, stressY1, stressXY1, CharLen)
        Sc_[ESet+Tim] = sc
        Sc1_[ESet+Tim] = sc1
    return Sc_, Sc1_

def PostElem2D( ElList, NodeList, MatList, f2, Scales_, Scales1_, SE, SP ):
    def IniP(Text, Time, ElSet, proj):
        if proj=='3d': P0 = figure().add_subplot(111,title=Text+'Time '+Time+', Element Set '+ElSet, projection=proj)
        else:
            P0 = figure().add_subplot(111,title=Text+'Time '+Time+', Element Set '+ElSet)
            P0.axis('equal')
            P0.grid()
        yticks(fontsize=FonSizAx) 
        xticks(fontsize=FonSizAx) 
        return True, P0
    def RotLocal(Local1x, Local1y, pp):                 # coordinates of 1st local axis
        phi = -arcsin(Local1y/sqrt(Local1x**2+Local1y**2))
        if Local1x<0.: phi = pi-phi
        pp1 =  pp[1]*cos(phi) + pp[2]*sin(phi)          # plane transformation of vector into new coordinate system
        pp2 = -pp[1]*sin(phi) + pp[2]*cos(phi)
        return pp1, pp2
    def PloPrin( scaleP, XX, YY, P0, ELaMax, ELaMin, Xmax, Ymax, Smax, S2max, Xmin, Ymin, Smin, S2min, ElLabel):
        SS = pp[0]
        if SS>=0: P0.plot([XX-SS*scaleP*pp[1],XX+SS*scaleP*pp[1]],[YY-SS*scaleP*pp[2],YY+SS*scaleP*pp[2]],'r-')
        else:     P0.plot([XX-SS*scaleP*pp[1],XX+SS*scaleP*pp[1]],[YY-SS*scaleP*pp[2],YY+SS*scaleP*pp[2]],'g-')
        if SS>Smax: ELaMax, Xmax, Ymax, Smax, S2max = ElLabel, XX, YY, SS, pp[3]
        if SS<Smin: ELaMin, Xmin, Ymin, Smin, S2min = ElLabel, XX, YY, SS, pp[3]
        SS = pp[3]
        if SS>=0: P0.plot([XX-SS*scaleP*pp[2],XX+SS*scaleP*pp[2]],[YY+SS*scaleP*pp[1],YY-SS*scaleP*pp[1]],'r-')
        else:     P0.plot([XX-SS*scaleP*pp[2],XX+SS*scaleP*pp[2]],[YY+SS*scaleP*pp[1],YY-SS*scaleP*pp[1]],'g-')
        if SS>Smax: ELaMax, Xmax, Ymax, Smax, S2max = ElLabel, XX, YY, SS, pp[0]
        if SS<Smin: ELaMin, Xmin, Ymin, Smin, S2min = ElLabel, XX, YY, SS, pp[0]
        return ELaMax, ELaMin, Xmax, Ymax, Smax, S2max, Xmin, Ymin, Smin, S2min
    def Annot( ):
        if Xmax<>None:
            P0.annotate('max s1 '+str(Smax)[0:7]+' s2 '+str(S2max)[0:7]+' '+'%6i'%(Elmax),xy=(Xmax, Ymax),xycoords='data',xytext=(0.40,0.87),textcoords='figure fraction',arrowprops=dict(arrowstyle="->"),fontsize='medium')
            if ElType=='SH4': P1.annotate('max s1 '+'%7.4f'%(Smax_)+' s2 '+'%7.4f'%(S2max_),xy=(Xmax_, Ymax_),xycoords='data',xytext=(0.50,0.87),textcoords='figure fraction',arrowprops=dict(arrowstyle="->"),fontsize='large')
        if Xmin<>None:
            P0.annotate('min s1 '+str(Smin)[0:7]+' s2 '+str(S2min)[0:7]+' '+'%6i'%(Elmin),xy=(Xmin, Ymin),xycoords='data',xytext=(0.40,0.84),textcoords='figure fraction',arrowprops=dict(arrowstyle="->"),fontsize='medium')
            if ElType=='SH4': P1.annotate('min s1 '+'%7.4f'%(Smin_)+' s2 '+'%7.4f'%(S2min_),xy=(Xmin_, Ymin_),xycoords='data',xytext=(0.50,0.84),textcoords='figure fraction',arrowprops=dict(arrowstyle="->"),fontsize='large')
        
    ESet, Tim, Xmax, Ymax, Xmin, Ymin, Elmax, Elmin = None, None, None, None, None, None, None, None
    Xmax_, Ymax_, Xmin_, Ymin_ = None, None, None, None
    ESet, Tim = ' ', ' '
    Flag = False                                                        # Flag for elasticLT and flavors
    z1 = f2.readline()
    z2 = z1.split()
    while z1<>"":
        if z2[0]=="Time":                                               # time key found
            Time = z2[1]
            print "process plot for time ", Time
        elif z2[0]=="El":                                               # element key found
            ElType = z2[2]                                              # key for element type
            if ElType in ['CPE4','CPE4R','CPE3','CPS4','CPS4R','CPS3','SB3','SH4','SH3', 'T2D2']:
                ElSet = z2[3]                                           # key for element set
                Elem   = ElList[FindIndexByLabel(ElList, int(z2[1]))]
                if isinstance( MatList[Elem.MatN], ElasticLT) or Elem.ShellRCFlag: Flag=True
                if isinstance( MatList[Elem.MatN], IsoDamage): MatType='IsoDamage'
                else:                                          MatType=''    
                if ElType=='CPS4' or ElType=='CPS4R' or ElType=='CPS3' or ElType=='CPE4' or ElType=='CPE4R' or ElType=='CPE3' or ElType=='SB3' or ElType=='SH4' or ElType=='SH3':
                    if ElSet<>ESet or Time<>Tim:                        # new time or element set found
                        Annot()
                        IniFl, P0 = IniP('PostElem2D P0: ', Time, ElSet, None)                                          # normal forces
                        if ElType=='SB3' or ElType=='SH4' or ElType=='SH3': IniFl, P1 = IniP('PostElem2D P1: ', Time, ElSet, None)       # bending moments
                        if ElType=='SH4' and Flag:                          IniFl, P2 = IniP('PostElem2D P2: ', Time, ElSet, '3d')       # strains in 3d
                        Smax, Smin, Smax_, Smin_, S2max, S2min, S2max_, S2min_ = -9999, 9999, -9999, 9999, None, None, None, None
                        ScaleKey = ElSet+Time
                        ESet, Tim = ElSet, Time
                        scaleP = SE*Scales_[ScaleKey]
                        if ElType=='SH4' or ElType=='SH3': scaleP1 = SE*Scales1_[ScaleKey]
                    if ElType=='CPS4' or ElType=='CPS4R' or ElType=='CPE4' or ElType=='CPE4R' or ElType=='SH4':
                        xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[3]].XCo,NodeList[Elem.Inzi[0]].XCo]
                        yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[3]].YCo,NodeList[Elem.Inzi[0]].YCo]
                    elif ElType=='CPS3' or ElType=='CPE3' or ElType=='SB3' or ElType=='SH3':
                        xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[0]].XCo]
                        yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[0]].YCo]
                    P0.plot(xN,yN, 'b--')
                elif ElType=='T2D2':
                    if ElSet<>ESet or Time<>Tim:                        # new time or element set found
                        IniFl, P4 = IniP( '', Time, ElSet, None)        # bar forces
                        ESet, Tim = ElSet, Time
                    xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo]
                    yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo]
                    P4.plot(xN,yN, 'ro')
                    P4.plot(xN,yN, 'b--')
                if ElType=='SB3' or ElType=='SH4' or ElType=='SH3': P1.plot(xN,yN, 'b--')
        else:                                                           # data found
            XX = float(z2[0])                                           # x-coordinate
            YY = float(z2[1])                                           # y-coordinate
            if ElType=='CPS4' or ElType=='CPS4R' or ElType=='CPS3' or ElType=='CPE4' or ElType=='CPE4R' or ElType=='CPE3' or ElType=='SB3':
#                if len(z2)==10: pp = PrinC( float(z2[6]), float(z2[7]), float(z2[9])) # principal stresses Isodam_only; shure? there might other mat types with ouptut length 10
                if MatType=='IsoDamage': pp = PrinC( float(z2[6]), float(z2[7]), float(z2[9])) 
                else:                    pp = PrinC( float(z2[5]), float(z2[6]), float(z2[7]))
                Elmax,Elmin,Xmax,Ymax,Smax,S2max,Xmin,Ymin,Smin,S2min=PloPrin(scaleP,XX,YY,P0,Elmax,Elmin,Xmax,Ymax,Smax,S2max,Xmin,Ymin,Smin,S2min,Elem.Label)
            elif ElType=='SH4' or ElType=='SH3':
                if z2[3]=='Z':
                    xyLength = float(z2[12])**2+float(z2[13])**2        # length in xy-plane
                    pp = PrinC( float(z2[4]), float(z2[5]), float(z2[6]))
                    if xyLength>ZeroD: pp[1], pp[2] = RotLocal( float(z2[12]), float(z2[13]), pp) # transformation of local axes into global
                    Elmax,Elmin,Xmax,Ymax,Smax,S2max,Xmin,Ymin,Smin,S2min=PloPrin(scaleP,XX,YY,P0,Elmax,Elmin,Xmax,Ymax,Smax,S2max,Xmin,Ymin,Smin,S2min,Elem.Label) # principal membrane forces
                    pp = PrinC( float(z2[7]), float(z2[8]), float(z2[9]))
                    if xyLength>ZeroD: pp[1], pp[2] = RotLocal( float(z2[12]), float(z2[13]), pp) # transformation of local axes into global
                    Elmax,Elmin,Xmax_,Ymax_,Smax_,S2max_,Xmin_,Ymin_,Smin_,S2min_=PloPrin(scaleP1,XX,YY,P1,Elmax,Elmin,Xmax_,Ymax_,Smax_,S2max_,Xmin_,Ymin_,Smin_,S2min_,Elem.Label) # principal moments
                elif Flag:                              # plot data of every layer in case of ELASTICLT
                    ZZ = float(z2[3])
                    if z2[4]=='R':
                        col='blue'
 #                   elif ZZ<-0.1 or ZZ>0.1:
                    else: # mode C
                        scaleX = SP
                        pp = PrinC( float(z2[5]), float(z2[6]), 0.5*float(z2[7])) # principal strains concrete
                        if float(z2[10])>0: linestyle='--'# crack in 1(?)-direction
                        else:               linestyle='-'
                        SS = pp[0]
                        if SS>=0: P2.plot([XX-SS*scaleX*pp[1],XX+SS*scaleX*pp[1]],[YY-SS*scaleX*pp[2],YY+SS*scaleX*pp[2]],[ZZ,ZZ],'r'+linestyle)
                        else:     P2.plot([XX-SS*scaleX*pp[1],XX+SS*scaleX*pp[1]],[YY-SS*scaleX*pp[2],YY+SS*scaleX*pp[2]],[ZZ,ZZ],'g'+linestyle)
                        if float(z2[11])>0: linestyle='--'# crack in 1(?)-direction
                        else:               linestyle='-'
                        SS = pp[3]
                        if SS>=0: P2.plot([XX-SS*scaleX*pp[2],XX+SS*scaleX*pp[2]],[YY+SS*scaleX*pp[1],YY-SS*scaleX*pp[1]],[ZZ,ZZ],'r'+linestyle)
                        else:     P2.plot([XX-SS*scaleX*pp[2],XX+SS*scaleX*pp[2]],[YY+SS*scaleX*pp[1],YY-SS*scaleX*pp[1]],[ZZ,ZZ],'g'+linestyle)
                        col='red'
                    P2.plot([XX],[YY],[ZZ],'o',color=col)  # 'o' 
            elif ElType=='T2D2':
                FF = float(z2[3])
                Angle = arcsin(Elem.Geom[1,2]/sqrt(Elem.Geom[1,3]))*180/3.141593 #asin(Elem.Trans[0,1])*180/3.141593
                if FF>0: P4.text(XX,YY,format(FF,".2f"),ha='center',va='bottom',rotation=Angle,color='red',fontsize=FonSizAx)
                else: P4.text(XX,YY,format(FF,".2f"),ha='center',va='bottom',rotation=Angle,color='green',fontsize=FonSizAx)
            if ElType=='SB3':
                xS = (NodeList[Elem.Inzi[0]].XCo+NodeList[Elem.Inzi[1]].XCo+NodeList[Elem.Inzi[2]].XCo)/3.
                yS = (NodeList[Elem.Inzi[0]].YCo+NodeList[Elem.Inzi[1]].YCo+NodeList[Elem.Inzi[2]].YCo)/3.
                qx = float(z2[8])
                qy = float(z2[9])
                scaleQ = 10.
                if qx>0: P1.plot([xS,xS+scaleQ*qx],[yS,yS],'-',color='red')
                else:    P1.plot([xS,xS+scaleQ*qx],[yS,yS],'-',color='green')
                if qy>0: P1.plot([xS,xS],[yS,yS+scaleQ*qy],'-',color='red')
                else:    P1.plot([xS,xS],[yS,yS+scaleQ*qy],'-',color='green')
                P1.plot([xS],[yS],'o',color='red')
        z1 = f2.readline()
        z2 = z1.split()
    Annot()
    return 0

def PostElem3D( ElList, NodeList, VecU, SN, mod_Z_limit):
    XList, YList, ZList, DisElemList, sig1POSMElemList, sig2POSMElemList, sig1NEGMElemList, sig2NEGMElemList, nxLi,nyLi,nxyLi,mxLi,myLi,mxyLi,qxLi,qyLi, aaLi = [], [], [], [], [], [], [], [],[],[],[],[],[],[],[],[],[]  # ts
    for i in NodeList:
        XList += [i.XCo]
        YList += [i.YCo]
        ZList += [i.ZCo]
    mX, mY, mZ = [np.min(XList),np.max(XList),np.mean(XList)], [np.min(YList),np.max(YList),np.mean(YList)], [np.min(ZList),np.max(ZList),np.mean(ZList)]
    for elem in ElList:
        DisList = []
        for o in elem.Inzi:
            node = NodeList[o]
            DofInd = node.GlobDofStart
            if   elem.Type=='SH4': DisList += [sqrt(VecU[DofInd]**2+VecU[DofInd+1]**2+VecU[DofInd+2]**2)]
            elif elem.Type=='SB3': DisList += [VecU[DofInd]]       
        DisElemList += [np.mean(DisList)]
    plotDeformFigure('deform', DisElemList,VecU,ElList,NodeList,' deformed mesh', mX, mY, mZ)
    ############################################################################################################################### # ts
    for elem in ElList: # for SH4 only
        # comes mainly from WriteElementData
        offset, nRe = 0, 0                                                                                                              # ts
        if elem.ShellRCFlag: nRe = elem.Geom.shape[0]-2     # number of reinforcement layers
        else:           nRe = 0
        Lis = elem.Lists1()                             # indices for first integration points in base area      
        LisLen = len(Lis)                                                                       # ts
        nx, ny, nxy, qy, qx, mx, my, mxy, sig1POSSumme, sig2POSSumme, sig1NEGSumme, sig2NEGSumme, aa_ = 0., 0., 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0.
        for j in Lis:                                                                                                                   # ts
            r = SamplePoints[elem.IntT,elem.nInt-1,j][0]                                                                                # ts
            s = SamplePoints[elem.IntT,elem.nInt-1,j][1]                                                                                # ts
            aa = dot( elem.FormX(r,s,0), elem.a)      # interpolated shell thickness from node thicknesses
            Lis2, Lis3 = elem.Lists2( nRe, j)           # integration point indices specific for base point
            for k in Lis2:                                                                                                              # ts
                jj = j+k                                                                                                                # ts
                if (k in Lis3): 
                    t  =        SamplePointsRCShell[elem.Set,elem.IntT,elem.nInt-1,jj][2]
                    ff = 0.5*aa*SampleWeightRCShell[elem.Set,elem.IntT,elem.nInt-1,jj]  # 0.5 seems to compensate for SampleWeight local coordinates
                else:           
                    t  =               SamplePoints[         elem.IntT,elem.nInt-1,jj][2]
                    ff = 0.5*aa*       SampleWeight[         elem.IntT,elem.nInt-1,jj]  # 0.5 seems to compensate for SampleWeight local coordinates
                ppp = elem.Data[jj]
                nx = nx + ff*ppp[0+offset]
                ny = ny + ff*ppp[1+offset]
                qy = qy + ff*ppp[3+offset]
                qx = qx + ff*ppp[4+offset]
                nxy= nxy+ ff*ppp[5+offset]
                mx = mx + 0.5*aa*t*ff*ppp[0+offset]
                my = my + 0.5*aa*t*ff*ppp[1+offset]
                mxy= mxy+ 0.5*aa*t*ff*ppp[5+offset]
                sig_x  = ppp[0+offset]                                                                                         # ts
                sig_y  = ppp[1+offset]                                                                                         # ts
                sig_xy = ppp[5+offset]                                                                                         # ts
                sigma1,sigma2 = mainStressConverter(sig_x, sig_y, sig_xy)#, f_mainStress)                                                # ts
                if k == 0:                                                                                                              # ts
                    sig1NEGSumme = sig1NEGSumme + sigma1                                                                                # ts 
                    sig2NEGSumme = sig2NEGSumme + sigma2                                                                                # ts 
                elif k == 3:                                                                                                            # ts
                    sig1POSSumme = sig1POSSumme + sigma1                                                                                # ts
                    sig2POSSumme = sig2POSSumme + sigma2                                                                                # ts 
            aa_ = aa_ + aa
        sig1NEGMElemList.append(sig1NEGSumme/LisLen)
        sig2NEGMElemList.append(sig2NEGSumme/LisLen)                                                                                               # ts sig1M = sig1Summe/(len(Lis))                                                                                                    # ts
        sig1POSMElemList.append(sig1POSSumme/LisLen)                                                                                               # ts 
        sig2POSMElemList.append(sig2POSSumme/LisLen)
        nxLi.append( nx/LisLen)
        nyLi.append( ny/LisLen)
        nxyLi.append(nxy/LisLen)
        mxLi.append( mx/LisLen)
        myLi.append( my/LisLen)
        mxyLi.append(mxy/LisLen)
        qxLi.append( qx/LisLen)
        qyLi.append( qy/LisLen)
        aaLi.append( aa_/LisLen)
    ############################################################################################################################### # ts                                 # ts
#        plotMainStressFigures2(sig1NEGMElemList,'S1 SNEG',XList,YList,ZList,ElList,NodeList)
#        plotMainStressFigures2(sig2NEGMElemList,'S2 SNEG',XList,YList,ZList,ElList,NodeList)
#        plotMainStressFigures2(sig1POSMElemList,'S1 SPOS',XList,YList,ZList,ElList,NodeList)
#        plotMainStressFigures2(sig2POSMElemList,'S2 SPOS',XList,YList,ZList,ElList,NodeList)
    plotMainStressFigures2(nxLi, 'n_x', XList,YList,ZList,ElList,NodeList, mod_Z_limit)
    plotMainStressFigures2(nyLi, 'n_y', XList,YList,ZList,ElList,NodeList, mod_Z_limit)
    plotMainStressFigures2(nxyLi,'n_xy',XList,YList,ZList,ElList,NodeList, mod_Z_limit)
    plotMainStressFigures2(mxLi, 'm_x', XList,YList,ZList,ElList,NodeList, mod_Z_limit)
    plotMainStressFigures2(myLi, 'm_y', XList,YList,ZList,ElList,NodeList, mod_Z_limit)
    plotMainStressFigures2(mxyLi,'m_xy',XList,YList,ZList,ElList,NodeList, mod_Z_limit)
    plotMainStressFigures2(qxLi, 'q_x', XList,YList,ZList,ElList,NodeList, mod_Z_limit)
    plotMainStressFigures2(qyLi, 'q_y', XList,YList,ZList,ElList,NodeList, mod_Z_limit)
    #plotMainStressFigures2(aaLi, 'thickness', XList,YList,ZList,ElList,NodeList, mod_Z_limit)
    #
    return 0

def PostNode( ElList, NodeList, VecU, SN, mod_Z_limit, PE3DFlag):
    d2, d3 = False, False
    for i in xrange(len(ElList)):
        Elem = ElList[i]
        if Elem.Type not in ['']: d2 = True #['SB3','CPE4','T1D2','S1D2','CPE3','B23E']: d2 = True
        if Elem.Type in ['SH4','SB3']: d3 = True
    if d2:
        LX, LY, LZ, LU = [], [], [], []
        for i in xrange(len(NodeList)):
            LX.append(NodeList[i].XCo)
            LY.append(NodeList[i].YCo)
            LZ.append(NodeList[i].ZCo)
        for i in xrange(len(VecU)): LU.append(fabs(VecU[i]))
        Xmin, Xmax, Ymin, Ymax, Zmin, Zmax = min(LX), max(LX), min(LY), max(LY), min(LZ), max(LZ)
        D = max(Xmax-Xmin,Ymax-Ymin,Zmax-Zmin)
        U = max(LU)
        if U<ZeroD: raise NameError("ConFemInOut::PostNode: Zero Displacements")
        scale = SN*0.5*D/U                                     # estimation for scaling factor
        figure()
        ElSets = {}
        colI = 0                                                # color index
        for i in xrange(len(ElList)):
            xS, yS, xP, yP, xN, yN = [], [], [], [], [], []
            Elem = ElList[i]
            Set = Elem.Set
            if Set not in ElSets:
                ElSets[Set] = colI
                colI = colI+1
            if Elem.Type=='SB3':
    #            d3=True
                xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[0]].YCo]
                xS_ = (NodeList[Elem.Inzi[0]].XCo+NodeList[Elem.Inzi[1]].XCo+NodeList[Elem.Inzi[2]].XCo)/3.
                yS_ = (NodeList[Elem.Inzi[0]].YCo+NodeList[Elem.Inzi[1]].YCo+NodeList[Elem.Inzi[2]].YCo)/3.
                eL = "%i/%i"%(i,Elem.Label)
                text(xS_,yS_,eL,ha='center',va='center')
            elif Elem.Type=='CPE4' or Elem.Type=='CPS4':
                s = scale
                xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[3]].XCo,NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[3]].YCo,NodeList[Elem.Inzi[0]].YCo]
                xP = [NodeList[Elem.Inzi[0]].XCo+scale*VecU[Elem.DofI[0,0]],NodeList[Elem.Inzi[1]].XCo+scale*VecU[Elem.DofI[1,0]],NodeList[Elem.Inzi[2]].XCo+scale*VecU[Elem.DofI[2,0]],NodeList[Elem.Inzi[3]].XCo+scale*VecU[Elem.DofI[3,0]],NodeList[Elem.Inzi[0]].XCo+scale*VecU[Elem.DofI[0,0]]]
                yP = [NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,1]],NodeList[Elem.Inzi[1]].YCo+scale*VecU[Elem.DofI[1,1]],NodeList[Elem.Inzi[2]].YCo+scale*VecU[Elem.DofI[2,1]],NodeList[Elem.Inzi[3]].YCo+scale*VecU[Elem.DofI[3,1]],NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,1]]]
            elif Elem.Type=='CPE3' or Elem.Type=='CPS3':
                s = scale
                xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[0]].YCo]
                xP = [NodeList[Elem.Inzi[0]].XCo+scale*VecU[Elem.DofI[0,0]],NodeList[Elem.Inzi[1]].XCo+scale*VecU[Elem.DofI[1,0]],NodeList[Elem.Inzi[2]].XCo+scale*VecU[Elem.DofI[2,0]],NodeList[Elem.Inzi[0]].XCo+scale*VecU[Elem.DofI[0,0]]]
                yP = [NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,1]],NodeList[Elem.Inzi[1]].YCo+scale*VecU[Elem.DofI[1,1]],NodeList[Elem.Inzi[2]].YCo+scale*VecU[Elem.DofI[2,1]],NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,1]]]
            elif Elem.Type=='SH4':
                s = scale
                xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[3]].XCo,NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[3]].YCo,NodeList[Elem.Inzi[0]].YCo]
                xP = [NodeList[Elem.Inzi[0]].XCo+scale*VecU[Elem.DofI[0,0]],NodeList[Elem.Inzi[1]].XCo+scale*VecU[Elem.DofI[1,0]],NodeList[Elem.Inzi[2]].XCo+scale*VecU[Elem.DofI[2,0]],NodeList[Elem.Inzi[3]].XCo+scale*VecU[Elem.DofI[3,0]],NodeList[Elem.Inzi[0]].XCo+scale*VecU[Elem.DofI[0,0]]]
                yP = [NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,1]],NodeList[Elem.Inzi[1]].YCo+scale*VecU[Elem.DofI[1,1]],NodeList[Elem.Inzi[2]].YCo+scale*VecU[Elem.DofI[2,1]],NodeList[Elem.Inzi[3]].YCo+scale*VecU[Elem.DofI[3,1]],NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,1]]]
            elif Elem.Type=='SH3':
                s = scale
                xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[0]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[0]].YCo]
                xP = [NodeList[Elem.Inzi[0]].XCo+scale*VecU[Elem.DofI[0,0]],NodeList[Elem.Inzi[1]].XCo+scale*VecU[Elem.DofI[1,0]],NodeList[Elem.Inzi[2]].XCo+scale*VecU[Elem.DofI[2,0]],NodeList[Elem.Inzi[0]].XCo+scale*VecU[Elem.DofI[0,0]]]
                yP = [NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,1]],NodeList[Elem.Inzi[1]].YCo+scale*VecU[Elem.DofI[1,1]],NodeList[Elem.Inzi[2]].YCo+scale*VecU[Elem.DofI[2,1]],NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,1]]]
            elif Elem.Type=='T1D2':
                xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo]
                yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo]
                xP = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo]
                yP = [NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,0]],NodeList[Elem.Inzi[1]].YCo+scale*VecU[Elem.DofI[1,0]]]
            elif Elem.Type=='S1D2':
                j=0
                k0 = Elem.DofI[j,0]
                k1 = Elem.DofI[j+1,0]
                xS += [NodeList[Elem.Inzi[j]].XCo]
                yS += [NodeList[Elem.Inzi[j]].YCo+scale*(VecU[k1]-VecU[k0])]
            elif Elem.Type=='B23E':
                xN += [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[2]].XCo]
                yN += [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[2]].YCo]
                xP = [NodeList[Elem.Inzi[0]].XCo+scale*VecU[Elem.DofI[0,0]],NodeList[Elem.Inzi[2]].XCo+scale*VecU[Elem.DofI[2,0]]]
                yP = [NodeList[Elem.Inzi[0]].YCo+scale*VecU[Elem.DofI[0,1]],NodeList[Elem.Inzi[2]].YCo+scale*VecU[Elem.DofI[2,1]]]
            elif Elem.Type not in ['SB3','CPE4','CPS4','SH4', 'SH3', 'T1D2','S1D2','CPE3','CPS3','B23E']:
                for j in xrange(Elem.nNod):
                    k0 = Elem.DofI[j,0]
                    k1 = Elem.DofI[j,1]
                    xN += [NodeList[Elem.Inzi[j]].XCo]      # undeformed 2D geometry
                    yN += [NodeList[Elem.Inzi[j]].YCo]      # undeformed 2D geometry
                    xP += [NodeList[Elem.Inzi[j]].XCo+scale*VecU[Elem.DofI[j,0]]]
                    yP += [NodeList[Elem.Inzi[j]].YCo+scale*VecU[Elem.DofI[j,1]]]
            plot(xN,yN, 'b--')
            if (len(xP)+len(yP))>0:
                plot(xP,yP, 'b-',linewidth=LiWiCrv)
                plot(xP,yP, 'o', color=Colors[ElSets[Set]])
            if len(xS)>0:
                plot(xS,yS, 'b-',linewidth=LiWiCrv)
                plot(xS,yS, 'o', color=Colors[ElSets[Set]])
        grid()
        title('Deformed mesh final step (scale '+'%7.4f'%(scale)+')')
        axis('equal')
        yticks(fontsize=FonSizAx) 
        xticks(fontsize=FonSizAx) 
    return 0

def plotMainStressFigures2(sig_MElemList,Sx_S,XList,YList,ZList,ElList,NodeList, mod_Z_limit):
    ColM = plt.get_cmap('brg')
    fig = pl.figure() #figsize=(15.,10.))
    fig.text( 0.05, 0.93, Sx_S, size='x-large')
    maxSig = max(sig_MElemList)
    minSig = min(sig_MElemList)
    if abs(maxSig-minSig)<1.e-6: return 0
    norm_ = colors.Normalize(vmin=minSig, vmax=maxSig)
#    ax1, orie = fig.add_axes([0.90, 0.05, 0.02, 0.8]), 'vertical'
    ax1, orie = fig.add_axes([0.05, 0.05, 0.9, 0.02]), 'horizontal' # left, bottom, width, height] where all quantities are in fractions of figure width and height
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=ColM,  norm=norm_, orientation=orie) # norm=norm_,
    ax = a3.Axes3D(fig)
    GesamtDiff = abs(max(sig_MElemList) - min(sig_MElemList))
    prozentualerNegAnteil = abs(min(sig_MElemList)) / GesamtDiff
    prozentualerPosAnteil = abs(max(sig_MElemList)) / GesamtDiff
    for i in xrange(len(ElList)):
        elem, CoordList = ElList[i], []
        for j in xrange(len(elem.Inzi)):
            node = NodeList[elem.Inzi[j]]
            CoordList += [[node.XCo, node.YCo, node.ZCo]]
        tri = a3.art3d.Poly3DCollection([CoordList])
#        if sig_MElemList[i] <= 0.:
#            sig_MElemList[i] = minSig - sig_MElemList[i] # i-tes Element wird zu einem Differenzwert
#            sig_MElemList[i] = sig_MElemList[i] / minSig # i-tes Element wird zu einem Prozentwert, der den Spannungswert auf einer Skala von 0 bis 1 fuer den negativen Bereich abbildet
#            sig_MElemList[i] = sig_MElemList[i] * prozentualerNegAnteil # i-tes Element bildet den Prozentwert, der sich im vorigen Schritt noch nur auf den negativen Bereich bezogen hat, auf den gesamten Bereich ab
#        else:
#            sig_MElemList[i] = sig_MElemList[i] #sig_MElemList[i] = max(sig_MElemList) - sig_MElemList[i] # i-tes Element wird zu einem Differenzwert
#            sig_MElemList[i] = sig_MElemList[i] / maxSig # i-tes Element wird zu einem Prozentwert, welcher den Spannungswert auf einer Skala von 0 bis 1 fuer den positiven Bereich abbildet
#            sig_MElemList[i] = sig_MElemList[i] * prozentualerPosAnteil + prozentualerNegAnteil# i-tes Element bildet den Prozentwert, der sich im vorigen Schritt noch nur auf den positiven Bereich bezogen hat, auf den gesamten Bereich ab
#        tri.set_color(ColM(sig_MElemList[i]))
        tri.set_color(ColM((sig_MElemList[i] - minSig)/(maxSig-minSig)))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    ax.set_xlim3d(min(XList), max(XList))
    ax.set_ylim3d(min(YList), max(YList))
    ax.set_zlim3d(mod_Z_limit*min(ZList), mod_Z_limit*max(ZList))
    ax.set_xlabel('x',fontsize='x-large')
    ax.set_ylabel('y',fontsize='x-large')
    ax.set_zlabel('z',fontsize='x-large')
    ax.tick_params(axis='x', labelsize='x-large')
    ax.tick_params(axis='y', labelsize='x-large')
    ax.tick_params(axis='z', labelsize='x-large')
    ax.set_aspect('equal') #,'box', 'equal')
    ax.grid()
    return 0

def plotDeformFigure(flag, DisElemList,VecU,ElList,NodeList,text, mX, mY, mZ):
    xMin, xMax, XM = mX[0], mX[1], mX[2]
    yMin, yMax, YM = mY[0], mY[1], mY[2]
    zMin, zMax, ZM = mZ[0], mZ[1], mZ[2]
    dMax, dMin = max(DisElemList), min(DisElemList)
    dMax = max(abs(dMax),abs(dMin))
    scale_ = 20. # 20.  # displayed deformation should be roughly 1/scale_ of parts dimensions                     # ts
    scale = max(XM,YM,ZM)/dMax/scale_
    ColM = plt.get_cmap('summer')
    fig = pl.figure()#(figsize=(15.,10.))
    fig.text( 0.05, 0.93, 'Deformed shape - scale: {:7.3f} '.format(scale)+text, size='x-large') # , ha='right')
    norm_ = colors.Normalize(vmin=0., vmax=dMax)
#    ax1, orie = fig.add_axes([0.90, 0.05, 0.02, 0.8]), 'vertical' # left, bottom, width, height] where all quantities are in fractions of figure width and height
    ax1, orie = fig.add_axes([0.05, 0.05, 0.9, 0.02]), 'horizontal' # left, bottom, width, height] where all quantities are in fractions of figure width and height
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=ColM, norm=norm_, orientation=orie)
    ax = a3.Axes3D(fig)
    DisElemList = map(abs,DisElemList)/dMax
    for i in xrange(len(ElList)):
        elem, CoordList = ElList[i], []
        for j in xrange(len(elem.Inzi)):
            node = NodeList[elem.Inzi[j]]
            DofInd = node.GlobDofStart
            if   elem.Type=='SH4': CoordList += [[node.XCo-xMin+scale*VecU[DofInd], node.YCo-yMin+scale*VecU[DofInd+1], node.ZCo-zMin+scale*VecU[DofInd+2]]]
            elif elem.Type=='SB3': CoordList += [[node.XCo-xMin, node.YCo-yMin, node.ZCo-zMin+scale*VecU[DofInd]]]
            else: 
                print elem.Type
                raise NameError ("Plot3DElements not yet support this type of elements")
        tri = a3.art3d.Poly3DCollection([CoordList])
        tri.set_color(ColM(DisElemList[i]))
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    limits = max(zMax - zMin, xMax -xMin, yMax - yMin)
    axesratio = 0.3     # minimum of smallest dimension to largest dimension in 3d coordinate system displayed     # ts
    ax.set_zlim3d(0.,max( axesratio*limits, zMax - zMin))
    ax.set_xlim3d(0.,max( axesratio*limits, xMax - xMin))
    ax.set_ylim3d(0.,max( axesratio*limits, yMax - yMin))
    ax.set_xlabel('x',fontsize='x-large')
    ax.set_ylabel('y',fontsize='x-large')
    ax.set_zlabel('z',fontsize='x-large')
    ax.tick_params(axis='x', labelsize='x-large')
    ax.tick_params(axis='y', labelsize='x-large')
    ax.tick_params(axis='z', labelsize='x-large')
    ax.set_aspect('equal')
    ax.grid()
    return 0

def mainStressConverter(sig_x,sig_y,sig_xy): # Anwendung des Mohrschen Spannungskreises zur Ausgabe der Hauptspannungen
    point1 = [sig_x,sig_xy]
    #point2 = [sig_y,-sig_xy]
    xCenterPoint=min(sig_x,sig_y)+abs(sig_x - sig_y)/2
    centerPoint = [xCenterPoint,0] # entspricht dem Mittelpunkt des Morschenspannungskreises
    radiusMohr = sqrt((point1[0]-centerPoint[0])**2 + (point1[1]-centerPoint[1])**2)
    sigma1 = xCenterPoint + radiusMohr
    sigma2 = xCenterPoint - radiusMohr
    #kreisSehne = # fuer die Berechnung der Hauptspannungsorientierung
    #mittelPunktsWinkel = # fuer die Berechnung der Hauptspannungsorientierung
    return sigma1,sigma2

def FinishAllStuff(PloF, Name, ElList, NodeList, MatList, f2, VecU, WrNodes, ResType):
    if PloF:
        SE, SN, SP, PE2DFlag, mod_Z_limit, PE3DFlag = 1.0, 1.0, 1.0, True, 1.0, False
        if path.isfile(Name+".plt.txt"):                     # read plot options file if there is any
            f1=open( Name+".plt.txt", 'r')
            SE, SN, SP, PE2DFlag, mod_Z_limit, PE3DFlag = ReadPlotOptionsFile(f1)                    # to modify plot scaling factors
        else: PE3DFlag = False
        f2= file( Name+".elemout.txt", "rb")
        Scales, Scales1 = PostScales( ElList, NodeList, f2 )
        f2.seek(0)
        if PE2DFlag: PostElem2D( ElList, NodeList, MatList, f2, Scales, Scales1, SE, SP )
        if PE3DFlag: PostElem3D( ElList, NodeList, VecU, SN, mod_Z_limit)
        f2.seek(0)
        PostElem1D( f2 )
        PostNode( ElList, NodeList, VecU, SN, mod_Z_limit, PE3DFlag)    # plot deformed mesh
        if WrNodes<>None and len(WrNodes)>0:
            f5_name= Name+".timeout.txt"
            try:
                PlotNodes( f5_name, WrNodes, PlotOpt=2 )    # plot Displ-Reaction curve based on WrNode file
            except NameError:
                print NameError("ConFemInOut:PlotNodes: unknown key")
            except:
                print TypeError("ConFemInOut:PlotNodes: too much variables in .timeout file")
        #try:
        #    XList, YList, ZList, DisElemList = [], [], [], []
        #    for i in NodeList:
        #        XList += [i.XCo]
        #        YList += [i.YCo]
        #        ZList += [i.ZCo]
        #    mX, mY, mZ = [np.min(XList), np.max(XList), np.mean(XList)], [np.min(YList), np.max(YList),
        #                                                                  np.mean(YList)], [np.min(ZList),
        #                                                                                    np.max(ZList),
        #                                                                                    np.mean(ZList)]
        #    for elem in ElList:
        #        DisList = []
        #        for o in elem.Inzi:
        #            node = NodeList[o]
        #            DofInd = node.GlobDofStart
        #            if elem.Type == 'SH4':
        #                DisList += [sqrt(VecU[DofInd] ** 2 + VecU[DofInd + 1] ** 2 + VecU[DofInd + 2] ** 2)]
        #            elif elem.Type == 'SB3':
        #                DisList += [VecU[DofInd]]
        #        DisElemList += [np.mean(DisList)]
        #    plotDeformFigure('deform', DisElemList, VecU, ElList, NodeList, ' deformed mesh', mX, mY, mZ)
        #except:
        #    print NameError("ConFemInOut:plotDeformFigure: detected error in 3D plot figure, only SB3 and SH4 are supported")

        show()
        return 0
    else:
        import hashlib
        mmm = hashlib.md5()
        fp= file( Name+"."+ResType+".txt", "rb")
        while True:
            data= fp.read(65536)
            if not data: break
            mmm.update(data)
        fp.close()
        RC = mmm.hexdigest()
        return RC

def LogResiduals(LogName, counter, i, NodeList, VecR, VecU):
    if not path.exists(LogName):
        makedirs(LogName)
    ff = open( LogName+"/log-"+str(counter)+"-"+str(i)+".txt", 'w')              #
    f1 = open( LogName+"/logD-"+str(counter)+"-"+str(i)+".txt", 'w')              #
    for Node in NodeList:
        iS = Node.GlobDofStart
        if len(Node.DofT)>0:
            ff.write('%5i%9.4f%9.4f%9.4f%4i'%(Node.Label,Node.XCo,Node.YCo,Node.ZCo,len(Node.DofT)))
            f1.write('%5i%9.4f%9.4f%9.4f%4i'%(Node.Label,Node.XCo,Node.YCo,Node.ZCo,len(Node.DofT)))
            xxx = 0.
            for j in xrange(len(Node.DofT)):
                xxx = xxx +VecR[iS+j]**2
                ff.write('%12.4e'%(VecR[iS+j]))
                f1.write('%12.4e'%(VecU[iS+j]))
            ff.write('   %12.4e\n'%(sqrt(xxx)))
            f1.write('\n'%(sqrt(xxx)))
    ff.close()
    f1.close()
    return

def PlotResiduals( LogName, Name, counter, i):
    def Stuff( Pl, Name, scale, scale2):
        ff, xx, yy, rr = open( Name, 'r'), [], [], 0.
        z1 = ff.readline()                                  # 1st input line
        z2 = z1.split()
        while z1<>"":
            x, y, rx, ry = float(z2[1]),float(z2[2]), float(z2[5]), float(z2[6])
 #           x, y, rx, ry, ax, ay = float(z2[1]),float(z2[2]), float(z2[4]), float(z2[5]), float(z2[7]), float(z2[8]) 
            rr = rr + rx**2 + ry**2
            xx += [x]
            yy += [y]
            Pl.plot([x,x+scale*rx],[y,y+scale*ry],color='red')
    #        Pl.plot([x,x+scale2*ax],[y,y+scale2*ay],color='blue')
            z1 = ff.readline()                                  # 1st input line
            z2 = z1.split()
        ff.close()
        print sqrt(rr)
        Pl.plot(xx,yy,'x')
        Pl.set_aspect('equal')
        Pl.grid()
        return 
    Name = LogName+"/log-"+str(counter)+"-"+str(i)+".txt"
    Pl1 = plt.figure().add_subplot(111,title='Resid '+str(counter)+'/'+str(i))
#    Stuff( Pl1, Name, 1.0e-3, 1.0e1) # 1e1
    Stuff( Pl1, Name, 3.0e+1, 1.0e1) # 1e1
#    Name = LogName+"/logD-"+str(counter)+"-"+str(i)+".txt"
#    Pl2 = plt.figure().add_subplot(111,title='Displ '+str(counter)+'/'+str(i))
#    Stuff( Pl2, Name, 1.0e2, 1.0e1)
    show()
    return

def WriteResiduals( LogName, counter, i, N):
    class Residual():                                  # 1D Truss 2 nodes
        def __init__(self, Label, X, Y, Z, Res):
            self.Label = Label
            self.X   = X
            self.Y   = Y
            self.Z   = Z
            self.Res = Res
    Name = LogName+"/log-"+str(counter)+"-"+str(i)+".txt"
    Nam1 = LogName+"/logWr-"+str(counter)+"-"+str(i)+".txt"
    ResList = []
    ff  = open( Name, 'r')
    f1  = open( Nam1, 'w')
    z1 = ff.readline()                              
    z2 = z1.split()
    while z1<>"":
        nL, x, y, z, nD, rr  = int(z2[0]), float(z2[1]), float(z2[2]), float(z2[3]), int(z2[4]), float(z2[-1])
        ResList += [Residual( nL, x, y, z, rr)]
        f1.write("%5i ; %8.2f ; %8.2f ; %8.2f ; %16.6e\n"%(nL,x,y,z,rr))
        z1 = ff.readline()
        z2 = z1.split()
    ff.close()
    f1.close()
    n = len(ResList)
    ResList.sort(key=lambda t: t.Res)
    print Name
    for i in xrange(N):
        r = ResList[n-1-i]
        print i+1, r.Label, r.Res
