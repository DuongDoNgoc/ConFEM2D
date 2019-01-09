# ConFemElem
# Copyright (C) [2014] [Ulrich Haeussler-Combe]
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License (GNU GPLv3) as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program; if not, see <http://www.gnu.org/licenses
#
"""Element library for ConFem"""

from numpy import zeros, array, sqrt, double, fabs, ma, dot, copy, transpose
from numpy.linalg import inv
from bisect import bisect_left, bisect_right
import sys 

from ConFEM2D_Basics import FindIndexByLabel, SamplePoints, SampleWeight, SamplePointsRCShell, SampleWeightRCShell, ZeroD

class Node(object):
    def __init__(self, Label, XCo, YCo, ZCo, XDir, YDir, ZDir):
        self.DofT = set([])
        self.GlobDofStart = 0
        self.c = 0
        self.Label = Label
        self.XCo = XCo
        self.YCo = YCo
        self.ZCo = ZCo
        self.XDi = XDir
        self.YDi = YDir
        self.ZDi = ZDir
        self.CMLabel_ = 0                               # modified label from application or cuthill mckee

class NodeResult(object):                               # for postprocessing with read from nodeout.txt
    def __init__(self, Label, XDi, YDi, ZDi, XRo, YRo, ZRo, XFo, YFo, ZFo, XMo, YMo, ZMo):
        self.Label = Label
        self.XDi = XDi
        self.YDi = YDi
        self.ZDi = ZDi
        self.XRo = XRo
        self.YRo = YRo
        self.ZRo = ZRo
        self.XFo = XFo
        self.YFo = YFo
        self.ZFo = ZFo
        self.XMo = XMo
        self.YMo = YMo
        self.ZMo = ZMo
        self.ResNorm = 0.                               # length / norm of nodal residuum
        self.Result = 0.                                # ?
        self.ResultCounter = 0                          # ?
        
class ElemResult(object):
    # used by ConFemInOut::ReadElemResults
    def __init__(self, Type, Label, elIndex, InzInd, ipList, LLCm, LLCp, LLR, LLDlo, LLDup):
        self.Type = Type
        self.Label = Label
        self.elIndex = elIndex                          # index of element in ElList
        self.SH4_InzInd = InzInd                        # indices of incidencies / nodes in NodeList
        self.SH4_ipList = ipList                        # results / internal integrated forces
        self.SH4_ipCrSm = LLCm                          # results / solid stresses lower (ip over height) principal values
        self.SH4_ipCrSp = LLCp                          # results / solid stresses upper (ip over height) principal values
        self.SH4_ipRe   = LLR                           # results reinforcement stresses
        self.SH4_ipDlo  = LLDlo                         # results damage lower ip 
        self.SH4_ipDup  = LLDup                         # results damage upper ip

class SolidSection(object):
    """ solid section  parameters (-> continuum elements)
    
    @var: Set    element set label (string) 
    @var: Mat    material label (string)
    @var: Val:    generally thickness (float)
    @author: 
    """
    def __init__(self, SetLabel, Mat, Val):
        self.Set = SetLabel
        self.Mat = Mat
        self.Val = Val

class BeamSectionRect(object):
    """ XXX.
    """
    def __init__(self, Type, SetLabel, Mat, Width, Height, Reinf):
        self.Type = Type                                # Shape, Type
        self.Set = SetLabel                             # Element Set
        self.Mat = Mat                                  # Material
        self.Width = Width                              # Width
        self.Height = Height                            # Height
        self.Reinf = Reinf                              # Reinforcement

class BeamSectionPoly(object):
    """ XXX.
    """
    def __init__(self, Type, SetLabel, Mat, Reinf):
        self.Type = Type                                # Shape, Type
        self.Set = SetLabel                             # Element Set
        self.Mat = Mat                                  # Material
        self.Reinf = Reinf                              # Reinforcement
        self.Poly = []                                  # Description of cross section by polyline
    def AddPoint(self, zz, bb):
        nn = len(self.Poly)
        if nn>0: 
            lz=self.Poly[nn-1][0]
            if lz>zz: raise NameError ("ConFemElements::AddPoint: wrong data order")
        self.Poly += [[zz,bb]]

class ShellSection(object):
    """ YYY.
    """
    def __init__(self, SetLabel, Mat, Height, Reinf):
        self.Set = SetLabel                             # Element Set
        self.Mat = Mat                                  # Material
        self.Height = Height                            # Value
        self.Reinf = Reinf                              # Reinforcement

class Element(object):
    """ holds element parameters.
    """
    def __init__(self, TypeVal,nNodVal,DofEVal,nFieVal, IntTVal,nIntVal,nIntLVal, DofTVal,DofNVal, dimVal,NLGeomIVal):
        self.Rot = False                                # Flag for elements with local coordinate system
        self.RotM= False                                # Flag for elements with local coordinate system for materials
        self.RotG= False                                # Flag for elements with local coordinate system for geometric stiffness
        self.ZeroD = 1.e-9                              # Smallest float for division by Zero
        self.StateVar = []                              # default no state variables
        self.Type = TypeVal                             # element type
        self.nNod = nNodVal                             # number of nodes
        self.DofE = DofEVal                             # number of dofs for whole element
        self.DofEini = DofEVal                          # as DofE may be subject to change, e.g. for sh4
        self.nFie = nFieVal                             # degrees of freedom for external loading
                                                        # corresponds to largest load-DofT used by element and its dimension/structure of shape matrix N 
        self.IntT = IntTVal                             # Integration type 0: 1D Gaussian, 1: 2D Gaussian, 2: 3D Gaussian, 3: 2D triangular, 4: shells
        self.nInt = nIntVal                             # integration order (--> index for points and weights = nInt-1)
        self.nIntL= nIntLVal                            # total number of integration points (may be increased dynamically for some element/material type combinations, nInt is probably used for distinction of cases 
        self.nIntLi = nIntLVal                          # initial value for number of integration points (is not increased dynamically, used for distinction of bulk and reinforcement)
        self.DofT = DofTVal                             # tuple, type of dof for every node of this element
                                                        # 1 -> u_x, 2->u_y, 3->u_z, 4->phi around x, 5->phi around y, 6->phi around z, 7->gradient field 
        self.DofN = DofNVal                             # tuple, number of dofs ( -> len(DofT[i]) ) for every node of this element
        self.DofNini=DofNVal                            # as DofN may be subject to change, e.g. for sh4
        self.dim  = dimVal                              # material type index, bernoulli beam
        self.NLGeomI = NLGeomIVal                       # Flag whether large deformations are already implemented for this element
        self.RegType = 0                                # default value no regularization
        self.ElemUpdate = False                         # default for update of element data
        self.NLGeomCase = 0                             # determines how geometric stiffness is built from element contributions
        self.ShellRCFlag = False                        # indicator for combination of reinforcement with bulk. Currently used for postprocessing of SH4
        self.CrBwS = 1.                                 # scaling factor for crack regularization
    def CrossSecGeom(self, zc1, zc2):                   # cross section area, statical moment, inertial moment for 2D cross secttion defined with polyline within z limits
        pList = self.CrSecPolyL
        n = len(pList)
        if n<2: raise NameError ("ConFemElements::CrossSecGeom: not enough data")
        AA, SS, JJ = 0, 0, 0
        On6, On12 = 1./6, 1./12
        for i in range(n-1):
            z1, b1 = max(zc1,pList[i][0]), pList[i][1]
            z2, b2 = min(zc2,pList[i+1][0]), pList[i+1][1]
            if z1>z2: z1=z2                             # Section with tension only
            AA = AA+0.5*(b1+b2)*(z2-z1)                 # cross section area
            S1, S2, S3 =(2*b2+b1), (b1-b2), (2*b1+b2)
            SS = SS + On6*(S1*z2**2+S2*z1*z2-S3*z1**2)  # cross section statical moment
            J1, J2, J3 = (3*b2+b1), (b1-b2), -(3*b1+b2) 
            JJ = JJ + On12*(J1*z2**3+J2*(z1*z2**2+z1**2*z2)+J3*z1**3) # cross section intertial moment
#            print self.Label, zc1,max(zc1,pList[i][0]),min(zc2,pList[i+1][0])
#        print self.Label, AA, SS, JJ
        return AA, SS, JJ, pList[n-1][0]-pList[0][0]
    def WidthOfPolyline(self, zz):                      # width of 2D cross section defined by polyline depending on z
        pList = self.CrSecPolyL
        n = len(pList)
        for i in range(n-1):
            z1, b1 = pList[i][0], pList[i][1]
            z2, b2 = pList[i+1][0], pList[i+1][1]
#            if zz>=z1: # and (zz-z2)<=ZeroD:
            if zz <= (z2+1.e-6): # and (zz-z2)<=ZeroD:
                if (z2-z1)>ZeroD: bb = ((b2-b1)*zz+z2*b1-b2*z1)/(z2-z1)
                else:             bb = b1
                return bb
    def CrossSecGeomA(self, zc1, zc2, s1, s2):      # matrix AZ, geometrical tangential stiffness for 2D polyline cross section 
        pList = self.CrSecPolyL
        n = len(pList)
        AZ = zeros((2,2))
        for i in range(n-1):
            z1, b1, z2, b2 = pList[i][0], pList[i][1], pList[i+1][0], pList[i+1][1]
            if zc1>=z1:
                if (z2-z1)>ZeroD: 
                    AZ[0,0] = -1./6*s1*b2-1./3*s1*b1-1./3*s2*b2-1./6*s2*b1
                    AZ[1,0] = 1./6*(b2*zc1+zc2*b2+zc1*b1)*s2+1./6*(b2*zc1-zc2*b1+3.*zc1*b1)*s1
                else:             AZ[0,0] = 0; AZ[1,0] = 0
                break
        for i in range(n-2,0,-1):
            z1, b1, z2, b2 = pList[i][0], pList[i][1], pList[i+1][0], pList[i+1][1]
            if zc2<=z2:
                if (z2-z1)>ZeroD: 
                    AZ[0,1] = 1./6*s1*b2+1./3*s1*b1+1./3*s2*b2+1./6*s2*b1;
                    AZ[1,1] = 1./6*(b2*zc1-3.*zc2*b2-zc2*b1)*s2-(1./6*zc1*b1+1./6*zc2*b2+1./6*zc2*b1)*s1
                else:             AZ[0,1] = 0; AZ[1,1] = 0
                break
        return AZ
    # initialization for all 2D beam elements
    def IniBeam2(self, BeamSec, ReinfD, NData, StateV):
        self.TensStiff = True                               # flag for tension stiffening
        self.NLGeomCase = 1                                 # determines how geometric stiffness is built from element contributions, see ConFemBasics:Intforces
        nRe = len(ReinfD)                                   # ReinfD presumably comes from *BEAM REINF input
        if nRe>0: ReiP = ReinfD
        else:     nRe, ReiP = len(BeamSec.Reinf), BeamSec.Reinf # comes from *BEAM SSECTION / RECT input 
        self.Geom = zeros( (2+nRe,6), dtype=double)         # all geometry data for concrete and reinforcement, tension stiffening
        self.Geom[1,0] = 1                                  # for compatibility reasons
        if BeamSec.Type=="POLYLINE": 
            self.CrSecPolyL = BeamSec.Poly
            AA, SS, JJ, hh = self.CrossSecGeom(self.CrSecPolyL[0][0],self.CrSecPolyL[len(self.CrSecPolyL)-1][0])# area, statical moment, inertial moment
            print "A", AA, SS, JJ
            self.Geom[1,1] = self.CrSecPolyL[0][0]          # lower coordinate
            self.Geom[1,2] = self.CrSecPolyL[-1][0]         # upper coordinate
        else:                 
            AA, SS, JJ     = BeamSec.Width*BeamSec.Height, 0, BeamSec.Width*BeamSec.Height**3/12.  
            self.Geom[1,1] = BeamSec.Width
            self.Geom[1,2] = BeamSec.Height
        self.Geom[1,3] = AA
        self.Geom[1,4] = SS
        self.Geom[1,5] = JJ
        self.RType = []                                     # Type of reinforcement, RC ordinary rebar, TC textile or else
        for j in range(nRe):
            self.Geom[2+j,0] = ReiP[j][0]                   # reinforcement cross section
            self.Geom[2+j,1] = ReiP[j][1]                   # " lever arm
            self.Geom[2+j,2] = ReiP[j][2]                   # effective reinforcement ratio
            self.Geom[2+j,3] = ReiP[j][3]                   # tension stiffening parameter betat
            self.RType += [ReiP[j][4]]
        self.Data = zeros((self.nIntL,NData), dtype=float)  # storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)  # storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL*(nRe+1),StateV)) #, dtype=float)
            self.StateVarN= zeros((self.nIntL*(nRe+1),StateV)) #, dtype=float)

class B23E(Element):
    """ Bernoulli Beam 2D 2 nodes, cubic shape for transverse displacements.
    """
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, BeamSec, ReinfD, StateV, NData, Full):
        self.Full = Full
        if Full:
            Element.__init__(self,"B23E",3,7,2, 0,3,3, (set([1, 2, 6]),set([1]),set([1, 2, 6])),(3,1,3), 10,  True)
#            Element.__init__(self,"B23E",3,7,2, 0,2,2, (set([1, 2, 6]),set([1]),set([1, 2, 6])),(3,1,3), 10,  True)
            self.Label = Label                              # element number in input
#            self.DofI = zeros( (self.nNod,3), dtype=int)    # indices of global dofs per node
            self.DofI = ma.masked_array( [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]], dtype=int, mask=[[0,0,0],[0,1,1],[0,0,0]])    # indices of global dofs per node
            self.MatN = MatName                             # name of material
            self.Set = SetLabel                             # label of corresponding element set
            self.CrSecPolyL = None                          # polyline to describe cross section
            self.InzList = [InzList[0], InzList[1], InzList[2]]
        else: 
            self.nIntL =  1
            self.dim   = 10
        self.IniBeam2( BeamSec, ReinfD, NData, StateV)
    def Ini2(self, NoList, MaList):
        if self.Full:
            i0 = FindIndexByLabel( NoList, self.InzList[0]) # find node index from node label
            i1 = FindIndexByLabel( NoList, self.InzList[1])
            i2 = FindIndexByLabel( NoList, self.InzList[2])
            self.CoordRef = array( [NoList[i0].XCo, NoList[i0].YCo, NoList[i2].XCo, NoList[i2].YCo]) # Nodal coordinates 
            self.CoordDis = array( [NoList[i0].XCo, NoList[i0].YCo, NoList[i2].XCo, NoList[i2].YCo]) # Nodal coordinates displaced 
            L = sqrt( (NoList[i2].XCo-NoList[i0].XCo)**2 + (NoList[i2].YCo-NoList[i0].YCo)**2 )# element length
            self.Geom[0,0] = 0.5*L                          # Jacobi determinant value
            self.Lch_ = L                                    # characteristic length
            cosA = (NoList[i2].XCo-NoList[i0].XCo)/L        # direction cosine
            sinA = (NoList[i2].YCo-NoList[i0].YCo)/L        # direction sine
            if fabs(sinA) >= self.ZeroD:                    # element direction rotated to global axis
                self.Rot = True
                self.Trans = zeros((self.DofE, self.DofE), dtype=float)# coordinate transformation matrix
                for i in xrange(self.DofE): self.Trans[i,i] = cosA# fill transformation axis
                self.Trans[2,2] = 1.
                self.Trans[3,3] = 1.                        # longitudinal displ of center node remains local!
                self.Trans[6,6] = 1.
                self.Trans[0,1] = sinA
                self.Trans[1,0] = -sinA
                self.Trans[4,5] = sinA
                self.Trans[5,4] = -sinA
            self.Inzi = ( i0, i1, i2 )                      # tupel, indices of nodes belonging to element
            NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
            NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
            NoList[i2].DofT = NoList[i2].DofT.union(self.DofT[2])
        else:
            pass
        pass
    def UpdateCoord(self, dis, ddis ):
        if not self.Rot: self.Trans = zeros((self.DofE, self.DofE), dtype=float)# coordinate transformation matrix
        self.Rot = True
        self.CoordDis[0] = self.CoordRef[0]+dis[0]
        self.CoordDis[1] = self.CoordRef[1]+dis[1]
        self.CoordDis[2] = self.CoordRef[2]+dis[4]
        self.CoordDis[3] = self.CoordRef[3]+dis[5]
        L = sqrt( (self.CoordDis[2]-self.CoordDis[0])**2 + (self.CoordDis[3]-self.CoordDis[1])**2 )# element length
        cosA = (self.CoordDis[2]-self.CoordDis[0])/L    # direction cosine
        sinA = (self.CoordDis[3]-self.CoordDis[1])/L    # direction sine
        for i in xrange(self.DofE): self.Trans[i,i] = cosA# fill transformation axis
        self.Trans[2,2] = 1.
        self.Trans[3,3] = 1.                        # longitudinal displ of center node remains local!
        self.Trans[6,6] = 1.
        self.Trans[0,1] = sinA
        self.Trans[1,0] = -sinA
        self.Trans[4,5] = sinA
        self.Trans[5,4] = -sinA
        self.Geom[0,0] = 0.5*L
        self.Geom[0,1] = sinA
        self.Geom[0,2] = cosA
        return
    def GeomStiff(self, r, s, t, sig):
        GeomK = zeros((self.DofE, self.DofE), dtype=float)
#        return GeomK
        L = 2.*self.Geom[0,0]
        sinA = self.Geom[0,1]
        cosA = self.Geom[0,2]
        NI = -sig[0]/L
        VI =  sig[1]*6*r/L**2
        AI = sinA*NI + cosA*VI
        BI = cosA*NI - sinA*VI
        GeomK[0,0] = -AI*sinA
        GeomK[0,1] =  AI*cosA
        GeomK[0,4] =  AI*sinA
        GeomK[0,5] = -AI*cosA
        GeomK[1,0] =  BI*sinA
        GeomK[1,1] = -BI*cosA
        GeomK[1,4] = -BI*sinA
        GeomK[1,5] =  BI*cosA
        GeomK[4,0] = -GeomK[0,0]
        GeomK[4,1] = -GeomK[0,1]
        GeomK[4,4] = -GeomK[0,4]
        GeomK[4,5] = -GeomK[0,5]
        GeomK[5,0] = -GeomK[1,0]
        GeomK[5,1] = -GeomK[1,1]
        GeomK[5,4] = -GeomK[1,4]
        GeomK[5,5] = -GeomK[1,5]
        return GeomK
    def FormX(self, r, s, t):
        """ Form functions.
        X
        """
        X = array([ 0.5*(1-r), 0., 0.5*(1+r)])
        return X
    def FormN(self, r, s, t):
        """ Form functions.
        X
        """
        L = 2.*self.Geom[0,0]
        N = array([[0.5*r*(r-1), 0., 0.,1-r**2,0.5*r*(1+r), 0., 0.],
                   [0., 0.25*(r**3-3*r+2), L*0.125*(r**3-r**2-r+1), 0., 0., 0.25*(-r**3+3*r+2), L*0.125*(r**3+r**2-r-1)]])
        return N
    def FormB(self, r, s, t, NLg):
        """ Derivatives of form functions.
        1st row: force, 3rd row: local vertical coordinate, 4th row: local inclination
        """
        L = 2.*self.Geom[0,0]
        B = array([[(2*r-1)/L, 0., 0., -4.*r/L, (2*r+1)/L, 0., 0.],
                   [0., 6*r/L**2, (3*r-1)/L, 0., 0., -6*r/L**2, (3*r+1)/L]])
        return B, 1, 0
    def FormT(self, r, s, t):                                 # for temperatures
        """ Form functions for temperature interpolation.
        1st row: force, 3rd row: local vertical coordinate, 4th row: local inclination
        """
        T = array([[ 0.5*(1-r), 0., 0., 0., 0.5*(1+r), 0., 0.],
                   [ 0., 0.5*(1-r), 0., 0., 0., 0.5*(1+r), 0.]])
        return T
    def FormP(self, r, s): 
        """ Form functions for prestressing interpolation.
        1st row: force, 3rd row: local vertical coordinate, 4th row: local inclination
        """
        L = 2.*self.Geom[0,0]
        P = array([[0.5*(1-r), 0., 0., 0, 0.5*(1+r), 0., 0.],
                   [0, 0.25*(r**3-3*r+2), L*0.125*(r**3-r**2-r+1),0,0, 0.25*(-r**3+3*r+2), L*0.125*(r**3+r**2-r-1)],
                   [0, 0.5* (3*r**2-3)/L, 0.25*(3*r**2-2*r-1),    0,0, 0.5* (-3*r**2+3)/L, 0.25*(3*r**2+2*r-1)]])
        return P
    def JacoD(self, r, s, t):
        """ Dummy for jacobian determinant.
        """
        return 1

class B23(Element):                                     # Bernoulli Beam 2D 2 nodes, cubic shape
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, BeamSec, ReinfD, StateV, NData):
        Element.__init__(self,"B23", 2,6,2, 0,2,2, (set([1, 2, 6]),         set([1, 2, 6])),(3,3),   10, True)
#        Element.__init__(self,"B23", 2,6,2, 0,1,1, (set([1, 2, 6]),         set([1, 2, 6])),(3,3),   10, True)
        self.Label = Label                              # element number in input
#        self.DofI = zeros( (self.nNod,3), dtype=int)# indices of global dofs per node
        self.DofI = ma.masked_array( [[-1,-1,-1],[-1,-1,-1]], dtype=int, mask=[[0,0,0],[0,0,0]])    # indices of global dofs per node
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.CrSecPolyL = None                          # polyline to describe cross section
        self.TensStiff = True                           # flag for tension stiffening
        self.InzList = [InzList[0], InzList[1]]
        self.IniBeam2( BeamSec, ReinfD, NData, StateV)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        self.CoordRef = array( [NoList[i0].XCo, NoList[i0].YCo, NoList[i1].XCo, NoList[i1].YCo]) # Nodal coordinates 
        self.CoordDis = array( [NoList[i0].XCo, NoList[i0].YCo, NoList[i1].XCo, NoList[i1].YCo]) # Nodal coordinates displaced 
        L = sqrt( (NoList[i1].XCo-NoList[i0].XCo)**2 + (NoList[i1].YCo-NoList[i0].YCo)**2 )# element length
        self.Geom[0,0] = 0.5*L                          # Jacobi determinant value
        self.Lch_ = L                                    # characteristic length
        cosA = (NoList[i1].XCo-NoList[i0].XCo)/L        # direction cosine
        sinA = (NoList[i1].YCo-NoList[i0].YCo)/L        # direction sine
        if fabs(sinA) >= self.ZeroD:                    # element direction rotated to global axis
            self.Rot = True
            self.Trans = zeros((self.DofE, self.DofE), dtype=float)# coordinate transformation matrix
            for i in xrange(self.DofE): self.Trans[i,i] = cosA# fill transformation axis
            self.Trans[2,2] = 1.
            self.Trans[5,5] = 1.
            self.Trans[0,1] = sinA
            self.Trans[1,0] = -sinA
            self.Trans[3,4] = sinA
            self.Trans[4,3] = -sinA
        self.Inzi = [ i0, i1]                           # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
    def UpdateCoord(self, dis, ddis ):
        if not self.Rot: self.Trans = zeros((self.DofE, self.DofE), dtype=float)# coordinate transformation matrix
        self.Rot = True
        self.CoordDis[0] = self.CoordRef[0]+dis[0]
        self.CoordDis[1] = self.CoordRef[1]+dis[1]
        self.CoordDis[2] = self.CoordRef[2]+dis[3]
        self.CoordDis[3] = self.CoordRef[3]+dis[4]
        L = sqrt( (self.CoordDis[2]-self.CoordDis[0])**2 + (self.CoordDis[3]-self.CoordDis[1])**2 )# element length
        cosA = (self.CoordDis[2]-self.CoordDis[0])/L    # direction cosine
        sinA = (self.CoordDis[3]-self.CoordDis[1])/L    # direction sine
        for i in xrange(self.DofE): self.Trans[i,i] = cosA# fill transformation axis
        self.Trans[2,2] = 1.
        self.Trans[5,5] = 1.
        self.Trans[0,1] = sinA
        self.Trans[1,0] = -sinA
        self.Trans[3,4] = sinA
        self.Trans[4,3] = -sinA
        self.Geom[0,0] = 0.5*L
        self.Geom[0,1] = sinA
        self.Geom[0,2] = cosA
        return
    def GeomStiff(self, r, s, t, sig):
        GeomK = zeros((self.DofE, self.DofE), dtype=float)
        sinA = self.Geom[0,1]
        cosA = self.Geom[0,2]
        L = 2.*self.Geom[0,0]
        NI = -sig[0]/L
        VI =  sig[1]*6*r/L**2
#        NJ =  sig[0]/L
#        VJ = -sig[1]*6*r/L**2
        AI = sinA*NI + cosA*VI
        BI = cosA*NI - sinA*VI
        AJ = -AI # sinA*NJ + cosA*VJ
        BJ = -BI # cosA*NJ - sinA*VJ
        GeomK[0,0] = -AI*sinA
        GeomK[0,1] =  AI*cosA
        GeomK[0,3] =  AI*sinA
        GeomK[0,4] = -AI*cosA
        GeomK[1,0] =  BI*sinA
        GeomK[1,1] = -BI*cosA
        GeomK[1,3] = -BI*sinA
        GeomK[1,4] =  BI*cosA

#        GeomK[3,0] = -AJ*sinA
#        GeomK[3,1] =  AJ*cosA
#        GeomK[3,3] =  AJ*sinA
#        GeomK[3,4] = -AJ*cosA
#        GeomK[4,0] =  BJ*sinA
#        GeomK[4,1] = -BJ*cosA
#        GeomK[4,3] = -BJ*sinA
#        GeomK[4,4] =  BJ*cosA

        GeomK[3,0] = -GeomK[0,0]
        GeomK[3,1] = -GeomK[0,1]
        GeomK[3,3] = -GeomK[0,3]
        GeomK[3,4] = -GeomK[0,4]
        GeomK[4,0] = -GeomK[1,0]
        GeomK[4,1] = -GeomK[1,1]
        GeomK[4,3] = -GeomK[1,3]
        GeomK[4,4] = -GeomK[1,4]
        return GeomK
    def FormX(self, r, s, t):
        X = array([ 0.5*(1-r), 0.5*(1+r)])
        return X
    def FormN(self, r, s, t):
        L = 2.*self.Geom[0,0]
        N = array([[0.5*(1-r),0.,0.,0.5*(1+r),0.,0.],
                   [0, 0.25*(r**3-3*r+2), L*0.125*(r**3-r**2-r+1),0, 0.25*(-r**3+3*r+2), L*0.125*(r**3+r**2-r-1)]])
        return N
    def FormB(self, r, s, t, NLg):
        L = 2.*self.Geom[0,0]
        B = array([[-1./L,0,0,1./L,0,0],
                   [0, 6*r/L**2, (3*r-1)/L, 0, -6*r/L**2, (3*r+1)/L]])
        return B, 1, 0
    def FormT(self, r, s, t):                              # for temperatures
        T = array([[ 0.5*(1-r), 0., 0., 0.5*(1+r), 0., 0.],
                   [ 0., 0.5*(1-r), 0., 0., 0.5*(1+r), 0.]])
        return T
    def FormP(self, r, s):                              # prestressing interpolation, 1st row: force, 2nd row: local vertical coordinate, 3rd row: local inclineation
        L = 2.*self.Geom[0,0]
        P = array([[0.5*(1-r), 0., 0., 0.5*(1+r), 0., 0.],
                   [0, 0.25*(r**3-3*r+2), L*0.125*(r**3-r**2-r+1),0, 0.25*(-r**3+3*r+2), L*0.125*(r**3+r**2-r-1)],
                   [0, 0.5* (3*r**2-3)/L, 0.25*(3*r**2-2*r-1),    0, 0.5* (-3*r**2+3)/L, 0.25*(3*r**2+2*r-1)]])
        return P
    def JacoD(self, r, s, t):
        """ Dummy for jacobian determinant.
        """
        return 1

class B21(Element):                                     # Timoshenko Beam 2D 2 nodes, linear shape
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, BeamSec, ReinfD, StateV, NData):
        Element.__init__(self,"B21", 2,6,3, 0,1,1, (set([1, 2, 6]),         set([1, 2, 6])),(3,3),   11, False)
        self.Label = Label                              # element number in input
#        self.DofI = zeros( (self.nNod,3), dtype=int)    # indices of global dofs per node
        self.DofI = ma.masked_array( [[-1,-1,-1],[-1,-1,-1]], dtype=int, mask=[[0,0,0],[0,0,0]])    # indices of global dofs per node
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.CrSecPolyL = None                          # polyline to describe cross section
        self.TensStiff = True                           # flag for tension stiffening
        self.InzList = [InzList[0], InzList[1]]
        self.IniBeam2( BeamSec, ReinfD, NData, StateV)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        L = sqrt( (NoList[i1].XCo-NoList[i0].XCo)**2 + (NoList[i1].YCo-NoList[i0].YCo)**2 )# element length
        self.Geom[0,0] = 0.5*L                          # Jacobi determinant value
        self.Lch_ = L                                    # characteristic length
        cosA = (NoList[i1].XCo-NoList[i0].XCo)/L        # direction cosine
        sinA = (NoList[i1].YCo-NoList[i0].YCo)/L        # direction sine
        if fabs(sinA) >= self.ZeroD:                    # element direction rotated to global axis
            self.Rot = True
            self.Trans = zeros((self.DofE, self.DofE), dtype=float)# coordinate transformation matrix
            for i in xrange(self.DofE): self.Trans[i,i] = cosA# fill transformation axis
            self.Trans[2,2] = 1.
            self.Trans[5,5] = 1.
            self.Trans[0,1] = sinA
            self.Trans[1,0] = -sinA
            self.Trans[3,4] = sinA
            self.Trans[4,3] = -sinA
        self.Inzi = [ i0, i1]                           # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
    def FormX(self, r, s, t):
        X = array([ 0.5*(1-r), 0.5*(1+r)])
        return X
    def FormN(self, r, s, t):
        N = array([[0.5*(1-r),0.,0.,0.5*(1+r),0.,0.],
                   [0.,0.5*(1-r),0.,0.,0.5*(1+r),0.],
                   [0.,0.,0.5*(1-r),0.,0.,0.5*(1+r)]])
        return N
    def FormB(self, r, s, t, NLg):
        L = 2.*self.Geom[0,0]
        B = array([[-1./L,0,0,1./L,0,0],
                   [0,0,-1./L,0,0,1./L],
                   [0,-1./L,-0.5*(1-r),0,1./L,-0.5*(1+r)]])
        return B, 1, 0
    def FormT(self, r, s, t):                              # for temperatures
        T = array([[ 0.5*(1-r), 0., 0., 0.5*(1+r), 0., 0.],
                   [ 0., 0.5*(1-r), 0., 0., 0.5*(1+r), 0.]])
        return T
    def JacoD(self, r, s, t):
        """ Dummy for jacobian determinant.
        """
        return 1

class B21E(Element):                                    # Timoshenko Beam 2D 2 nodes, linear shape quadratic shape for displacements
    dim  = 11                                           # material type index, timoshenko beam
    NLGeomI = False                                     # Flag whether large deformations are already implemented for this element
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, BeamSec, ReinfD, StateV, NData):
        Element.__init__(self,"B21E",3,9,3, 0,2,2, (set([1, 2, 6]),set([1, 2, 6]),set([1, 2, 6])),(3,3,3),   11,False)
#        Element.__init__(self,"B21", 2,6,3, 0,1,1, (set([1, 2, 6]),         set([1, 2, 6])),(3,3),   11,False)
#        Element.__init__(self,"B23E",3,7,2, 0,3,3, (set([1, 2, 6]),set([1]),set([1, 2, 6])),(3,1,3), 10,False)
        self.Label = Label                              # element number in input
#        self.DofI = zeros( (self.nNod,3), dtype=int)    # indices of global dofs per node
        self.DofI = ma.masked_array( [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]], dtype=int, mask=[[0,0,0],[0,0,0],[0,0,0]])    # indices of global dofs per node
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.CrSecPolyL = None                          # polyline to describe cross section
        self.TensStiff = True                           # flag for tension stiffening
        self.InzList = [InzList[0], InzList[1], InzList[2]]
        self.IniBeam2( BeamSec, ReinfD, NData, StateV)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        i2 = FindIndexByLabel( NoList, self.InzList[2])
        L = sqrt( (NoList[i2].XCo-NoList[i0].XCo)**2 + (NoList[i2].YCo-NoList[i0].YCo)**2 )# element length
        self.Geom[0,0] = 0.5*L                          # Jacobi determinant value
        self.Lch = L                                    # characteristic length
        cosA = (NoList[i2].XCo-NoList[i0].XCo)/L        # direction cosine
        sinA = (NoList[i2].YCo-NoList[i0].YCo)/L        # direction sine
        if fabs(sinA) >= self.ZeroD:                    # element direction rotated to global axis
            self.Rot = True
            self.Trans = zeros((self.DofE, self.DofE), dtype=float)# coordinate transformation matrix
            for i in xrange(self.DofE): self.Trans[i,i] = cosA# fill transformation axis
            self.Trans[0,1] = sinA
            self.Trans[1,0] = -sinA
            self.Trans[2,2] = 1.
            self.Trans[3,4] = sinA
            self.Trans[4,3] = -sinA
            self.Trans[5,5] = 1.
            self.Trans[6,7] = sinA
            self.Trans[7,6] = -sinA
            self.Trans[8,8] = 1.
        self.Inzi = ( i0, i1, i2 )                      # tupel, indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        NoList[i2].DofT = NoList[i2].DofT.union(self.DofT[2])
    def FormX(self, r, s, t):
        X = array([ 0.5*(1-r), 0., 0.5*(1+r)])
        return X
    def FormN(self, r, s, t):
        N = array([[0.5*(r**2-r),0.,          0.,           1-r**2,   0.,    0.,       0.5*(r**2+r),0.,          0.],
                   [0.,          0.5*(r**2-r),0.,           0.,       1-r**2,0.,       0.,          0.5*(r**2+r),0.],
                   [0.,          0.,          0.5*(r**2-r), 0.,       0.,    1-r**2,   0.,          0.,          0.5*(r**2+r)]])
        return N
    def FormB(self, r, s, t, NLg):
        L = 2.*self.Geom[0,0]
        B = array([[(2*r-1)/L,   0.,          0.,          -4*r/L,    0.,    0.,      (2*r+1)/L,    0.,          0.],
                   [ 0.,         0.,         (2*r-1)/L,     0.,       0.,   -4*r/L,    0.,          0.,         (2*r+1)/L],
                   [ 0.,        (2*r-1)/L,   -0.5*(r**2-r), 0.,      -4*r/L,-(1-r**2), 0.,         (2*r+1)/L,   -0.5*(r**2+r)]])
        return B, 1, 0
    def FormT(self, r, s, t):                                 # for temperatures
        T = array([[ 0.5*(1-r), 0., 0., 0., 0., 0., 0.5*(1+r), 0., 0.],
                   [ 0., 0.5*(1-r), 0., 0., 0., 0., 0., 0.5*(1+r), 0.]])
        return T
    def JacoD(self, r, s, t):
        """ Dummy for jacobian determinant.
        """
        return 1

class T3D2( Element ):                                  # 1D Truss 2 nodes
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, SolSecDic, StateV, NData, Val):
        # Element.__init__((self, TypeVal,nNodVal,DofEVal,nFieVal, IntTVal,nIntVal,nIntLVal, DofTVal,DofNVal, dimVal,NLGeomIVal)
        Element.__init__(self,"T3D2",2,6,3, 0,1,1, (set([1,2,3]),set([1,2,3])),(3,3), 1,True)
        self.Label = Label                              # element number in input
        self.DofI = zeros( (self.nNod,3), dtype=int)    # indices of global dofs per node
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.TensStiff = False                          # flag for tension stiffening
        self.InzList = [InzList[0], InzList[1]]
        self.Geom = zeros( (2,5), dtype=double)
        self.Geom[1,0] = SolSecDic.Val*Val              # cross section area
        self.Data = zeros((self.nIntL,NData), dtype=float)# storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)# storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        self.CoordRef = array( [NoList[i0].XCo, NoList[i0].YCo, NoList[i0].ZCo, NoList[i1].XCo, NoList[i1].YCo, NoList[i1].ZCo]) # Nodal coordinates 
        self.CoordDis = array( [NoList[i0].XCo, NoList[i0].YCo, NoList[i0].ZCo, NoList[i1].XCo, NoList[i1].YCo, NoList[i1].ZCo]) # Nodal coordinates displaced 
        self.Inzi = [ i0, i1]                           # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        dx = self.CoordRef[3]-self.CoordRef[0]
        dy = self.CoordRef[4]-self.CoordRef[1]
        dz = self.CoordRef[5]-self.CoordRef[2]
        L = sqrt(dx**2 + dy**2 + dz**2)
        self.Lch_ = L                         # length in undeformed configuration !
        self.Geom[0,0] = 0.5*L                          # Jacobian
        self.Geom[1,1] = dx
        self.Geom[1,2] = dy
        self.Geom[1,3] = dz
        self.Geom[1,4] = L*L
    def UpdateCoord(self, dis, ddis ):
        self.CoordDis[0] = self.CoordRef[0]+dis[0]
        self.CoordDis[1] = self.CoordRef[1]+dis[1]
        self.CoordDis[2] = self.CoordRef[2]+dis[2]
        self.CoordDis[3] = self.CoordRef[3]+dis[3]
        self.CoordDis[4] = self.CoordRef[4]+dis[4]
        self.CoordDis[5] = self.CoordRef[5]+dis[5]
        self.Geom[1,1] = self.CoordDis[3]-self.CoordDis[0]
        self.Geom[1,2] = self.CoordDis[4]-self.CoordDis[1]
        self.Geom[1,3] = self.CoordDis[5]-self.CoordDis[2]
        return
    def GeomStiff(self, r, s, t, sig):
        GeomK = zeros((self.DofE, self.DofE), dtype=float)
        val = sig[0]/self.Geom[1,4]
        GeomK[0,0] = val
        GeomK[0,3] = -val
        GeomK[1,1] = val
        GeomK[1,4] = -val
        GeomK[2,2] = val
        GeomK[2,5] = -val
        GeomK[3,0] = -val
        GeomK[3,3] = val
        GeomK[4,1] = -val
        GeomK[4,4] = val
        GeomK[5,2] = -val
        GeomK[5,5] = val 
        return GeomK
    def FormN(self, r, s, t):
        N = array([[ 0.5*(1-r), 0., 0., 0.5*(1+r), 0., 0.],
                   [ 0., 0.5*(1-r), 0., 0., 0.5*(1+r), 0.],
                   [ 0., 0., 0.5*(1-r), 0., 0., 0.5*(1+r)]])
        return N
    def FormB(self, r, s, t, NLg):
        ll = self.Geom[1,1]/self.Geom[1,4]              # Jacobian incorporated
        mm = self.Geom[1,2]/self.Geom[1,4]
        nn = self.Geom[1,3]/self.Geom[1,4]
        B = array([[ -ll, -mm, -nn, ll, mm, nn], [ 0, 0, 0, 0, 0, 0]])
        return B, 1, 0
    def FormX(self, r, s, t):
        X = array([ 0.5*(1-r), 0.5*(1+r)])
        return X
    def FormT(self, r, s, t):
        T = array([ 0.5*(1-r),0.5*(1-r),0.5*(1-r), 0.5*(1+r),0.5*(1+r),0.5*(1+r)])
        return T
    def JacoD(self, r, s, t):
        """ Dummy for jacobian determinant.
        """
        return 1

class T2D2( Element ):                                  # 1D Truss 2 nodes
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, SolSecDic, StateV, NData, Val):
        # Element.__init__((self, TypeVal,nNodVal,DofEVal,nFieVal, IntTVal,nIntVal,nIntLVal, DofTVal,DofNVal, dimVal,NLGeomIVal)
        Element.__init__(self,"T2D2",2,4,1, 0,1,1, (set([1,2]),set([1,2])),(2,2), 1,True)
        self.Label = Label                              # element number in input
        self.DofI = zeros( (self.nNod,2), dtype=int)    # indices of global dofs per node
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.TensStiff = False                          # flag for tension stiffening
        self.InzList = [InzList[0], InzList[1]]
        self.Geom = zeros( (2,4), dtype=double)
        self.Geom[1,0] = SolSecDic.Val*Val              # cross section area
        self.Data = zeros((self.nIntL,NData), dtype=float)# storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)# storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        self.CoordRef = array( [NoList[i0].XCo, NoList[i0].YCo, NoList[i1].XCo, NoList[i1].YCo]) # Nodal coordinates 
        self.CoordDis = array( [NoList[i0].XCo, NoList[i0].YCo, NoList[i1].XCo, NoList[i1].YCo]) # Nodal coordinates displaced 
        self.Inzi = [ i0, i1]                           # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        dx = self.CoordRef[2]-self.CoordRef[0]
        dy = self.CoordRef[3]-self.CoordRef[1]
        L = sqrt(dx**2 + dy**2)
        self.Geom[0,0] = 0.5*L                          # Jacobian
        self.Geom[1,1] = dx
        self.Geom[1,2] = dy
        self.Geom[1,3] = L*L
        self.Lch_ = L                         # length in undeformed configuration !
    def UpdateCoord(self, dis, ddis ):
        self.CoordDis[0] = self.CoordRef[0]+dis[0]
        self.CoordDis[1] = self.CoordRef[1]+dis[1]
        self.CoordDis[2] = self.CoordRef[2]+dis[2]
        self.CoordDis[3] = self.CoordRef[3]+dis[3]
        self.Geom[1,1] = self.CoordDis[2]-self.CoordDis[0]
        self.Geom[1,2] = self.CoordDis[3]-self.CoordDis[1]
        return
    def GeomStiff(self, r, s, t, sig):
        GeomK = zeros((self.DofE, self.DofE), dtype=float)
        val = sig[0]/self.Geom[1,3]
        GeomK[0,0] = val
        GeomK[0,2] = -val
        GeomK[1,1] = val
        GeomK[1,3] = -val
        GeomK[2,0] = -val
        GeomK[2,2] = val
        GeomK[3,1] = -val
        GeomK[3,3] = val
        return GeomK
    def FormX(self, r, s, t):
        X = array([ 0.5*(1-r), 0.5*(1+r)])
        return X
    def FormN(self, r, s, t):
        N = array([[ 0.5*(1-r), 0., 0.5*(1+r), 0.],
                   [ 0., 0.5*(1-r), 0., 0.5*(1+r)]])
#        N = array([[ 0.5*(1-r), 0.5*(1+r)],[ 0, 0]])
        return N
    def FormB(self, r, s, t, NLg):                      # Jacobian incorporated
        cc = self.Geom[1,1]/self.Geom[1,3]
        ss = self.Geom[1,2]/self.Geom[1,3]
        B = array([[ -cc, -ss, cc, ss], [ 0, 0, 0, 0]])
        return B, 1, 0
    def FormT(self, r, s, t):
        T = array([ 0.5*(1-r), 0.5*(1+r), 0, 0])
        return T
    def JacoD(self, r, s, t):
        """ Dummy for jacobian determinant.
        """
        return 1

class T1D2(Element):
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, SolSecDic, StateV, NData, RegType):
        # Element.__init__((self, TypeVal,nNodVal,DofEVal,nFieVal, IntTVal,nIntVal,nIntLVal, DofTVal,DofNVal, dimVal,NLGeomIVal)
        if RegType==1: Element.__init__(self,"T1D2",2,4,1, 0,1,1, (set([1,7]),set([1,7])),(2,2), 1,False)
        else:          Element.__init__(self,"T1D2",2,2,1, 0,1,1, (set([1]),set([1])),    (1,1), 1,False)
        self.Label = Label                              # element number in input
        if RegType==1: self.DofI = zeros( (self.nNod,2), dtype=int)
        else:          self.DofI = zeros( (self.nNod,1), dtype=int) # indices of global dofs per node
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.TensStiff = False                          # flag for tension stiffening
        self.RegType = RegType                          # 0 for no regularization
        self.InzList = [InzList[0], InzList[1]]
        self.Geom = zeros( (2,1), dtype=double)
#        self.Geom[0,0] = 0.5*L                         # Jacobi determinant value / length measure
        self.Geom[1,0] = SolSecDic.Val
        self.Data = zeros((self.nIntL,NData), dtype=float)# storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)# storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0]) # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        L = sqrt( (NoList[i1].XCo-NoList[i0].XCo)**2 + (NoList[i1].YCo-NoList[i0].YCo)**2 )# element length
        self.Geom[0,0] = 0.5*L                          # Jacobi determinant value / length measure
        self.Lch_ = L                                    # length in undeformed configuration !
        self.Inzi = [ i0, i1]                           # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        self.CrBwS=1.
    def FormX(self, r, s, t):
        X = array([ 0.5*(1-r), 0.5*(1+r)])
        return X
    def FormN(self, r, s, t):
        N = array([[ 0.5*(1-r), 0.5*(1+r)],[ 0, 0]])
        return N
    def FormB(self, r, s, t, NLg):
        L = 2.*self.Geom[0,0]
        if self.RegType==1: 
            B = array([[ -1./L, 0, 1./L, 0], [ 0, -1./L, 0, 1./L]])
            BN= array([[ -1./L, 0, 1./L, 0], [ 0, 0.5*(1-r), 0, 0.5*(1+r)]])
            return B, BN, 1, 0
        else:               
            B = array([[ -1./L, 1./L], [ 0, 0]])
            return B, 1, 0
    def FormT(self, r, s, t):
        if self.RegType==1: T = array([ 0.5*(1-r), 0, 0.5*(1+r), 0])
        else:               T = array([ 0.5*(1-r), 0.5*(1+r)])
        return T
    def JacoD(self, r, s, t):
        """ Dummy for jacobian determinant.
        """
        return 1

class S1D2(Element):                                    # 1D Spring 2 nodes
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, SolSecDic, StateV, NData):
        # Element.__init__((self, TypeVal,nNodVal,DofEVal,nFieVal, IntTVal,nIntVal,nIntLVal, DofTVal,DofNVal, dimVal,NLGeomIVal)
        Element.__init__(self,"S1D2",2,2,1, 0,1,1, (set([1]),set([1])),(1,1), 99,False)
        self.Label = Label                              # element number in input
        self.DofI = zeros( (self.nNod,1), dtype=int)    # indices of global dofs per node
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.InzList = [InzList[0], InzList[1]]
        self.Geom = zeros( (2,1), dtype=double)
        self.Geom[0,0] = 0.5                           # Jacobi determinant value
        self.Geom[1,0] = SolSecDic.Val
        self.Data = zeros((self.nIntL,NData), dtype=float)# storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)# storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        self.Lch_ = 0                                    # dummy
        self.Inzi = [ i0, i1]                           # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
    def FormX(self, r, s, t):
        X = array([ 0.5, 0.5])
        return X
    def FormN(self, r, s, t):
        N = array([[ -1., 1.], [ 0, 0]])
        return N
    def FormB(self, r, s, t, NLg):
        B = array([[ -1., 1.], [ 0, 0]])
        return B, 1, 0
    def FormT(self, r, s, t):                              # interpolation on temperature
        T = array([ 0., 0.])
        return T
    def JacoD(self, r, s, t):
        """ Dummy for jacobian determinant.
        """
        return 1

class S2D6(Element):                                    # 2D rotational Spring 2 nodes
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, SolSecDic, StateV, NData):
        Element.__init__(self,"S2D6",2,6,3, 0,1,1, (set([1,2,6]),set([1,2,6])),(3,3), 98,False)
        self.Label = Label                              # element number in input
        self.DofI = zeros( (self.nNod,3), dtype=int)    # indices of global dofs per node
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.InzList = [InzList[0], InzList[1]]
        self.Geom = zeros( (2,1), dtype=double)
        self.Geom[0,0] = 0.5                           # Jacobi determinant value
        self.Geom[1,0] = SolSecDic.Val
        self.Data = zeros((self.nIntL,NData), dtype=float)# storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)# storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        self.Lch = 0                                    # dummy
        self.Inzi = [ i0, i1]                           # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
    def FormX(self, r, s, t):
        X = array([ 0.5, 0.5])
        return X
    def FormN(self, r, s, t):
        N = array([[ -1.,  0.,  0., 1., 0., 0.],
                   [  0., -1.,  0., 0., 1., 0.], 
                   [  0.,  0., -1., 0., 0., 1.]])
        return N
    def FormB(self, r, s, t, NLg):
        B = array([[ -1.,  0.,  0., 1., 0., 0.],
                   [  0., -1.,  0., 0., 1., 0.], 
                   [  0.,  0., -1., 0., 0., 1.]])
        return B, 1, 0
    def FormT(self, r, s, t):                           # interpolation on temperature -- here dummy, actually not interpolated
        T = array([[ 0., 0., 0., 0., 0., 0.], [ 0., 0., 0., 0., 0., 0.], [ 0., 0., 0., 0., 0., 0.]])
        return T
    def JacoD(self, r, s, t):
        """ Dummy for jacobian determinant.
        """
        return 1

class CPE3(Element):
    NLGeomI = False                                     # Flag whether large deformations are already implemented for this element
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, SolSecDic, StateV, NData, PlSt, nI):
        if PlSt: XXX = "CPS3"
        else:    XXX = "CPE3"
        Element.__init__(self,XXX,3,6,2, 3,1,1, (set([1,2]),set([1,2]),set([1,2])),(2,2,2), 2,False)
        self.PlSt = PlSt                                # flag for plane stress (True->plane stress, False->plane strain)
        self.Label = Label                              # element number in input
        self.DofI = zeros( (self.nNod,2), dtype=int)    # indices of global dofs per node
        self.TensStiff = False                          # flag for tension stiffening
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.InzList = [InzList[0], InzList[1], InzList[2]]
        self.Geom = zeros( (2,2), dtype=double)
        self.Geom[1,0] = SolSecDic.Val                  # thickness
        self.Data = zeros((self.nIntL,NData), dtype=float)# storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)# storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        i2 = FindIndexByLabel( NoList, self.InzList[2])      # find node index from node label
        self.Inzi = [ i0, i1, i2]                       # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        NoList[i2].DofT = NoList[i2].DofT.union(self.DofT[2])
        X0 = NoList[i0].XCo
        Y0 = NoList[i0].YCo
        X1 = NoList[i1].XCo
        Y1 = NoList[i1].YCo
        X2 = NoList[i2].XCo
        Y2 = NoList[i2].YCo
        self.AA = -Y0*X1+Y0*X2+Y2*X1+Y1*X0-Y1*X2-Y2*X0 # double of element area
        if self.AA<=0.: raise NameError("Something is wrong with this CPE3-element")
        self.Geom[0,0] = 0.5*self.AA                    # element area for numerical integration
        self.Lch_ = sqrt(self.AA)              # characteristic length
#        self.Lch = sqrt(self.AA) /self.nInt              # characteristic length - presumably obsolete
        self.b1=(Y1-Y2)/self.AA
        self.b2=(Y2-Y0)/self.AA
        self.b3=(Y0-Y1)/self.AA
        self.c1=(X2-X1)/self.AA
        self.c2=(X0-X2)/self.AA
        self.c3=(X1-X0)/self.AA
    def FormN(self, L1, L2, L3):
        L3=1-L1-L2
        N = array([[L1,0,L2,0,L3,0],[0,L1,0,L2,0,L3]])        
        return N
    def FormB(self, L1, L2, L3, NLg):
        c1=self.c1
        c2=self.c2
        c3=self.c3
        b1=self.b1
        b2=self.b2
        b3=self.b3
        B =array([[b1,0,b2,0,b3,0],[0,c1,0,c2,0,c3],[c1,b1,c2,b2,c3,b3]])
        return B, 1, 0 
    def FormT(self, L1, L2, L3):
        L3=1-L1-L2
        T = array([L1, 0, L2, 0, L3, 0])
        return T
    def FormX(self, L1, L2, L3):
        L3=1-L1-L2
        X = array([L1, L2, L3])
        return X
    def JacoD(self, r, s, t):
        return 1

class CPE4( Element ): 
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, SolSecDic, StateV, NData, PlSt, nI, RegType):
        if PlSt: XXX = "CPS4"  # plane stress
        else:    XXX = "CPE4"  # plane strain
        if RegType==1: Element.__init__(self,XXX,4,12,2, 1,2,4, (set([1,2,7]),set([1,2,7]),set([1,2,7]),set([1,2,7])),(3,3,3,3), 2,False)
        else:          Element.__init__(self,XXX,4,8, 2, 1,2,4, (set([1,2]),  set([1,2]),  set([1,2]),  set([1,2])),  (2,2,2,2), 2,False)
        self.PlSt = PlSt                                # flag for plane stress (True->plane stress, False->plane strain)
        self.Label = Label                              # element number in input
        if nI<>2:
            self.nInt = nI                              # integration order
            self.nIntL= nI*nI                           # total number of integration points
        if RegType==1: self.DofI = zeros( (self.nNod,3), dtype=int)
        else:          self.DofI = zeros( (self.nNod,2), dtype=int)    # indices of global dofs per node
        self.TensStiff = False                          # flag for tension stiffening
        self.MatN = MatName                             # name of material
        self.RegType = RegType                          # 0 for no regularization
        self.Set = SetLabel                             # label of corresponding element set
        self.InzList = [InzList[0], InzList[1], InzList[2], InzList[3]]
        self.Geom = zeros( (2,1), dtype=double)
        self.Geom[0,0] = 1.                             # dummy for Area / Jacobi determinant used instead
        self.Geom[1,0] = SolSecDic.Val                  # thickness
        self.Data = zeros((self.nIntL,NData), dtype=float)# storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)# storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        i2 = FindIndexByLabel( NoList, self.InzList[2])      # find node index from node label
        i3 = FindIndexByLabel( NoList, self.InzList[3])
        self.Inzi = [ i0, i1, i2, i3]                   # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        NoList[i2].DofT = NoList[i2].DofT.union(self.DofT[2])
        NoList[i3].DofT = NoList[i3].DofT.union(self.DofT[3])
        self.X0 = NoList[i0].XCo
        self.Y0 = NoList[i0].YCo
        self.X1 = NoList[i1].XCo
        self.Y1 = NoList[i1].YCo
        self.X2 = NoList[i2].XCo
        self.Y2 = NoList[i2].YCo
        self.X3 = NoList[i3].XCo
        self.Y3 = NoList[i3].YCo
        self.AA = 0.5*( (self.X1-self.X3)*(self.Y2-self.Y0) + (self.X2-self.X0)*(self.Y3-self.Y1)) # element area
        if self.AA<=0.: raise NameError("Something is wrong with this CPE4-element")
        self.Lch = sqrt(self.AA)/self.nInt              # characteristic length
        self.Lch_ = sqrt(self.AA)                       # previous approach might be misleading
        if MaList[self.MatN].RType ==2:                 # find scaling factor for band width regularization 
            x = MaList[self.MatN].bw/self.Lch_
            CrX, CrY = MaList[self.MatN].CrX, MaList[self.MatN].CrY # support points for tensile softening scaling factor 
            i = bisect_left(CrX, x) 
            if i>0 and i<(MaList[self.MatN].CrBwN+1): 
                self.CrBwS = CrY[i-1] + (x-CrX[i-1])/(CrX[i]-CrX[i-1])*(CrY[i]-CrY[i-1]) # scling factor by linear interpolation
            else:
                print 'ZZZ', self.Lch_,x,CrX, CrY, i, MaList[self.MatN].CrBwN  
                raise NameError("ConFemElem:CPE4.Ini2: RType 2 - element char length exceeds scaling factor interpolation")
        else: self.CrBwS=1.
    def FormN(self, r, s, t):
        N = array([[(1+r)*(1+s)*0.25, 0, (1-r)*(1+s)*0.25, 0, (1-r)*(1-s)*0.25, 0, (1+r)*(1-s)*0.25, 0],
                   [0, (1+r)*(1+s)*0.25, 0, (1-r)*(1+s)*0.25, 0, (1-r)*(1-s)*0.25, 0, (1+r)*(1-s)*0.25]])
        return N
    def FormB(self, r, s, t, NLg):
        h00= ( 1+s)*0.25
        h01= ( 1+r)*0.25
        h10=-( 1+s)*0.25
        h11= ( 1-r)*0.25
        h20= (-1+s)*0.25
        h21= (-1+r)*0.25
        h30= ( 1-s)*0.25
        h31=-( 1+r)*0.25
        JAC00 = h00*self.X0 + h10*self.X1 + h20*self.X2 + h30*self.X3
        JAC01 = h00*self.Y0 + h10*self.Y1 + h20*self.Y2 + h30*self.Y3
        JAC10 = h01*self.X0 + h11*self.X1 + h21*self.X2 + h31*self.X3
        JAC11 = h01*self.Y0 + h11*self.Y1 + h21*self.Y2 + h31*self.Y3
        det = JAC00*JAC11 - JAC01*JAC10
        deti = 1./det
        a1 = self.Y0*h01 + self.Y1*h11 + self.Y2*h21 + self.Y3*h31
        a2 = self.Y0*h00 + self.Y1*h10 + self.Y2*h20 + self.Y3*h30
        b1 = self.X0*h01 + self.X1*h11 + self.X2*h21 + self.X3*h31
        b2 = self.X0*h00 + self.X1*h10 + self.X2*h20 + self.X3*h30
        B00 = deti*( h00 * a1 - h01 * a2 )
        B10 = deti*( h10 * a1 - h11 * a2 )
        B20 = deti*( h20 * a1 - h21 * a2 )
        B30 = deti*( h30 * a1 - h31 * a2 )
        B01 = deti*(-h00 * b1 + h01 * b2 )
        B11 = deti*(-h10 * b1 + h11 * b2 )
        B21 = deti*(-h20 * b1 + h21 * b2 )
        B31 = deti*(-h30 * b1 + h31 * b2 )
        if self.RegType==1:
            N0 = (1+r)*(1+s)*0.25 
            N1 = (1-r)*(1+s)*0.25 
            N2 = (1-r)*(1-s)*0.25 
            N3 = (1+r)*(1-s)*0.25
            B = array([[ B00, 0,   0,   B10, 0,   0,   B20, 0,   0,   B30, 0   ,0  ],
                       [ 0,   B01, 0,   0,   B11, 0,   0,   B21, 0,   0,   B31, 0  ],
                       [ B01, B00, 0,   B11, B10, 0,   B21, B20, 0,   B31, B30, 0  ],
                       [ 0,   0,   B00, 0,   0,   B10, 0,   0,   B20, 0,   0,   B30],
                       [ 0,   0,   B01, 0,   0,   B11, 0,   0,   B21, 0,   0,   B31]])
            BN= array([[ B00, 0,   0,   B10, 0,   0,   B20, 0,   0,   B30, 0   ,0  ],
                       [ 0,   B01, 0,   0,   B11, 0,   0,   B21, 0,   0,   B31, 0  ],
                       [ B01, B00, 0,   B11, B10, 0,   B21, B20, 0,   B31, B30, 0  ],
                       [ 0,   0,   N0,  0,   0,   N1,  0,   0,   N2,  0,   0,   N3 ]])
            return B, BN, det, 0
        else:
            B = array([[ B00, 0,   B10, 0,   B20, 0,   B30, 0  ],
                       [ 0,   B01, 0,   B11, 0,   B21, 0,   B31],
                       [ B01, B00, B11, B10, B21, B20, B31, B30]])
            return B, det, 0
    def FormT(self, r, s, t):                                 # interpolation on temperature
        # !!! ordering might not yet be correct !!!
        if self.RegType==1: T = array([(1+r)*(1+s)*0.25, (1-r)*(1+s)*0.25, (1-r)*(1-s)*0.25, (1+r)*(1-s)*0.25, 0, 0, 0, 0, 0, 0, 0, 0])
        else:               T = array([(1+r)*(1+s)*0.25, (1-r)*(1+s)*0.25, (1-r)*(1-s)*0.25, (1+r)*(1-s)*0.25, 0, 0, 0, 0])
        return T
    def FormX(self, r, s, t):
        X = array([(1+r)*(1+s)*0.25, (1-r)*(1+s)*0.25, (1-r)*(1-s)*0.25, (1+r)*(1-s)*0.25])
        return X
    def JacoD(self, r, s, t):
	    h00= ( 1+s)*0.25
	    h01= ( 1+r)*0.25
	    h10=-( 1+s)*0.25
	    h11= ( 1-r)*0.25
	    h20= (-1+s)*0.25
	    h21= (-1+r)*0.25
	    h30= ( 1-s)*0.25
	    h31=-( 1+r)*0.25
	    JAC00 = h00*self.X0 + h10*self.X1 + h20*self.X2 + h30*self.X3
	    JAC01 = h00*self.Y0 + h10*self.Y1 + h20*self.Y2 + h30*self.Y3
	    JAC10 = h01*self.X0 + h11*self.X1 + h21*self.X2 + h31*self.X3
	    JAC11 = h01*self.Y0 + h11*self.Y1 + h21*self.Y2 + h31*self.Y3
	    return JAC00*JAC11 - JAC01*JAC10
    
class C3D8(Element):    # volumic cube element
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, StateV, NData, RegType):
        # Element.__init__((self, TypeVal,nNodVal,DofEVal,nFieVal, IntTVal,nIntVal,nIntLVal, DofTVal,DofNVal, dimVal,NLGeomIVal)
        if RegType==1: Element.__init__(self,"C3D8",8,32,3, 2,2,8, (set([1,2,3,7]),set([1,2,3,7]),set([1,2,3,7]),set([1,2,3,7]),set([1,2,3,7]),set([1,2,3,7]),set([1,2,3,7]),set([1,2,3,7])),(4,4,4,4,4,4,4,4), 3,True)
        else:          Element.__init__(self,"C3D8",8,24,3, 2,2,8, (set([1,2,3]),  set([1,2,3]),  set([1,2,3]),  set([1,2,3]),  set([1,2,3]),  set([1,2,3]),  set([1,2,3]),  set([1,2,3])),  (3,3,3,3,3,3,3,3), 3,True)
        self.Label = Label                              # element number in input
        if RegType==1: self.DofI = zeros( (self.nNod,4), dtype=int)
        else:          self.DofI = zeros( (self.nNod,3), dtype=int) # indices of global dofs per node
        self.TensStiff = False                          # flag for tension stiffening
        self.PlSt = False                               # flag for plane stress
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.RegType = RegType                          # 0 for no regularization
        self.InzList = [InzList[0], InzList[1], InzList[2], InzList[3], InzList[4], InzList[5], InzList[6], InzList[7]]
        self.Geom = zeros( (2,2), dtype=double)
        self.Geom[0,0] = 1.                             # dummy
        self.Geom[1,0] = 1.                             # dummy
        self.Data = zeros((self.nIntL,NData+2), dtype=float)# storage for element data
        self.DataP= zeros((self.nIntL,NData+2), dtype=float)# storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        i2 = FindIndexByLabel( NoList, self.InzList[2])      # find node index from node label
        i3 = FindIndexByLabel( NoList, self.InzList[3])      # find node index from node label
        i4 = FindIndexByLabel( NoList, self.InzList[4])
        i5 = FindIndexByLabel( NoList, self.InzList[5])      # find node index from node label
        i6 = FindIndexByLabel( NoList, self.InzList[6])
        i7 = FindIndexByLabel( NoList, self.InzList[7])      # find node index from node label
        self.Inzi = [ i0, i1, i2, i3, i4, i5, i6, i7]                       # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        NoList[i2].DofT = NoList[i2].DofT.union(self.DofT[2])
        NoList[i3].DofT = NoList[i3].DofT.union(self.DofT[3])# attach element dof types to node dof types
        NoList[i4].DofT = NoList[i4].DofT.union(self.DofT[4])
        NoList[i5].DofT = NoList[i5].DofT.union(self.DofT[5])
        NoList[i6].DofT = NoList[i6].DofT.union(self.DofT[6])
        NoList[i7].DofT = NoList[i7].DofT.union(self.DofT[7])
        self.X0, self.Y0, self.Z0 = NoList[i0].XCo, NoList[i0].YCo, NoList[i0].ZCo
        self.X1, self.Y1, self.Z1 = NoList[i1].XCo, NoList[i1].YCo, NoList[i1].ZCo
        self.X2, self.Y2, self.Z2 = NoList[i2].XCo, NoList[i2].YCo, NoList[i2].ZCo
        self.X3, self.Y3, self.Z3 = NoList[i3].XCo, NoList[i3].YCo, NoList[i3].ZCo
        self.X4, self.Y4, self.Z4 = NoList[i4].XCo, NoList[i4].YCo, NoList[i4].ZCo
        self.X5, self.Y5, self.Z5 = NoList[i5].XCo, NoList[i5].YCo, NoList[i5].ZCo
        self.X6, self.Y6, self.Z6 = NoList[i6].XCo, NoList[i6].YCo, NoList[i6].ZCo
        self.X7, self.Y7, self.Z7 = NoList[i7].XCo, NoList[i7].YCo, NoList[i7].ZCo
        self.X0u = self.X0 
        self.X1u = self.X1 
        self.X2u = self.X2 
        self.X3u = self.X3 
        self.X4u = self.X4 
        self.X5u = self.X5 
        self.X6u = self.X6 
        self.X7u = self.X7 
        self.Y0u = self.Y0 
        self.Y1u = self.Y1 
        self.Y2u = self.Y2 
        self.Y3u = self.Y3 
        self.Y4u = self.Y4 
        self.Y5u = self.Y5 
        self.Y6u = self.Y6 
        self.Y7u = self.Y7 
        self.Z0u = self.Z0 
        self.Z1u = self.Z1 
        self.Z2u = self.Z2 
        self.Z3u = self.Z3 
        self.Z4u = self.Z4 
        self.Z5u = self.Z5 
        self.Z6u = self.Z6 
        self.Z7u = self.Z7 

        AA, it, io = 0., self.IntT, self.nInt-1                                 # integration type, integration order
        for i in xrange(self.nIntLi):                        # volume by numerical integration for characteristic length  
            r = SamplePoints[it,io,i][0]
            s = SamplePoints[it,io,i][1]
            t = SamplePoints[it,io,i][2]
            JJ = self.JacoD(r,s,t) 
            if JJ<= 0.0: raise NameError("ConFemElements::C3D84.Ini2: something wrong with geometry")
            f = JJ*SampleWeight[it,io,i]
            AA = AA + f
        self.Lch = AA**(1./3.)/self.nInt                     # characteristic length
        self.Lch_ = AA**(1./3.)                              # previous approach might be misleading
        mat = MaList[self.MatN]
        if mat.RType ==2:                                    # find scaling factor for band width regularization 
            x = mat.bw/self.Lch_
            CrX, CrY = mat.CrX, mat.CrY                      # support points for tensile softening scaling factor 
            i = bisect_left(CrX, x) 
            if i>0 and i<(mat.CrBwN+1):
                self.CrBwS = CrY[i-1] + (x-CrX[i-1])/(CrX[i]-CrX[i-1])*(CrY[i]-CrY[i-1]) # scling factor by linear interpolation
            else:
                print 'ZZZ', self.Lch_,x,'\n', CrX, '\n', CrY, i, mat.CrBwN  
                raise NameError("ConFemElem:C3D8.Ini2: RType 2 - element char length exceeds scaling factor interpolation")
    def UpdateCoord(self, dis, ddis ):
        self.X0 = self.X0u + dis[0] 
        self.Y0 = self.Y0u + dis[1] 
        self.Z0 = self.Z0u + dis[2] 
        self.X1 = self.X1u + dis[3] 
        self.Y1 = self.Y1u + dis[4] 
        self.Z1 = self.Z1u + dis[5] 
        self.X2 = self.X2u + dis[6] 
        self.Y2 = self.Y2u + dis[7] 
        self.Z2 = self.Z2u + dis[8] 
        self.X3 = self.X3u + dis[9] 
        self.Y3 = self.Y3u + dis[10] 
        self.Z3 = self.Z3u + dis[11] 
        self.X4 = self.X4u + dis[12] 
        self.Y4 = self.Y4u + dis[13] 
        self.Z4 = self.Z4u + dis[14] 
        self.X5 = self.X5u + dis[15] 
        self.Y5 = self.Y5u + dis[16] 
        self.Z5 = self.Z5u + dis[17] 
        self.X6 = self.X6u + dis[18] 
        self.Y6 = self.Y6u + dis[19] 
        self.Z6 = self.Z6u + dis[20] 
        self.X7 = self.X7u + dis[21] 
        self.Y7 = self.Y7u + dis[22] 
        self.Z7 = self.Z7u + dis[23] 
        return
    def GeomStiff(self, r, s, t, sig):
        GeomK = zeros((self.DofE, self.DofE), dtype=float)
        B, det = self.Basics( r, s, t)
        for i in xrange(8):
            ii = 3*i
            for j in xrange(8):
                jj = 3*j
                HH = sig[0]*B[0,i]*B[0,j] + sig[1]*B[1,i]*B[1,j] + sig[2]*B[2,i]*B[2,j] + sig[3]*(B[1,i]*B[2,j]+B[2,i]*B[1,j]) + sig[4]*(B[0,i]*B[2,j]+B[2,i]*B[0,j]) + sig[5]*(B[0,i]*B[1,j]+B[1,i]*B[0,j])
                for k in xrange(3):
                    GeomK[ii+k,jj+k] = HH
        return GeomK
    def Basics(self, r, s, t):
        br0, bs0, bt0 = -0.125*(1.-s)*(1.-t), -0.125*(1.-r)*(1.-t), -0.125*(1.-r)*(1.-s)
        br1, bs1, bt1 =  0.125*(1.-s)*(1.-t), -0.125*(1.+r)*(1.-t), -0.125*(1.+r)*(1.-s)
        br2, bs2, bt2 =  0.125*(1.+s)*(1.-t),  0.125*(1.+r)*(1.-t), -0.125*(1.+r)*(1.+s)
        br3, bs3, bt3 = -0.125*(1.+s)*(1.-t),  0.125*(1.-r)*(1.-t), -0.125*(1.-r)*(1.+s)
        br4, bs4, bt4 = -0.125*(1.-s)*(1.+t), -0.125*(1.-r)*(1.+t),  0.125*(1.-r)*(1.-s)
        br5, bs5, bt5 =  0.125*(1.-s)*(1.+t), -0.125*(1.+r)*(1.+t),  0.125*(1.+r)*(1.-s)
        br6, bs6, bt6 =  0.125*(1.+s)*(1.+t),  0.125*(1.+r)*(1.+t),  0.125*(1.+r)*(1.+s)
        br7, bs7, bt7 = -0.125*(1.+s)*(1.+t),  0.125*(1.-r)*(1.+t),  0.125*(1.-r)*(1.+s)
        JJ = zeros((3,3), dtype=float)
#        JJ[0,0] = br0*self.X0 + br1*self.X1 + br2*self.X2 + br3*self.X3 + br4*self.X4 + br5*self.X5 + br6*self.X6 + br7*self.X7
#        JJ[0,1] = bs0*self.X0 + bs1*self.X1 + bs2*self.X2 + bs3*self.X3 + bs4*self.X4 + bs5*self.X5 + bs6*self.X6 + bs7*self.X7
#        JJ[0,2] = bt0*self.X0 + bt1*self.X1 + bt2*self.X2 + bt3*self.X3 + bt4*self.X4 + bt5*self.X5 + bt6*self.X6 + bt7*self.X7
#        JJ[1,0] = br0*self.Y0 + br1*self.Y1 + br2*self.Y2 + br3*self.Y3 + br4*self.Y4 + br5*self.Y5 + br6*self.Y6 + br7*self.Y7
#        JJ[1,1] = bs0*self.Y0 + bs1*self.Y1 + bs2*self.Y2 + bs3*self.Y3 + bs4*self.Y4 + bs5*self.Y5 + bs6*self.Y6 + bs7*self.Y7
#        JJ[1,2] = bt0*self.Y0 + bt1*self.Y1 + bt2*self.Y2 + bt3*self.Y3 + bt4*self.Y4 + bt5*self.Y5 + bt6*self.Y6 + bt7*self.Y7
#        JJ[2,0] = br0*self.Z0 + br1*self.Z1 + br2*self.Z2 + br3*self.Z3 + br4*self.Z4 + br5*self.Z5 + br6*self.Z6 + br7*self.Z7
#        JJ[2,1] = bs0*self.Z0 + bs1*self.Z1 + bs2*self.Z2 + bs3*self.Z3 + bs4*self.Z4 + bs5*self.Z5 + bs6*self.Z6 + bs7*self.Z7
#        JJ[2,2] = bt0*self.Z0 + bt1*self.Z1 + bt2*self.Z2 + bt3*self.Z3 + bt4*self.Z4 + bt5*self.Z5 + bt6*self.Z6 + bt7*self.Z7

        JJ[0,0] = br0*self.X0 + br1*self.X1 + br2*self.X2 + br3*self.X3 + br4*self.X4 + br5*self.X5 + br6*self.X6 + br7*self.X7
        JJ[0,1] = br0*self.Y0 + br1*self.Y1 + br2*self.Y2 + br3*self.Y3 + br4*self.Y4 + br5*self.Y5 + br6*self.Y6 + br7*self.Y7
        JJ[0,2] = br0*self.Z0 + br1*self.Z1 + br2*self.Z2 + br3*self.Z3 + br4*self.Z4 + br5*self.Z5 + br6*self.Z6 + br7*self.Z7
        JJ[1,0] = bs0*self.X0 + bs1*self.X1 + bs2*self.X2 + bs3*self.X3 + bs4*self.X4 + bs5*self.X5 + bs6*self.X6 + bs7*self.X7
        JJ[1,1] = bs0*self.Y0 + bs1*self.Y1 + bs2*self.Y2 + bs3*self.Y3 + bs4*self.Y4 + bs5*self.Y5 + bs6*self.Y6 + bs7*self.Y7
        JJ[1,2] = bs0*self.Z0 + bs1*self.Z1 + bs2*self.Z2 + bs3*self.Z3 + bs4*self.Z4 + bs5*self.Z5 + bs6*self.Z6 + bs7*self.Z7
        JJ[2,0] = bt0*self.X0 + bt1*self.X1 + bt2*self.X2 + bt3*self.X3 + bt4*self.X4 + bt5*self.X5 + bt6*self.X6 + bt7*self.X7
        JJ[2,1] = bt0*self.Y0 + bt1*self.Y1 + bt2*self.Y2 + bt3*self.Y3 + bt4*self.Y4 + bt5*self.Y5 + bt6*self.Y6 + bt7*self.Y7
        JJ[2,2] = bt0*self.Z0 + bt1*self.Z1 + bt2*self.Z2 + bt3*self.Z3 + bt4*self.Z4 + bt5*self.Z5 + bt6*self.Z6 + bt7*self.Z7
        det = JJ[0,0]*JJ[1,1]*JJ[2,2]-JJ[0,0]*JJ[1,2]*JJ[2,1]-JJ[1,0]*JJ[0,1]*JJ[2,2]+JJ[1,0]*JJ[0,2]*JJ[2,1]+JJ[2,0]*JJ[0,1]*JJ[1,2]-JJ[2,0]*JJ[0,2]*JJ[1,1]
        JI = inv(JJ)
        BB = zeros((3,8), dtype=float)
        BB[0,0] = JI[0,0]*br0+JI[0,1]*bs0+JI[0,2]*bt0
        BB[0,1] = JI[0,0]*br1+JI[0,1]*bs1+JI[0,2]*bt1
        BB[0,2] = JI[0,0]*br2+JI[0,1]*bs2+JI[0,2]*bt2
        BB[0,3] = JI[0,0]*br3+JI[0,1]*bs3+JI[0,2]*bt3
        BB[0,4] = JI[0,0]*br4+JI[0,1]*bs4+JI[0,2]*bt4
        BB[0,5] = JI[0,0]*br5+JI[0,1]*bs5+JI[0,2]*bt5
        BB[0,6] = JI[0,0]*br6+JI[0,1]*bs6+JI[0,2]*bt6
        BB[0,7] = JI[0,0]*br7+JI[0,1]*bs7+JI[0,2]*bt7
        BB[1,0] = JI[1,0]*br0+JI[1,1]*bs0+JI[1,2]*bt0
        BB[1,1] = JI[1,0]*br1+JI[1,1]*bs1+JI[1,2]*bt1
        BB[1,2] = JI[1,0]*br2+JI[1,1]*bs2+JI[1,2]*bt2
        BB[1,3] = JI[1,0]*br3+JI[1,1]*bs3+JI[1,2]*bt3
        BB[1,4] = JI[1,0]*br4+JI[1,1]*bs4+JI[1,2]*bt4
        BB[1,5] = JI[1,0]*br5+JI[1,1]*bs5+JI[1,2]*bt5
        BB[1,6] = JI[1,0]*br6+JI[1,1]*bs6+JI[1,2]*bt6
        BB[1,7] = JI[1,0]*br7+JI[1,1]*bs7+JI[1,2]*bt7
        BB[2,0] = JI[2,0]*br0+JI[2,1]*bs0+JI[2,2]*bt0
        BB[2,1] = JI[2,0]*br1+JI[2,1]*bs1+JI[2,2]*bt1
        BB[2,2] = JI[2,0]*br2+JI[2,1]*bs2+JI[2,2]*bt2
        BB[2,3] = JI[2,0]*br3+JI[2,1]*bs3+JI[2,2]*bt3
        BB[2,4] = JI[2,0]*br4+JI[2,1]*bs4+JI[2,2]*bt4
        BB[2,5] = JI[2,0]*br5+JI[2,1]*bs5+JI[2,2]*bt5
        BB[2,6] = JI[2,0]*br6+JI[2,1]*bs6+JI[2,2]*bt6
        BB[2,7] = JI[2,0]*br7+JI[2,1]*bs7+JI[2,2]*bt7
        return BB, det
    def FormN(self, r, s, t):
        N = array([[(1.-r)*(1.-s)*(1.-t)*0.125, 0, 0, (1.+r)*(1.-s)*(1.-t)*0.125, 0, 0, (1+r)*(1+s)*(1.-t)*0.125, 0, 0, (1.-r)*(1.+s)*(1.-t)*0.125, 0, 0, (1.-r)*(1.-s)*(1.+t)*0.125, 0, 0, (1.+r)*(1.-s)*(1.+t)*0.125, 0, 0, (1+r)*(1+s)*(1.+t)*0.125, 0, 0, (1.-r)*(1.+s)*(1.+t)*0.125, 0, 0],
                   [0, (1.-r)*(1.-s)*(1.-t)*0.125, 0, 0, (1.+r)*(1.-s)*(1.-t)*0.125, 0, 0, (1+r)*(1+s)*(1.-t)*0.125, 0, 0, (1.-r)*(1.+s)*(1.-t)*0.125, 0, 0, (1.-r)*(1.-s)*(1.+t)*0.125, 0, 0, (1.+r)*(1.-s)*(1.+t)*0.125, 0, 0, (1+r)*(1+s)*(1.+t)*0.125, 0, 0, (1.-r)*(1.+s)*(1.+t)*0.125, 0],
                   [0, 0, (1.-r)*(1.-s)*(1.-t)*0.125, 0, 0, (1.+r)*(1.-s)*(1.-t)*0.125, 0, 0, (1+r)*(1+s)*(1.-t)*0.125, 0, 0, (1.-r)*(1.+s)*(1.-t)*0.125, 0, 0, (1.-r)*(1.-s)*(1.+t)*0.125, 0, 0, (1.+r)*(1.-s)*(1.+t)*0.125, 0, 0, (1+r)*(1+s)*(1.+t)*0.125, 0, 0, (1.-r)*(1.+s)*(1.+t)*0.125]])
        return N
    def FormB(self, r, s, t, NLg):
        BB, det = self.Basics( r, s, t)
        if self.RegType==1:
            B = array([[ BB[0,0], 0,   0,   0,  BB[0,1], 0,   0,   0,  BB[0,2], 0,   0,   0,  BB[0,3], 0,   0,   0,  BB[0,4], 0,   0,   0,  BB[0,5], 0,   0,   0,  BB[0,6], 0,   0,   0,  BB[0,7], 0,   0,  0],
                       [ 0,   BB[1,0], 0,   0,  0,   BB[1,1], 0,   0,  0,   BB[1,2], 0,   0,  0,   BB[1,3], 0,   0,  0,   BB[1,4], 0,   0,  0,   BB[1,5], 0,   0,  0,   BB[1,6], 0,   0,  0,   BB[1,7], 0,  0],
                       [ 0,   0,   BB[2,0], 0,  0,   0,   BB[2,1], 0,  0,   0,   BB[2,2], 0,  0,   0,   BB[2,3], 0,  0,   0,   BB[2,4], 0,  0,   0,   BB[2,5], 0,  0,   0,   BB[2,6], 0,  0,   0,   BB[2,7],0],
                       [ 0,   BB[2,0], BB[1,0], 0,  0,   BB[2,1], BB[1,1], 0,  0,   BB[2,2], BB[1,2], 0,  0,   BB[2,3], BB[1,3], 0,  0,   BB[2,4], BB[1,4], 0,  0,   BB[2,5], BB[1,5], 0,  0,   BB[2,6], BB[1,6], 0,  0,   BB[2,7], BB[1,7],0],
                       [ BB[2,0], 0,   BB[0,0], 0,  BB[2,1], 0,   BB[0,1], 0,  BB[2,2], 0,   BB[0,2], 0,  BB[2,3], 0,   BB[0,3], 0,  BB[2,4], 0,   BB[0,4], 0,  BB[2,5], 0,   BB[0,5], 0,  BB[2,6], 0,   BB[0,6], 0,  BB[2,7], 0,   BB[0,7],0],
                       [ BB[1,0], BB[0,0], 0,   0,  BB[1,1], BB[0,1], 0,   0,  BB[1,2], BB[0,2], 0,   0,  BB[1,3], BB[0,3], 0,   0,  BB[1,4], BB[0,4], 0,   0,  BB[1,5], BB[0,5], 0,   0,  BB[1,6], BB[0,6], 0,   0,  BB[1,7], BB[0,7], 0,  0],
                       [ 0,   0,   0,   BB[0,0],0,   0,   0,   BB[0,1],0,   0,   0,   BB[0,2],0,   0,   0,   BB[0,3],0,   0,   0,   BB[0,4],0,   0,   0,   BB[0,5],0,   0,   0,   BB[0,6],0,   0,   0,  BB[0,7]],
                       [ 0,   0,   0,   BB[1,0],0,   0,   0,   BB[1,1],0,   0,   0,   BB[1,2],0,   0,   0,   BB[1,3],0,   0,   0,   BB[1,4],0,   0,   0,   BB[1,5],0,   0,   0,   BB[1,6],0,   0,   0,  BB[1,7]],
                       [ 0,   0,   0,   BB[2,0],0,   0,   0,   BB[2,1],0,   0,   0,   BB[2,2],0,   0,   0,   BB[2,3],0,   0,   0,   BB[2,4],0,   0,   0,   BB[2,5],0,   0,   0,   BB[2,6],0,   0,   0,  BB[2,7]]])
            N0 = (1.-r)*(1.-s)*(1.-t)*0.125
            N1 = (1.+r)*(1.-s)*(1.-t)*0.125
            N2 = (1.+r)*(1.+s)*(1.-t)*0.125
            N3 = (1.-r)*(1.+s)*(1.-t)*0.125
            N4 = (1.-r)*(1.-s)*(1.+t)*0.125
            N5 = (1.+r)*(1.-s)*(1.+t)*0.125
            N6 = (1.+r)*(1.+s)*(1.+t)*0.125 
            N7 = (1.-r)*(1.+s)*(1.+t)*0.125
            BN =array([[BB[0,0],0,      0,   0, BB[0,1], 0,   0,   0, BB[0,2], 0,   0,   0, BB[0,3], 0,   0,   0, BB[0,4], 0,   0,   0, BB[0,5], 0,   0,   0, BB[0,6], 0,   0,   0, BB[0,7], 0,   0,  0],
                       [0,      BB[1,0],0,   0, 0,   BB[1,1], 0,   0, 0,   BB[1,2], 0,   0, 0,   BB[1,3], 0,   0, 0,   BB[1,4], 0,   0, 0,   BB[1,5], 0,   0, 0,   BB[1,6], 0,   0, 0,   BB[1,7], 0,  0],
                       [0,      0,      BB[2,0], 0, 0,   0,   BB[2,1], 0, 0,   0,   BB[2,2], 0, 0,   0,   BB[2,3], 0, 0,   0,   BB[2,4], 0, 0,   0,   BB[2,5], 0, 0,   0,   BB[2,6], 0, 0,   0,   BB[2,7],0],
                       [0,      BB[2,0],BB[1,0], 0, 0,   BB[2,1], BB[1,1], 0, 0,   BB[2,2], BB[1,2], 0, 0,   BB[2,3], BB[1,3], 0, 0,   BB[2,4], BB[1,4], 0, 0,   BB[2,5], BB[1,5], 0, 0,   BB[2,6], BB[1,6], 0, 0,   BB[2,7], BB[1,7],0],
                       [BB[2,0],0,      BB[0,0], 0, BB[2,1], 0,   BB[0,1], 0, BB[2,2], 0,   BB[0,2], 0, BB[2,3], 0,   BB[0,3], 0, BB[2,4], 0,   BB[0,4], 0, BB[2,5], 0,   BB[0,5], 0, BB[2,6], 0,   BB[0,6], 0, BB[2,7], 0,   BB[0,7],0],
                       [BB[1,0],BB[0,0], 0,   0, BB[1,1], BB[0,1], 0,   0, BB[1,2], BB[0,2], 0,   0, BB[1,3], BB[0,3], 0,   0, BB[1,4], BB[0,4], 0,   0, BB[1,5], BB[0,5], 0,   0, BB[1,6], BB[0,6], 0,   0, BB[1,7], BB[0,7], 0,  0],
                       [0,      0,   0,   N0,0,   0,   0,   N1,0,   0,   0,   N2,0,   0,   0,   N3,0,   0,   0,   N4,0,   0,   0,   N5,0,   0,   0,   N6,0,   0,   0,  N7]])
            return B, BN, det, 0
        else:
            B = array([[BB[0,0],0,      0,      BB[0,1],0,      0,      BB[0,2], 0,   0,   BB[0,3], 0,   0,   BB[0,4], 0,   0,   BB[0,5], 0,   0,   BB[0,6], 0,   0,   BB[0,7], 0,   0  ],
                       [0,      BB[1,0],0,      0,      BB[1,1],0,      0,   BB[1,2], 0,   0,   BB[1,3], 0,   0,   BB[1,4], 0,   0,   BB[1,5], 0,   0,   BB[1,6], 0,   0,   BB[1,7], 0  ],
                       [0,      0,      BB[2,0],0,      0,      BB[2,1],0,   0,   BB[2,2], 0,   0,   BB[2,3], 0,   0,   BB[2,4], 0,   0,   BB[2,5], 0,   0,   BB[2,6], 0,   0,   BB[2,7]],
                       [0,      BB[2,0],BB[1,0],0,      BB[2,1],BB[1,1],0,   BB[2,2], BB[1,2], 0,   BB[2,3], BB[1,3], 0,   BB[2,4], BB[1,4], 0,   BB[2,5], BB[1,5], 0,   BB[2,6], BB[1,6], 0,   BB[2,7], BB[1,7]],
                       [BB[2,0],0,      BB[0,0],BB[2,1],0,      BB[0,1],BB[2,2], 0,   BB[0,2], BB[2,3], 0,   BB[0,3], BB[2,4], 0,   BB[0,4], BB[2,5], 0,   BB[0,5], BB[2,6], 0,   BB[0,6], BB[2,7], 0,   BB[0,7]],
                       [BB[1,0],BB[0,0],0,      BB[1,1],BB[0,1],0,      BB[1,2], BB[0,2], 0,   BB[1,3], BB[0,3], 0,   BB[1,4], BB[0,4], 0,   BB[1,5], BB[0,5], 0,   BB[1,6], BB[0,6], 0,   BB[1,7], BB[0,7], 0  ]])
            return B, det, 0
    def FormT(self, r, s, t):                                 # interpolation on temperature
        # !!! ordering might not yet be correct !!!
        if self.RegType==1: T = array([(1.-r)*(1.-s)*(1.-t)*0.125, 0, 0, 0,(1.+r)*(1.-s)*(1.-t)*0.125, 0, 0, 0,(1+r)*(1+s)*(1.-t)*0.125, 0, 0, 0,(1.-r)*(1.+s)*(1.-t)*0.125, 0, 0, 0,(1.-r)*(1.-s)*(1.+t)*0.125, 0, 0, 0,(1.+r)*(1.-s)*(1.+t)*0.125, 0, 0, 0,(1+r)*(1+s)*(1.+t)*0.125, 0, 0, 0,(1.-r)*(1.+s)*(1.+t)*0.125, 0, 0, 0])
        else:               T = array([(1.-r)*(1.-s)*(1.-t)*0.125, 0, 0,   (1.+r)*(1.-s)*(1.-t)*0.125, 0, 0,   (1+r)*(1+s)*(1.-t)*0.125, 0, 0,   (1.-r)*(1.+s)*(1.-t)*0.125, 0, 0,   (1.-r)*(1.-s)*(1.+t)*0.125, 0, 0,   (1.+r)*(1.-s)*(1.+t)*0.125, 0, 0,   (1+r)*(1+s)*(1.+t)*0.125, 0, 0,   (1.-r)*(1.+s)*(1.+t)*0.125, 0, 0])
        return T
    def FormX(self, r, s, t):
        X = array([(1.-r)*(1.-s)*(1.-t)*0.125, (1.+r)*(1.-s)*(1.-t)*0.125, (1+r)*(1+s)*(1.-t)*0.125, (1.-r)*(1.+s)*(1.-t)*0.125, (1.-r)*(1.-s)*(1.+t)*0.125, (1.+r)*(1.-s)*(1.+t)*0.125, (1+r)*(1+s)*(1.+t)*0.125, (1.-r)*(1.+s)*(1.+t)*0.125])
        return X
    def JacoD(self, r, s, t):
        br0, bs0, bt0 = -0.125*(1.-s)*(1.-t), -0.125*(1.-r)*(1.-t), -0.125*(1.-r)*(1.-s)
        br1, bs1, bt1 =  0.125*(1.-s)*(1.-t), -0.125*(1.+r)*(1.-t), -0.125*(1.+r)*(1.-s)
        br2, bs2, bt2 =  0.125*(1.+s)*(1.-t),  0.125*(1.+r)*(1.-t), -0.125*(1.+r)*(1.+s)
        br3, bs3, bt3 = -0.125*(1.+s)*(1.-t),  0.125*(1.-r)*(1.-t), -0.125*(1.-r)*(1.+s)
        br4, bs4, bt4 = -0.125*(1.-s)*(1.+t), -0.125*(1.-r)*(1.+t),  0.125*(1.-r)*(1.-s)
        br5, bs5, bt5 =  0.125*(1.-s)*(1.+t), -0.125*(1.+r)*(1.+t),  0.125*(1.+r)*(1.-s)
        br6, bs6, bt6 =  0.125*(1.+s)*(1.+t),  0.125*(1.+r)*(1.+t),  0.125*(1.+r)*(1.+s)
        br7, bs7, bt7 = -0.125*(1.+s)*(1.+t),  0.125*(1.-r)*(1.+t),  0.125*(1.-r)*(1.+s)
        JJ = zeros((3,3), dtype=float)
        JJ[0,0] = br0*self.X0 + br1*self.X1 + br2*self.X2 + br3*self.X3 + br4*self.X4 + br5*self.X5 + br6*self.X6 + br7*self.X7
        JJ[0,1] = bs0*self.X0 + bs1*self.X1 + bs2*self.X2 + bs3*self.X3 + bs4*self.X4 + bs5*self.X5 + bs6*self.X6 + bs7*self.X7
        JJ[0,2] = bt0*self.X0 + bt1*self.X1 + bt2*self.X2 + bt3*self.X3 + bt4*self.X4 + bt5*self.X5 + bt6*self.X6 + bt7*self.X7
        JJ[1,0] = br0*self.Y0 + br1*self.Y1 + br2*self.Y2 + br3*self.Y3 + br4*self.Y4 + br5*self.Y5 + br6*self.Y6 + br7*self.Y7
        JJ[1,1] = bs0*self.Y0 + bs1*self.Y1 + bs2*self.Y2 + bs3*self.Y3 + bs4*self.Y4 + bs5*self.Y5 + bs6*self.Y6 + bs7*self.Y7
        JJ[1,2] = bt0*self.Y0 + bt1*self.Y1 + bt2*self.Y2 + bt3*self.Y3 + bt4*self.Y4 + bt5*self.Y5 + bt6*self.Y6 + bt7*self.Y7
        JJ[2,0] = br0*self.Z0 + br1*self.Z1 + br2*self.Z2 + br3*self.Z3 + br4*self.Z4 + br5*self.Z5 + br6*self.Z6 + br7*self.Z7
        JJ[2,1] = bs0*self.Z0 + bs1*self.Z1 + bs2*self.Z2 + bs3*self.Z3 + bs4*self.Z4 + bs5*self.Z5 + bs6*self.Z6 + bs7*self.Z7
        JJ[2,2] = bt0*self.Z0 + bt1*self.Z1 + bt2*self.Z2 + bt3*self.Z3 + bt4*self.Z4 + bt5*self.Z5 + bt6*self.Z6 + bt7*self.Z7
        det = JJ[0,0]*JJ[1,1]*JJ[2,2]-JJ[0,0]*JJ[1,2]*JJ[2,1]-JJ[1,0]*JJ[0,1]*JJ[2,2]+JJ[1,0]*JJ[0,2]*JJ[2,1]+JJ[2,0]*JJ[0,1]*JJ[1,2]-JJ[2,0]*JJ[0,2]*JJ[1,1]
        return det

class SB3(Element):
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, ShellSecDic, StateV, NData):
#        Element.__init__(self,"SB3",   3,9,2,                   3,2,3,                    (set([3, 4, 5]),set([3, 4, 5]),set([3, 4, 5])),(3,3,3), 20,False)
        Element.__init__(self,"SB3",   3,9,2,                   3,3,4,                    (set([3, 4, 5]),set([3, 4, 5]),set([3, 4, 5])),(3,3,3), 20,False)
#        Element.__init__(self,"SB3",   3,9,2,                   3,4,3,                    (set([3, 4, 5]),set([3, 4, 5]),set([3, 4, 5])),(3,3,3), 20,False)
#                       (self, TypeVal,nNodVal,DofEVal,nFieVal, IntTVal,nIntVal,nIntLVal, DofTVal, DofNVal, dimVal, NLGeomIVal):
        self.Label = Label                              # element number in input
        self.DofI = zeros( (self.nNod,3), dtype=int)    # indices of global dofs per node
        self.TensStiff = False                          # flag for tension stiffening
        self.MatN = MatName                             # name of material
        self.Set = SetLabel                             # label of corresponding element set
        self.InzList = [InzList[0], InzList[1], InzList[2]]
        self.Geom = zeros( (2,2), dtype=double)
#        self.Geom[0,0] = 0.5*self.AA                    # element area for numerical integration
        self.Geom[1,0] = 1                              # dummy for thickness
        self.Geom[1,1] = ShellSecDic.Height             # used in ConFem.Materials
        self.Data = zeros((self.nIntL,NData+2), dtype=float)# storage for element data, extra 2 for shear forces computed from moment derivatives
        self.DataP= zeros((self.nIntL,NData+2), dtype=float)# storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        i2 = FindIndexByLabel( NoList, self.InzList[2])      # find node index from node label
        self.Inzi = [ i0, i1, i2]                       # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0])# attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        NoList[i2].DofT = NoList[i2].DofT.union(self.DofT[2])
        X0 = NoList[i0].XCo
        Y0 = NoList[i0].YCo
        X1 = NoList[i1].XCo
        Y1 = NoList[i1].YCo
        X2 = NoList[i2].XCo
        Y2 = NoList[i2].YCo
        self.AA = -Y0*X1+Y0*X2+Y2*X1+Y1*X0-Y1*X2-Y2*X0  # double of element area
        self.Geom[0,0] = 0.5*self.AA                    # element area for numerical integration
        self.Lch_ = 0.3*sqrt(0.5*self.AA)                # characteristic length
        if self.AA<=0.: raise NameError("Something is wrong with this SB3-element")
        l1=sqrt((X2-X1)**2+(Y2-Y1)**2)
        l2=sqrt((X0-X2)**2+(Y0-Y2)**2)
        l3=sqrt((X1-X0)**2+(Y1-Y0)**2)
        self.mu1=(l3**2-l2**2)/l1**2
        self.mu2=(l1**2-l3**2)/l2**2
        self.mu3=(l2**2-l1**2)/l3**2
        self.b1=(Y1-Y2)/self.AA
        self.b2=(Y2-Y0)/self.AA
        self.b3=(Y0-Y1)/self.AA
        self.c1=(X2-X1)/self.AA
        self.c2=(X0-X2)/self.AA
        self.c3=(X1-X0)/self.AA
    def FormN(self, L1, L2, L3):
        L3=1-L1-L2
        c1=self.c1
        c2=self.c2
        c3=self.c3
        b1=self.b1
        b2=self.b2
        b3=self.b3
        N = array([[0,0,0,0,0,0,0,0,0],
                  [L1-L1*L2+2*L1**2*L2+L1*L2*L3*(3*(1-self.mu3)*L1-(1+3*self.mu3)*L2+(1+3*self.mu3)*L3)+L3*L1-2*L3**2*L1-L1*L2*L3*(3*(1-self.mu2)*L3-(1+3*self.mu2)*L1+(1+3*self.mu2)*L2),
                   c2*(L3**2*L1+0.5*L1*L2*L3*(3*(1-self.mu2)*L3-(1+3*self.mu2)*L1+(1+3*self.mu2)*L2))+c3*(L1**2*L2+0.5*L1*L2*L3*(3*(1-self.mu3)*L1-(1+3*self.mu3)*L2+(1+3*self.mu3)*L3))-c2*L3*L1,
                   -b2*(L3**2*L1+0.5*L1*L2*L3*(3*(1-self.mu2)*L3-(1+3*self.mu2)*L1+(1+3*self.mu2)*L2))-b3*(L1**2*L2+0.5*L1*L2*L3*(3*(1-self.mu3)*L1-(1+3*self.mu3)*L2+(1+3*self.mu3)*L3))+b2*L3*L1,
                   -2*L1**2*L2-L1*L2*L3*(3*(1-self.mu3)*L1-(1+3*self.mu3)*L2+(1+3*self.mu3)*L3)-L2*L3+L1*L2+L2+2*L2**2*L3+L1*L2*L3*(3*(1-self.mu1)*L2-(1+3*self.mu1)*L3+(1+3*self.mu1)*L1),
                   -c3*L1*L2+c3*(L1**2*L2+0.5*L1*L2*L3*(3*(1-self.mu3)*L1-(1+3*self.mu3)*L2+(1+3*self.mu3)*L3))+c1*(L2**2*L3+0.5*L1*L2*L3*(3*(1-self.mu1)*L2-(1+3*self.mu1)*L3+(1+3*self.mu1)*L1)),
                   b3*L1*L2-b3*(L1**2*L2+0.5*L1*L2*L3*(3*(1-self.mu3)*L1-(1+3*self.mu3)*L2+(1+3*self.mu3)*L3))-b1*(L2**2*L3+0.5*L1*L2*L3*(3*(1-self.mu1)*L2-(1+3*self.mu1)*L3+(1+3*self.mu1)*L1)),
                   L2*L3+2*L3**2*L1+L1*L2*L3*(3*(1-self.mu2)*L3-(1+3*self.mu2)*L1+(1+3*self.mu2)*L2)-2*L2**2*L3-L1*L2*L3*(3*(1-self.mu1)*L2-(1+3*self.mu1)*L3+(1+3*self.mu1)*L1)+L3-L3*L1,
                   c1*(L2**2*L3+0.5*L1*L2*L3*(3*(1-self.mu1)*L2-(1+3*self.mu1)*L3+(1+3*self.mu1)*L1))+c2*(L3**2*L1+0.5*L1*L2*L3*(3*(1-self.mu2)*L3-(1+3*self.mu2)*L1+(1+3*self.mu2)*L2))-c1*L2*L3,
                   -b1*(L2**2*L3+0.5*L1*L2*L3*(3*(1-self.mu1)*L2-(1+3*self.mu1)*L3+(1+3*self.mu1)*L1))-b2*(L3**2*L1+0.5*L1*L2*L3*(3*(1-self.mu2)*L3-(1+3*self.mu2)*L1+(1+3*self.mu2)*L2))+b1*L2*L3]])        
        return N
    def FormB(self, L1, L2, L3, NLg):
        L3=1-L1-L2
        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        b1 = self.b1
        b2 = self.b2
        b3 = self.b3
        mu1= self.mu1
        mu2= self.mu2
        mu3= self.mu3
#        B =array([[-2*(-3*self.b1*self.b2*self.mu3+2*self.b1*self.b2-3*self.b1*self.b2*self.mu2)*L3**2+(-2*(4*self.b3*self.b2-6*self.b2*self.b3*self.mu2+2*self.b2**2-8*self.b1*self.b2-6*self.b2*self.b3*self.mu3+3*self.b2**2*self.mu3+3*self.b2**2*self.mu2-6*self.b1*self.b2*self.mu2+6*self.b1*self.b2*self.mu3)*L1-2*(-3*self.b1**2*self.mu2+4*self.b1*self.b2+4*self.b1*self.b3+6*self.b1*self.b2*self.mu2-4*self.b1**2+6*self.b1*self.b2*self.mu3-6*self.b1*self.b3*self.mu2+3*self.b1**2*self.mu3-6*self.b1*self.b3*self.mu3)*L2-8*self.b1*self.b3)*L3-2*(2*self.b1*self.b3+3*self.b1*self.b3*self.mu2+3*self.b1*self.b3*self.mu3)*L2**2+(-2*(2*self.b3**2-3*self.b3**2*self.mu2-8*self.b1*self.b3-3*self.b3**2*self.mu3+4*self.b3*self.b2+6*self.b1*self.b3*self.mu3+6*self.b2*self.b3*self.mu2+6*self.b2*self.b3*self.mu3-6*self.b1*self.b3*self.mu2)*L1+4*self.b1**2)*L2-2*self.b1*self.b2+2*self.b1*self.b3-2*(-3*self.b2*self.b3*self.mu2-4*self.b3*self.b2+3*self.b2*self.b3*self.mu3)*L1**2-2*(-4*self.b1*self.b2+2*self.b3**2)*L1,
#                   -(3*self.b1*c2*self.b2*self.mu2-self.b1*self.b2*c3-3*self.b1*c3*self.b2*self.mu3-3*self.b1*c2*self.b2)*L3**2+(-(-c2*self.b2**2+6*self.b2*c2*self.b3*self.mu2+6*self.b1*c2*self.b2*self.mu2-6*self.b1*self.b2*c3+2*self.b1*c2*self.b2-6*self.b2*self.b3*c3*self.mu3-3*c2*self.b2**2*self.mu2+6*self.b1*c3*self.b2*self.mu3+3*c3*self.b2**2*self.mu3+c3*self.b2**2-6*self.b2*c2*self.b3-2*self.b2*self.b3*c3)*L1-(3*c2*self.b1**2*self.mu2-2*self.b1*c2*self.b2+c2*self.b1**2-6*self.b1*c2*self.b2*self.mu2+6*self.b1*c2*self.b3*self.mu2-6*self.b1*c3*self.b3*self.mu3-2*self.b1*c3*self.b3+2*self.b1*self.b2*c3+6*self.b1*c3*self.b2*self.mu3-3*c3*self.b1**2+3*c3*self.b1**2*self.mu3-6*self.b1*c2*self.b3)*L2+4*self.b1*c2*self.b3)*L3-(-3*self.b1*c2*self.b3*self.mu2-self.b1*c2*self.b3+self.b1*c3*self.b3+3*self.b1*c3*self.b3*self.mu3)*L2**2+(-(2*self.b1*c2*self.b3-3*self.b3**2*c2+6*self.b2*self.b3*c3*self.mu3-2*self.b2*c2*self.b3-6*self.b2*c2*self.b3*self.mu2-c3*self.b3**2-6*self.b1*c3*self.b3+2*self.b2*self.b3*c3+6*self.b1*c2*self.b3*self.mu2+6*self.b1*c3*self.b3*self.mu3+3*c2*self.b3**2*self.mu2-3*c3*self.b3**2*self.mu3)*L1+2*c3*self.b1**2)*L2-2*self.b1*c2*self.b3-(3*self.b2*self.b3*c3*self.mu3+self.b2*c2*self.b3+3*self.b2*c2*self.b3*self.mu2-3*self.b2*self.b3*c3)*L1**2-(-4*self.b1*self.b2*c3-2*self.b3**2*c2)*L1,
#                   (-self.b1*self.b3*self.b2+3*self.b1*self.b2**2*self.mu2-3*self.b1*self.b2*self.b3*self.mu3-3*self.b1*self.b2**2)*L3**2+((6*self.b1*self.b2**2*self.mu2+6*self.b1*self.b2*self.b3*self.mu3-6*self.b2*self.b3**2*self.mu3-6*self.b1*self.b3*self.b2+2*self.b1*self.b2**2-2*self.b2*self.b3**2-3*self.b2**3*self.mu2-5*self.b3*self.b2**2-self.b2**3+3*self.b2**2*self.b3*self.mu3+6*self.b3*self.b2**2*self.mu2)*L1+(-4*self.b1*self.b3*self.b2+6*self.b1*self.b3*self.b2*self.mu2-6*self.b1*self.b2**2*self.mu2-6*self.b1*self.b3**2*self.mu3+self.b1**2*self.b2-2*self.b1*self.b3**2-2*self.b1*self.b2**2+3*self.b1**2*self.b3*self.mu3-3*self.b1**2*self.b3+3*self.b1**2*self.b2*self.mu2+6*self.b1*self.b2*self.b3*self.mu3)*L2-4*self.b1*self.b3*self.b2)*L3+(-3*self.b1*self.b3*self.b2*self.mu2-self.b1*self.b3*self.b2+3*self.b1*self.b3**2*self.mu3+self.b1*self.b3**2)*L2**2+((-self.b2*self.b3**2+6*self.b1*self.b3*self.b2*self.mu2-self.b3**3+2*self.b1*self.b3*self.b2+6*self.b2*self.b3**2*self.mu3-3*self.b3**3*self.mu3-6*self.b1*self.b3**2-2*self.b3*self.b2**2+6*self.b1*self.b3**2*self.mu3+3*self.b3**2*self.b2*self.mu2-6*self.b3*self.b2**2*self.mu2)*L1-2*self.b1**2*self.b3)*L2+2*self.b1*self.b3*self.b2+(3*self.b3*self.b2**2*self.mu2+3*self.b2*self.b3**2*self.mu3-3*self.b2*self.b3**2+self.b3*self.b2**2)*L1**2+(-4*self.b1*self.b3*self.b2-2*self.b2*self.b3**2)*L1,
#                   2*(-3*self.b1*self.b2*self.mu3-3*self.b1*self.b2*self.mu1-2*self.b1*self.b2)*L3**2+(2*(-3*self.b2**2*self.mu1+6*self.b1*self.b2*self.mu3+6*self.b1*self.b2*self.mu1-4*self.b3*self.b2+4*self.b2**2-4*self.b1*self.b2-6*self.b2*self.b3*self.mu3-6*self.b2*self.b3*self.mu1+3*self.b2**2*self.mu3)*L1+2*(3*self.b1**2*self.mu1-6*self.b1*self.b3*self.mu3-2*self.b1**2-6*self.b1*self.b2*self.mu1-6*self.b1*self.b3*self.mu1+6*self.b1*self.b2*self.mu3+3*self.b1**2*self.mu3+8*self.b1*self.b2-4*self.b1*self.b3)*L2+4*self.b2**2)*L3+2*(4*self.b1*self.b3-3*self.b1*self.b3*self.mu1+3*self.b1*self.b3*self.mu3)*L2**2+(2*(-3*self.b3**2*self.mu3-6*self.b2*self.b3*self.mu1-3*self.b3**2*self.mu1-4*self.b1*self.b3+6*self.b2*self.b3*self.mu3+6*self.b1*self.b3*self.mu3-2*self.b3**2+8*self.b3*self.b2+6*self.b1*self.b3*self.mu1)*L1+8*self.b3*self.b2-4*self.b1**2)*L2+2*self.b1*self.b2-2*self.b3*self.b2+2*(3*self.b2*self.b3*self.mu3-2*self.b3*self.b2+3*self.b2*self.b3*self.mu1)*L1**2-8*self.b1*self.b2*L1,
#                   -(-self.b1*self.b2*c3+3*self.b1*c1*self.b2*self.mu1+c1*self.b1*self.b2-3*self.b1*c3*self.b2*self.mu3)*L3**2+(-(-6*self.b1*c1*self.b2*self.mu1-6*self.b2*self.b3*c3*self.mu3-6*self.b1*self.b2*c3+c3*self.b2**2+2*c1*self.b3*self.b2+3*c3*self.b2**2*self.mu3+3*c1*self.b2**2*self.mu1-2*self.b2*self.b3*c3-3*c1*self.b2**2+6*self.b1*c3*self.b2*self.mu3+6*self.b2*self.b3*c1*self.mu1-2*c1*self.b1*self.b2)*L1-(-6*c1*self.b1*self.b2-2*self.b1*c3*self.b3-3*c3*self.b1**2+6*self.b1*c1*self.b3*self.mu1+2*self.b1*self.b2*c3-6*self.b1*c3*self.b3*self.mu3+6*self.b1*c3*self.b2*self.mu3-c1*self.b1**2+2*c1*self.b1*self.b3-3*c1*self.b1**2*self.mu1+6*self.b1*c1*self.b2*self.mu1+3*c3*self.b1**2*self.mu3)*L2+2*c1*self.b2**2)*L3-(-3*c1*self.b1*self.b3+3*self.b1*c1*self.b3*self.mu1+self.b1*c3*self.b3+3*self.b1*c3*self.b3*self.mu3)*L2**2+(-(6*self.b2*self.b3*c3*self.mu3-c3*self.b3**2-6*self.b1*c1*self.b3*self.mu1-6*c1*self.b3*self.b2-3*c3*self.b3**2*self.mu3+6*self.b2*self.b3*c1*self.mu1+3*c1*self.b3**2*self.mu1+6*self.b1*c3*self.b3*self.mu3-6*self.b1*c3*self.b3-2*c1*self.b1*self.b3+2*self.b2*self.b3*c3+c1*self.b3**2)*L1+4*c1*self.b3*self.b2+2*c3*self.b1**2)*L2-2*self.b1*self.b2*c3-(-3*self.b2*self.b3*c3-c1*self.b3*self.b2+3*self.b2*self.b3*c3*self.mu3-3*self.b2*self.b3*c1*self.mu1)*L1**2+4*self.b1*self.b2*c3*L1,
#                   -(-self.b1**2*self.b2+3*self.b1*self.b2*self.b3*self.mu3+self.b1*self.b3*self.b2-3*self.b2*self.b1**2*self.mu1)*L3**2+(-(6*self.b2*self.b1**2*self.mu1-6*self.b1*self.b2*self.b3*self.mu3-self.b3*self.b2**2+2*self.b1**2*self.b2+4*self.b1*self.b3*self.b2+2*self.b2*self.b3**2-6*self.b2*self.b3*self.b1*self.mu1-3*self.b2**2*self.b3*self.mu3-3*self.b2**2*self.b1*self.mu1+6*self.b2*self.b3**2*self.mu3+3*self.b1*self.b2**2)*L1-(-6*self.b1*self.b2*self.b3*self.mu3+self.b1**3+2*self.b1*self.b3**2-6*self.b3*self.b1**2*self.mu1-2*self.b1*self.b3*self.b2+6*self.b1*self.b3**2*self.mu3+self.b1**2*self.b3-3*self.b1**2*self.b3*self.mu3+6*self.b1**2*self.b2+3*self.b1**3*self.mu1-6*self.b2*self.b1**2*self.mu1)*L2-2*self.b1*self.b2**2)*L3-(-3*self.b3*self.b1**2*self.mu1+3*self.b1**2*self.b3-3*self.b1*self.b3**2*self.mu3-self.b1*self.b3**2)*L2**2+(-(6*self.b3*self.b1**2*self.mu1-6*self.b1*self.b3**2*self.mu3+3*self.b3**3*self.mu3+6*self.b1*self.b3*self.b2-2*self.b2*self.b3**2+self.b3**3+2*self.b1**2*self.b3+5*self.b1*self.b3**2-6*self.b2*self.b3**2*self.mu3-3*self.b3**2*self.b1*self.mu1-6*self.b2*self.b3*self.b1*self.mu1)*L1-4*self.b1*self.b3*self.b2-2*self.b1**2*self.b3)*L2+2*self.b1*self.b3*self.b2-(3*self.b2*self.b3**2+3*self.b2*self.b3*self.b1*self.mu1+self.b1*self.b3*self.b2-3*self.b2*self.b3**2*self.mu3)*L1**2-4*self.b1*self.b3*self.b2*L1,
#                   -2*(-4*self.b1*self.b2+3*self.b1*self.b2*self.mu2-3*self.b1*self.b2*self.mu1)*L3**2+(-2*(4*self.b1*self.b2+6*self.b2*self.b3*self.mu2-3*self.b2**2*self.mu1-6*self.b2*self.b3*self.mu1-8*self.b3*self.b2+2*self.b2**2+6*self.b1*self.b2*self.mu1-3*self.b2**2*self.mu2+6*self.b1*self.b2*self.mu2)*L1-2*(2*self.b1**2-8*self.b1*self.b3-6*self.b1*self.b2*self.mu1+4*self.b1*self.b2+3*self.b1**2*self.mu2+3*self.b1**2*self.mu1-6*self.b1*self.b2*self.mu2+6*self.b1*self.b3*self.mu2-6*self.b1*self.b3*self.mu1)*L2-4*self.b2**2+8*self.b1*self.b3)*L3-2*(-3*self.b1*self.b3*self.mu2+2*self.b1*self.b3-3*self.b1*self.b3*self.mu1)*L2**2+(-2*(-6*self.b2*self.b3*self.mu1-6*self.b2*self.b3*self.mu2+6*self.b1*self.b3*self.mu2+4*self.b3*self.b2+3*self.b3**2*self.mu2+4*self.b1*self.b3-3*self.b3**2*self.mu1+6*self.b1*self.b3*self.mu1-4*self.b3**2)*L1-8*self.b3*self.b2)*L2-2*self.b1*self.b3+2*self.b3*self.b2-2*(2*self.b3*self.b2+3*self.b2*self.b3*self.mu2+3*self.b2*self.b3*self.mu1)*L1**2+4*self.b3**2*L1,
#                   (-3*self.b1*c1*self.b2*self.mu1-c1*self.b1*self.b2-3*self.b1*c2*self.b2*self.mu2+3*self.b1*c2*self.b2)*L3**2+((-6*self.b2*c2*self.b3*self.mu2-6*self.b2*self.b3*c1*self.mu1+6*self.b2*c2*self.b3+6*self.b1*c1*self.b2*self.mu1-3*c1*self.b2**2*self.mu1-2*self.b1*c2*self.b2-2*c1*self.b3*self.b2-6*self.b1*c2*self.b2*self.mu2+c2*self.b2**2+3*c2*self.b2**2*self.mu2+2*c1*self.b1*self.b2+3*c1*self.b2**2)*L1+(2*self.b1*c2*self.b2-6*self.b1*c1*self.b3*self.mu1-6*self.b1*c2*self.b3*self.mu2-2*c1*self.b1*self.b3+3*c1*self.b1**2*self.mu1+6*c1*self.b1*self.b2-c2*self.b1**2+6*self.b1*c2*self.b3+6*self.b1*c2*self.b2*self.mu2+c1*self.b1**2-3*c2*self.b1**2*self.mu2-6*self.b1*c1*self.b2*self.mu1)*L2+4*self.b1*c2*self.b3+2*c1*self.b2**2)*L3+(-3*self.b1*c1*self.b3*self.mu1+3*self.b1*c2*self.b3*self.mu2+3*c1*self.b1*self.b3+self.b1*c2*self.b3)*L2**2+((-6*self.b2*self.b3*c1*self.mu1-2*self.b1*c2*self.b3-6*self.b1*c2*self.b3*self.mu2-3*c2*self.b3**2*self.mu2-3*c1*self.b3**2*self.mu1+2*c1*self.b1*self.b3+2*self.b2*c2*self.b3-c1*self.b3**2+6*c1*self.b3*self.b2+6*self.b1*c1*self.b3*self.mu1+6*self.b2*c2*self.b3*self.mu2+3*self.b3**2*c2)*L1+4*c1*self.b3*self.b2)*L2-2*c1*self.b3*self.b2+(c1*self.b3*self.b2-3*self.b2*c2*self.b3*self.mu2-self.b2*c2*self.b3+3*self.b2*self.b3*c1*self.mu1)*L1**2+2*self.b3**2*c2*L1,
#                   -(-self.b1**2*self.b2-3*self.b2*self.b1**2*self.mu1-3*self.b1*self.b2**2*self.mu2+3*self.b1*self.b2**2)*L3**2+(-(2*self.b1**2*self.b2+6*self.b2*self.b1**2*self.mu1+self.b1*self.b2**2-3*self.b2**2*self.b1*self.mu1+3*self.b2**3*self.mu2-6*self.b3*self.b2**2*self.mu2-2*self.b1*self.b3*self.b2-6*self.b2*self.b3*self.b1*self.mu1+6*self.b3*self.b2**2-6*self.b1*self.b2**2*self.mu2+self.b2**3)*L1-(-6*self.b3*self.b1**2*self.mu1+6*self.b1*self.b2**2*self.mu2-3*self.b1**2*self.b2*self.mu2-6*self.b2*self.b1**2*self.mu1+self.b1**3-2*self.b1**2*self.b3+5*self.b1**2*self.b2-6*self.b1*self.b3*self.b2*self.mu2+3*self.b1**3*self.mu1+2*self.b1*self.b2**2+6*self.b1*self.b3*self.b2)*L2-2*self.b1*self.b2**2-4*self.b1*self.b3*self.b2)*L3-(3*self.b1**2*self.b3+self.b1*self.b3*self.b2+3*self.b1*self.b3*self.b2*self.mu2-3*self.b3*self.b1**2*self.mu1)*L2**2+(-(-6*self.b1*self.b3*self.b2*self.mu2+2*self.b3*self.b2**2+2*self.b1**2*self.b3+3*self.b2*self.b3**2+6*self.b3*self.b2**2*self.mu2-self.b1*self.b3**2-3*self.b3**2*self.b1*self.mu1+6*self.b3*self.b1**2*self.mu1-3*self.b3**2*self.b2*self.mu2+4*self.b1*self.b3*self.b2-6*self.b2*self.b3*self.b1*self.mu1)*L1-4*self.b1*self.b3*self.b2)*L2+2*self.b1*self.b3*self.b2-(-self.b3*self.b2**2-3*self.b3*self.b2**2*self.mu2+3*self.b2*self.b3*self.b1*self.mu1+self.b1*self.b3*self.b2)*L1**2-2*self.b2*self.b3**2*L1],
#                  [2*(3*c1*c2*self.mu3+3*c1*c2*self.mu2-2*c2*c1)*L3**2+(2*(-3*c2**2*self.mu2-2*c2**2+8*c2*c1+6*c1*c2*self.mu2+6*c2*c3*self.mu3+6*c2*c3*self.mu2-6*c1*c2*self.mu3-3*c2**2*self.mu3-4*c3*c2)*L1+2*(-4*c3*c1-3*c1**2*self.mu3-6*c1*c2*self.mu2+3*c1**2*self.mu2+4*c1**2-4*c2*c1+6*c1*c3*self.mu3-6*c1*c2*self.mu3+6*c1*c3*self.mu2)*L2-8*c3*c1)*L3+2*(-2*c3*c1-3*c1*c3*self.mu2-3*c1*c3*self.mu3)*L2**2+(2*(-6*c1*c3*self.mu3-6*c2*c3*self.mu2+3*c3**2*self.mu2+6*c1*c3*self.mu2-6*c2*c3*self.mu3+3*c3**2*self.mu3-4*c3*c2+8*c3*c1-2*c3**2)*L1+4*c1**2)*L2+2*c3*c1-2*c2*c1+2*(-3*c2*c3*self.mu3+4*c3*c2+3*c2*c3*self.mu2)*L1**2+2*(4*c2*c1-2*c3**2)*L1,
#                   (3*c1*c2**2+c1*c3*c2-3*c1*c2**2*self.mu2+3*c1*c2*c3*self.mu3)*L3**2+((c2**3-6*c1*c2*c3*self.mu3-6*c1*c2**2*self.mu2-3*c2**2*c3*self.mu3+3*c2**3*self.mu2+6*c2*c3**2*self.mu3+5*c2**2*c3+6*c1*c3*c2-6*c2**2*c3*self.mu2-2*c1*c2**2+2*c2*c3**2)*L1+(-6*c1*c2*c3*self.mu3+6*c1*c3**2*self.mu3-3*c1**2*c3*self.mu3-3*c1**2*c2*self.mu2+6*c1*c2**2*self.mu2+4*c1*c3*c2-c1**2*c2+2*c1*c2**2+2*c1*c3**2-6*c1*c2*c3*self.mu2+3*c1**2*c3)*L2+4*c1*c3*c2)*L3+(-c1*c3**2+3*c1*c2*c3*self.mu2+c1*c3*c2-3*c1*c3**2*self.mu3)*L2**2+((-6*c2*c3**2*self.mu3+6*c2**2*c3*self.mu2+c3**3-6*c1*c3**2*self.mu3-3*c2*c3**2*self.mu2-2*c1*c3*c2+6*c1*c3**2+2*c2**2*c3+3*c3**3*self.mu3+c2*c3**2-6*c1*c2*c3*self.mu2)*L1+2*c1**2*c3)*L2-2*c1*c3*c2+(-c2**2*c3-3*c2*c3**2*self.mu3-3*c2**2*c3*self.mu2+3*c2*c3**2)*L1**2+(4*c1*c3*c2+2*c2*c3**2)*L1,
#                   (3*c1*c2*self.b2*self.mu2-3*c1*c2*self.b3*self.mu3-c1*c2*self.b3-3*c1*c2*self.b2)*L3**2+((6*c1*c2*self.b2*self.mu2+3*c2**2*self.b3*self.mu3+6*c1*c2*self.b3*self.mu3-c2**2*self.b2+c2**2*self.b3+6*c3*self.b2*c2*self.mu2-6*c1*c2*self.b3+2*c1*c2*self.b2-6*c2*self.b3*c3*self.mu3-2*c3*self.b3*c2-6*c3*self.b2*c2-3*c2**2*self.b2*self.mu2)*L1+(-3*c1**2*self.b3-6*c1*self.b2*c3+2*c1*c2*self.b3+3*c1**2*self.b2*self.mu2-6*c1*c2*self.b2*self.mu2-6*c1*c3*self.b3*self.mu3-2*c1*c3*self.b3+6*c1*c3*self.b2*self.mu2-2*c1*c2*self.b2+c1**2*self.b2+6*c1*c2*self.b3*self.mu3+3*c1**2*self.b3*self.mu3)*L2-4*c1*self.b2*c3)*L3+(-3*c1*c3*self.b2*self.mu2+c1*c3*self.b3-c1*self.b2*c3+3*c1*c3*self.b3*self.mu3)*L2**2+((-6*c3*self.b2*c2*self.mu2+6*c1*c3*self.b3*self.mu3-c3**2*self.b3-2*c3*self.b2*c2+2*c1*self.b2*c3+2*c3*self.b3*c2+6*c1*c3*self.b2*self.mu2-3*c3**2*self.b3*self.mu3-6*c1*c3*self.b3+6*c2*self.b3*c3*self.mu3+3*c3**2*self.b2*self.mu2-3*c3**2*self.b2)*L1-2*c1**2*self.b3)*L2+2*c1*self.b2*c3+(-3*c3*self.b3*c2+3*c3*self.b2*c2*self.mu2+3*c2*self.b3*c3*self.mu3+c3*self.b2*c2)*L1**2+(-4*c1*c2*self.b3-2*c3**2*self.b2)*L1,
#                   -2*(3*c1*c2*self.mu3+2*c2*c1+3*c1*c2*self.mu1)*L3**2+(-2*(6*c2*c3*self.mu3-4*c2**2-6*c1*c2*self.mu3-3*c2**2*self.mu3+4*c3*c2+3*c2**2*self.mu1-6*c1*c2*self.mu1+6*c2*c3*self.mu1+4*c2*c1)*L1-2*(6*c1*c2*self.mu1-6*c1*c2*self.mu3+6*c1*c3*self.mu3+6*c1*c3*self.mu1-3*c1**2*self.mu1+4*c3*c1-8*c2*c1-3*c1**2*self.mu3+2*c1**2)*L2+4*c2**2)*L3-2*(3*c1*c3*self.mu1-4*c3*c1-3*c1*c3*self.mu3)*L2**2+(-2*(3*c3**2*self.mu1+3*c3**2*self.mu3-6*c1*c3*self.mu1+6*c2*c3*self.mu1-6*c1*c3*self.mu3+4*c3*c1-8*c3*c2-6*c2*c3*self.mu3+2*c3**2)*L1-4*c1**2+8*c3*c2)*L2+2*c2*c1-2*c3*c2-2*(-3*c2*c3*self.mu1-3*c2*c3*self.mu3+2*c3*c2)*L1**2-8*c2*c1*L1,
#                   (c1*c3*c2-c1**2*c2-3*c1**2*c2*self.mu1+3*c1*c2*c3*self.mu3)*L3**2+((6*c1**2*c2*self.mu1-c2**2*c3-6*c2*c3*c1*self.mu1-3*c2**2*c3*self.mu3-3*c1*c2**2*self.mu1+6*c2*c3**2*self.mu3+4*c1*c3*c2+2*c2*c3**2-6*c1*c2*c3*self.mu3+3*c1*c2**2+2*c1**2*c2)*L1+(-2*c1*c3*c2+6*c1**2*c2+3*c1**3*self.mu1-6*c1*c2*c3*self.mu3+2*c1*c3**2+c1**2*c3+6*c1*c3**2*self.mu3+c1**3-6*c1**2*c3*self.mu1-6*c1**2*c2*self.mu1-3*c1**2*c3*self.mu3)*L2+2*c1*c2**2)*L3+(3*c1**2*c3-3*c1*c3**2*self.mu3-3*c1**2*c3*self.mu1-c1*c3**2)*L2**2+((5*c1*c3**2+3*c3**3*self.mu3-6*c2*c3**2*self.mu3-6*c1*c3**2*self.mu3+6*c1**2*c3*self.mu1+c3**3-6*c2*c3*c1*self.mu1+6*c1*c3*c2-2*c2*c3**2-3*c1*c3**2*self.mu1+2*c1**2*c3)*L1+4*c1*c3*c2+2*c1**2*c3)*L2-2*c1*c3*c2+(3*c2*c3**2+c1*c3*c2+3*c2*c3*c1*self.mu1-3*c2*c3**2*self.mu3)*L1**2+4*c1*c3*c2*L1,
#                   (3*c2*self.b1*c1*self.mu1-c1*c2*self.b3+c1*c2*self.b1-3*c1*c2*self.b3*self.mu3)*L3**2+((-2*c1*c2*self.b1+3*c2**2*self.b3*self.mu3+6*c1*c2*self.b3*self.mu3-6*c2*self.b1*c1*self.mu1+3*c2**2*self.b1*self.mu1+2*c3*c2*self.b1+6*c2*c3*self.b1*self.mu1-3*c2**2*self.b1+c2**2*self.b3-6*c2*self.b3*c3*self.mu3-6*c1*c2*self.b3-2*c3*self.b3*c2)*L1+(-c1**2*self.b1+2*c1*c3*self.b1-3*c1**2*self.b1*self.mu1+6*c2*self.b1*c1*self.mu1-6*c1*c2*self.b1-3*c1**2*self.b3-6*c1*c3*self.b3*self.mu3+6*c1*c2*self.b3*self.mu3+2*c1*c2*self.b3-2*c1*c3*self.b3+6*c3*self.b1*c1*self.mu1+3*c1**2*self.b3*self.mu3)*L2-2*c2**2*self.b1)*L3+(3*c3*self.b1*c1*self.mu1+c1*c3*self.b3+3*c1*c3*self.b3*self.mu3-3*c1*c3*self.b1)*L2**2+((3*c3**2*self.b1*self.mu1+c3**2*self.b1-6*c3*self.b1*c1*self.mu1+6*c1*c3*self.b3*self.mu3-2*c1*c3*self.b1+6*c2*self.b3*c3*self.mu3-6*c3*c2*self.b1+2*c3*self.b3*c2-c3**2*self.b3-3*c3**2*self.b3*self.mu3-6*c1*c3*self.b3+6*c2*c3*self.b1*self.mu1)*L1-2*c1**2*self.b3-4*c3*c2*self.b1)*L2+2*c1*c2*self.b3+(3*c2*self.b3*c3*self.mu3-c3*c2*self.b1-3*c3*self.b3*c2-3*c2*c3*self.b1*self.mu1)*L1**2-4*c1*c2*self.b3*L1,
#                   2*(-3*c1*c2*self.mu2+4*c2*c1+3*c1*c2*self.mu1)*L3**2+(2*(6*c2*c3*self.mu1-2*c2**2+3*c2**2*self.mu2-6*c2*c3*self.mu2+3*c2**2*self.mu1-6*c1*c2*self.mu1-4*c2*c1-6*c1*c2*self.mu2+8*c3*c2)*L1+2*(-2*c1**2+8*c3*c1-6*c1*c3*self.mu2-3*c1**2*self.mu1-4*c2*c1-3*c1**2*self.mu2+6*c1*c2*self.mu1+6*c1*c2*self.mu2+6*c1*c3*self.mu1)*L2-4*c2**2+8*c3*c1)*L3+2*(-2*c3*c1+3*c1*c3*self.mu2+3*c1*c3*self.mu1)*L2**2+(2*(-3*c3**2*self.mu2+3*c3**2*self.mu1-6*c1*c3*self.mu1+6*c2*c3*self.mu1-6*c1*c3*self.mu2-4*c3*c2+4*c3**2-4*c3*c1+6*c2*c3*self.mu2)*L1-8*c3*c2)*L2-2*c3*c1+2*c3*c2+2*(-3*c2*c3*self.mu2-2*c3*c2-3*c2*c3*self.mu1)*L1**2+4*c3**2*L1,
#                   -(3*c1*c2**2*self.mu2-3*c1*c2**2+3*c1**2*c2*self.mu1+c1**2*c2)*L3**2+(-(-2*c1**2*c2-6*c2**2*c3+6*c2**2*c3*self.mu2-c1*c2**2+2*c1*c3*c2-6*c1**2*c2*self.mu1-3*c2**3*self.mu2+3*c1*c2**2*self.mu1+6*c2*c3*c1*self.mu1+6*c1*c2**2*self.mu2-c2**3)*L1-(-5*c1**2*c2-3*c1**3*self.mu1+2*c1**2*c3-c1**3+6*c1*c2*c3*self.mu2-2*c1*c2**2+6*c1**2*c3*self.mu1-6*c1*c3*c2+6*c1**2*c2*self.mu1-6*c1*c2**2*self.mu2+3*c1**2*c2*self.mu2)*L2+2*c1*c2**2+4*c1*c3*c2)*L3-(-3*c1**2*c3-c1*c3*c2-3*c1*c2*c3*self.mu2+3*c1**2*c3*self.mu1)*L2**2+(-(6*c2*c3*c1*self.mu1-6*c1**2*c3*self.mu1+c1*c3**2-2*c1**2*c3-3*c2*c3**2-2*c2**2*c3+6*c1*c2*c3*self.mu2+3*c2*c3**2*self.mu2+3*c1*c3**2*self.mu1-4*c1*c3*c2-6*c2**2*c3*self.mu2)*L1+4*c1*c3*c2)*L2-2*c1*c3*c2-(-3*c2*c3*c1*self.mu1+3*c2**2*c3*self.mu2+c2**2*c3-c1*c3*c2)*L1**2+2*c2*c3**2*L1,
#                   (3*c1*c2*self.b2*self.mu2-3*c1*c2*self.b2+c1*c2*self.b1+3*c2*self.b1*c1*self.mu1)*L3**2+((-2*c1*c2*self.b1+6*c3*self.b2*c2*self.mu2-3*c2**2*self.b2*self.mu2-6*c3*self.b2*c2-6*c2*self.b1*c1*self.mu1-c2**2*self.b2+2*c1*c2*self.b2+3*c2**2*self.b1*self.mu1+2*c3*c2*self.b1+6*c2*c3*self.b1*self.mu1-3*c2**2*self.b1+6*c1*c2*self.b2*self.mu2)*L1+(6*c3*self.b1*c1*self.mu1-c1**2*self.b1+2*c1*c3*self.b1-6*c1*c2*self.b1-3*c1**2*self.b1*self.mu1-6*c1*c2*self.b2*self.mu2-6*c1*self.b2*c3-2*c1*c2*self.b2+3*c1**2*self.b2*self.mu2+6*c1*c3*self.b2*self.mu2+6*c2*self.b1*c1*self.mu1+c1**2*self.b2)*L2-2*c2**2*self.b1-4*c1*self.b2*c3)*L3+(-3*c1*c3*self.b2*self.mu2+3*c3*self.b1*c1*self.mu1-c1*self.b2*c3-3*c1*c3*self.b1)*L2**2+((3*c3**2*self.b2*self.mu2+6*c1*c3*self.b2*self.mu2+6*c2*c3*self.b1*self.mu1-2*c1*c3*self.b1+c3**2*self.b1-2*c3*self.b2*c2+2*c1*self.b2*c3-3*c3**2*self.b2-6*c3*self.b1*c1*self.mu1+3*c3**2*self.b1*self.mu1-6*c3*c2*self.b1-6*c3*self.b2*c2*self.mu2)*L1-4*c3*c2*self.b1)*L2+2*c3*c2*self.b1+(c3*self.b2*c2+3*c3*self.b2*c2*self.mu2-c3*c2*self.b1-3*c2*c3*self.b1*self.mu1)*L1**2-2*c3**2*self.b2*L1],
#                  [-(2*self.b2*c1-3*c2*self.b1*self.mu3+2*c2*self.b1-3*c1*self.b2*self.mu2-3*c2*self.b1*self.mu2-3*c1*self.b2*self.mu3)*L3**2+(-(-6*c1*self.b2*self.mu2+4*c2*self.b2-8*self.b2*c1-6*c2*self.b1*self.mu2+6*c2*self.b2*self.mu3+6*c2*self.b2*self.mu2+6*c2*self.b1*self.mu3+6*c1*self.b2*self.mu3-8*c2*self.b1-6*c3*self.b2*self.mu2+4*c2*self.b3+4*self.b2*c3-6*c2*self.b3*self.mu2-6*c2*self.b3*self.mu3-6*c3*self.b2*self.mu3)*L1-(6*c1*self.b1*self.mu3+6*c2*self.b1*self.mu3+4*self.b3*c1+6*c2*self.b1*self.mu2+6*c1*self.b2*self.mu3-6*c1*self.b3*self.mu2-6*c3*self.b1*self.mu2-6*c1*self.b1*self.mu2+4*c3*self.b1-8*c1*self.b1+6*c1*self.b2*self.mu2-6*c1*self.b3*self.mu3+4*self.b2*c1+4*c2*self.b1-6*c3*self.b1*self.mu3)*L2-4*c3*self.b1-4*self.b3*c1)*L3-(3*c3*self.b1*self.mu3+2*self.b3*c1+3*c1*self.b3*self.mu3+3*c3*self.b1*self.mu2+2*c3*self.b1+3*c1*self.b3*self.mu2)*L2**2+(-(-8*c3*self.b1-8*self.b3*c1+6*c1*self.b3*self.mu3-6*c1*self.b3*self.mu2-6*c3*self.b3*self.mu3+6*c2*self.b3*self.mu3+4*c3*self.b3+4*c2*self.b3+6*c3*self.b1*self.mu3+4*self.b2*c3+6*c3*self.b2*self.mu3-6*c3*self.b1*self.mu2+6*c3*self.b2*self.mu2+6*c2*self.b3*self.mu2-6*c3*self.b3*self.mu2)*L1+4*c1*self.b1)*L2-self.b2*c1+c3*self.b1-c2*self.b1+self.b3*c1-(-4*self.b2*c3-3*c3*self.b2*self.mu2+3*c2*self.b3*self.mu3-4*c2*self.b3+3*c3*self.b2*self.mu3-3*c2*self.b3*self.mu2)*L1**2-(4*c3*self.b3-4*self.b2*c1-4*c2*self.b1)*L1,
#                   -0.5*(-3*c2**2*self.b1+3*c2**2*self.b1*self.mu2-3*c2*self.b1*c3*self.mu3-3*c1*c3*self.b2*self.mu3-c1*self.b2*c3-c3*c2*self.b1-3*c1*c2*self.b2+3*c1*c2*self.b2*self.mu2)*L3**2+(-0.5*(6*c2*c3*self.b2*self.mu3-2*c3*self.b3*c2-6*c3*c2*self.b1-2*c2**2*self.b2+2*c2**2*self.b1-6*c2**2*self.b2*self.mu2+6*c1*c3*self.b2*self.mu3+2*c1*c2*self.b2-2*c3**2*self.b2-6*c2**2*self.b3+6*c2*self.b1*c3*self.mu3+6*c2**2*self.b1*self.mu2-4*c3*self.b2*c2-6*c3**2*self.b2*self.mu3+6*c1*c2*self.b2*self.mu2-6*c1*self.b2*c3+6*c2**2*self.b3*self.mu2-6*c2*self.b3*c3*self.mu3+6*c3*self.b2*c2*self.mu2)*L1-0.5*(6*c1*c2*self.b1*self.mu2-4*c3*c2*self.b1-2*c1*c3*self.b3+6*c3*self.b1*c2*self.mu2-6*c1*c3*self.b1-2*c2**2*self.b1+6*c2*self.b1*c3*self.mu3+6*c1*c3*self.b1*self.mu3-2*c3**2*self.b1-6*c1*c2*self.b3-6*c1*c3*self.b3*self.mu3+6*c1*c3*self.b2*self.mu3-2*c1*c2*self.b2+6*c1*c2*self.b3*self.mu2-6*c2**2*self.b1*self.mu2-6*c1*c2*self.b2*self.mu2+2*c1*c2*self.b1-6*c3**2*self.b1*self.mu3+2*c1*self.b2*c3)*L2+2*c1*c2*self.b3+2*c3*c2*self.b1)*L3-0.5*(3*c1*c3*self.b3*self.mu3+c3**2*self.b1-3*c3*self.b1*c2*self.mu2-c3*c2*self.b1-3*c1*c2*self.b3*self.mu2+c1*c3*self.b3-c1*c2*self.b3+3*c3**2*self.b1*self.mu3)*L2**2+(-0.5*(-6*c3**2*self.b1-6*c2**2*self.b3*self.mu2+6*c1*c3*self.b3*self.mu3-6*c1*c3*self.b3+2*c1*c2*self.b3-2*c3**2*self.b3-6*c3**2*self.b3*self.mu3-4*c3*self.b3*c2-6*c3*self.b2*c2*self.mu2+6*c3*self.b1*c2*self.mu2+6*c3**2*self.b1*self.mu3+6*c3**2*self.b2*self.mu3-2*c2**2*self.b3+6*c1*c2*self.b3*self.mu2+2*c3**2*self.b2+2*c3*c2*self.b1-2*c3*self.b2*c2+6*c2*self.b3*c3*self.mu3+6*c3*c2*self.b3*self.mu2)*L1+2*c1*c3*self.b1)*L2-c3*c2*self.b1-c1*c2*self.b3-0.5*(3*c3**2*self.b2*self.mu3+3*c3*self.b2*c2*self.mu2+c3*self.b2*c2+3*c2*self.b3*c3*self.mu3+3*c2**2*self.b3*self.mu2-3*c3*self.b3*c2-3*c3**2*self.b2+c2**2*self.b3)*L1**2-0.5*(-4*c3*self.b3*c2-4*c3*c2*self.b1-4*c1*self.b2*c3)*L1,
#                   0.5*(-3*self.b1*c2*self.b2-3*c1*self.b2**2-c1*self.b2*self.b3+3*c1*self.b2**2*self.mu2+3*self.b1*c2*self.b2*self.mu2-self.b1*c2*self.b3-3*c1*self.b2*self.b3*self.mu3-3*c2*self.b1*self.b3*self.mu3)*L3**2+(0.5*(2*c1*self.b2**2+2*self.b1*c2*self.b2-4*self.b2*c2*self.b3-6*c1*self.b2*self.b3-2*c2*self.b2**2+6*c1*self.b2**2*self.mu2-6*c3*self.b2**2-6*c2*self.b2**2*self.mu2+6*self.b2*c2*self.b3*self.mu2+6*c3*self.b2**2*self.mu2-6*self.b2*self.b3*c3*self.mu3-2*self.b2*self.b3*c3-6*c2*self.b3**2*self.mu3+6*c2*self.b2*self.b3*self.mu3+6*self.b1*c2*self.b2*self.mu2-2*self.b3**2*c2+6*c2*self.b1*self.b3*self.mu3+6*c1*self.b2*self.b3*self.mu3-6*self.b1*c2*self.b3)*L1+0.5*(-4*c1*self.b2*self.b3-2*c1*self.b2**2-6*c1*self.b1*self.b3+6*c1*self.b1*self.b2*self.mu2+6*c3*self.b1*self.b2*self.mu2+2*c1*self.b1*self.b2+6*c1*self.b2*self.b3*self.mu2-2*c1*self.b3**2-6*self.b1*self.b2*c3-6*self.b1*c3*self.b3*self.mu3-6*c1*self.b3**2*self.mu3-2*self.b1*c3*self.b3+6*c2*self.b1*self.b3*self.mu3-6*c1*self.b2**2*self.mu2+2*self.b1*c2*self.b3+6*c1*self.b2*self.b3*self.mu3-6*self.b1*c2*self.b2*self.mu2-2*self.b1*c2*self.b2+6*c1*self.b1*self.b3*self.mu3)*L2-2*self.b1*self.b2*c3-2*c1*self.b2*self.b3)*L3+0.5*(-3*c3*self.b1*self.b2*self.mu2+self.b1*c3*self.b3+c1*self.b3**2-self.b1*self.b2*c3+3*self.b1*c3*self.b3*self.mu3-3*c1*self.b2*self.b3*self.mu2+3*c1*self.b3**2*self.mu3-c1*self.b2*self.b3)*L2**2+(0.5*(-2*c3*self.b3**2-6*self.b1*c3*self.b3-6*self.b2*c2*self.b3*self.mu2+6*self.b1*c3*self.b3*self.mu3-6*c3*self.b3**2*self.mu3-4*self.b2*self.b3*c3+6*c3*self.b2*self.b3*self.mu2+6*c1*self.b2*self.b3*self.mu2+6*c1*self.b3**2*self.mu3-6*c3*self.b2**2*self.mu2+2*self.b1*self.b2*c3-6*c1*self.b3**2+6*c3*self.b1*self.b2*self.mu2+6*c2*self.b3**2*self.mu3+6*self.b2*self.b3*c3*self.mu3-2*c3*self.b2**2+2*self.b3**2*c2+2*c1*self.b2*self.b3-2*self.b2*c2*self.b3)*L1-2*c1*self.b1*self.b3)*L2+self.b1*self.b2*c3+c1*self.b2*self.b3+0.5*(3*self.b2*c2*self.b3*self.mu2+3*self.b2*self.b3*c3*self.mu3+3*c3*self.b2**2*self.mu2-3*self.b2*self.b3*c3-3*self.b3**2*c2+c3*self.b2**2+3*c2*self.b3**2*self.mu3+self.b2*c2*self.b3)*L1**2+0.5*(-4*self.b1*c2*self.b3-4*c1*self.b2*self.b3-4*self.b2*self.b3*c3)*L1,
#                   (-2*c2*self.b1-3*c1*self.b2*self.mu3-3*c2*self.b1*self.mu3-3*c2*self.b1*self.mu1-3*c1*self.b2*self.mu1-2*self.b2*c1)*L3**2+((6*c2*self.b2*self.mu3-6*c3*self.b2*self.mu1-4*c2*self.b3-4*c2*self.b1-6*c2*self.b2*self.mu1-4*self.b2*c1+6*c1*self.b2*self.mu3+6*c2*self.b1*self.mu3-6*c2*self.b3*self.mu1+8*c2*self.b2+6*c1*self.b2*self.mu1-6*c2*self.b3*self.mu3-4*self.b2*c3-6*c3*self.b2*self.mu3+6*c2*self.b1*self.mu1)*L1+(-4*c3*self.b1+6*c2*self.b1*self.mu3+6*c1*self.b2*self.mu3-4*self.b3*c1-6*c3*self.b1*self.mu3-6*c2*self.b1*self.mu1-6*c3*self.b1*self.mu1-6*c1*self.b3*self.mu3-6*c1*self.b2*self.mu1-6*c1*self.b3*self.mu1+6*c1*self.b1*self.mu3+8*self.b2*c1+8*c2*self.b1+6*c1*self.b1*self.mu1-4*c1*self.b1)*L2+4*c2*self.b2)*L3+(-3*c1*self.b3*self.mu1-3*c3*self.b1*self.mu1+4*self.b3*c1+3*c1*self.b3*self.mu3+3*c3*self.b1*self.mu3+4*c3*self.b1)*L2**2+((-6*c3*self.b3*self.mu3-4*self.b3*c1+8*self.b2*c3-4*c3*self.b1+6*c3*self.b2*self.mu3+6*c3*self.b1*self.mu3-6*c2*self.b3*self.mu1+6*c1*self.b3*self.mu3-4*c3*self.b3+6*c3*self.b1*self.mu1-6*c3*self.b2*self.mu1+6*c2*self.b3*self.mu3-6*c3*self.b3*self.mu1+8*c2*self.b3+6*c1*self.b3*self.mu1)*L1-4*c1*self.b1+4*self.b2*c3+4*c2*self.b3)*L2-c2*self.b3+c2*self.b1-self.b2*c3+self.b2*c1+(3*c2*self.b3*self.mu1+3*c3*self.b2*self.mu3-2*self.b2*c3+3*c2*self.b3*self.mu3-2*c2*self.b3+3*c3*self.b2*self.mu1)*L1**2+(-4*self.b2*c1-4*c2*self.b1)*L1,
#                   0.5*(-c1**2*self.b2+c3*c2*self.b1+3*c2*self.b1*c3*self.mu3-c1*c2*self.b1+c1*self.b2*c3+3*c1*c3*self.b2*self.mu3-3*self.b2*c1**2*self.mu1-3*c2*self.b1*c1*self.mu1)*L3**2+(0.5*(-6*c1*c3*self.b2*self.mu3+6*self.b2*c1**2*self.mu1+2*c1*c2*self.b1+6*c1*c2*self.b2+2*c3*self.b3*c2-2*c3*self.b2*c2-2*c1*c2*self.b3+4*c1*self.b2*c3+6*c3*c2*self.b1-6*c2*self.b2*c1*self.mu1+2*c1**2*self.b2-6*c3*self.b2*c1*self.mu1+2*c3**2*self.b2-6*c2*self.b1*c3*self.mu3-6*c2*self.b3*c1*self.mu1+6*c2*self.b3*c3*self.mu3-6*c2*c3*self.b2*self.mu3+6*c3**2*self.b2*self.mu3+6*c2*self.b1*c1*self.mu1)*L1+0.5*(-2*c3*c2*self.b1+6*c1*c2*self.b1+6*c1*c3*self.b3*self.mu3+2*c1*c3*self.b3+4*c1*c3*self.b1-2*c1*self.b2*c3+2*c3**2*self.b1-6*c3*self.b1*c1*self.mu1-6*self.b2*c1**2*self.mu1-6*c2*self.b1*c1*self.mu1-6*self.b3*c1**2*self.mu1-6*c1*c3*self.b2*self.mu3+6*c3**2*self.b1*self.mu3+6*c1**2*self.b2-6*c1*c3*self.b1*self.mu3+2*self.b1*c1**2+6*self.b1*c1**2*self.mu1-6*c2*self.b1*c3*self.mu3-2*c1**2*self.b3)*L2+2*c1*c2*self.b2)*L3+0.5*(3*c1**2*self.b3-c1*c3*self.b3+3*c1*c3*self.b1-3*c3*self.b1*c1*self.mu1-3*c3**2*self.b1*self.mu3-3*c1*c3*self.b3*self.mu3-3*self.b3*c1**2*self.mu1-c3**2*self.b1)*L2**2+(0.5*(6*c1*c2*self.b3+6*c1*self.b2*c3-6*c3*self.b3*c1*self.mu1+6*c3**2*self.b1-2*c3*self.b3*c2+6*c3*self.b1*c1*self.mu1+2*c1*c3*self.b1-6*c3**2*self.b2*self.mu3+2*c1**2*self.b3-6*c1*c3*self.b3*self.mu3-6*c3*self.b2*c1*self.mu1-2*c3**2*self.b2+2*c3**2*self.b3+6*self.b3*c1**2*self.mu1+6*c3**2*self.b3*self.mu3+4*c1*c3*self.b3-6*c2*self.b3*c1*self.mu1-6*c2*self.b3*c3*self.mu3-6*c3**2*self.b1*self.mu3)*L1+2*c1*self.b2*c3+2*c1*c3*self.b1+2*c1*c2*self.b3)*L2-c1*self.b2*c3-c3*c2*self.b1+0.5*(c1*c2*self.b3+c1*self.b2*c3+3*c3*self.b2*c1*self.mu1+3*c2*self.b3*c1*self.mu1-3*c3**2*self.b2*self.mu3+3*c3**2*self.b2+3*c3*self.b3*c2-3*c2*self.b3*c3*self.mu3)*L1**2+0.5*(4*c1*self.b2*c3+4*c3*c2*self.b1)*L1,
#                   -0.5*(c1*self.b2*self.b3-c2*self.b1**2+self.b1*c2*self.b3+3*c1*self.b2*self.b3*self.mu3-3*self.b1*c1*self.b2*self.mu1-3*c2*self.b1**2*self.mu1+3*c2*self.b1*self.b3*self.mu3-c1*self.b1*self.b2)*L3**2+(-0.5*(2*c1*self.b1*self.b2-6*c1*self.b2*self.b3*self.mu3+6*c1*self.b2*self.b3-2*self.b2*c2*self.b3+2*self.b2*self.b3*c3+4*self.b1*c2*self.b3+6*self.b1*c2*self.b2-6*c2*self.b1*self.b3*self.mu3-6*c2*self.b2*self.b3*self.mu3+6*c2*self.b3**2*self.mu3-2*self.b1*self.b2*c3+6*self.b1*c1*self.b2*self.mu1+2*self.b3**2*c2-6*c3*self.b2*self.b1*self.mu1-6*c2*self.b3*self.b1*self.mu1+2*c2*self.b1**2+6*self.b2*self.b3*c3*self.mu3-6*c2*self.b2*self.b1*self.mu1+6*c2*self.b1**2*self.mu1)*L1-0.5*(-2*self.b1*c2*self.b3+6*c2*self.b1**2-6*c1*self.b2*self.b3*self.mu3+6*c1*self.b1**2*self.mu1+6*self.b1*c3*self.b3*self.mu3-6*c2*self.b1**2*self.mu1-6*c2*self.b1*self.b3*self.mu3+2*self.b1*c3*self.b3+6*c1*self.b1*self.b2-2*c1*self.b2*self.b3-6*c3*self.b1**2*self.mu1+6*c1*self.b3**2*self.mu3+2*c1*self.b1**2+4*c1*self.b1*self.b3-6*c1*self.b1*self.b3*self.mu3+2*c1*self.b3**2-6*self.b1*c1*self.b2*self.mu1-6*self.b1*c1*self.b3*self.mu1-2*c3*self.b1**2)*L2-2*self.b1*c2*self.b2)*L3-0.5*(-3*c3*self.b1**2*self.mu1-3*c1*self.b3**2*self.mu3-c1*self.b3**2+3*c1*self.b1*self.b3-3*self.b1*c3*self.b3*self.mu3-3*self.b1*c1*self.b3*self.mu1+3*c3*self.b1**2-self.b1*c3*self.b3)*L2**2+(-0.5*(2*c3*self.b3**2+6*c3*self.b3**2*self.mu3-6*self.b2*self.b3*c3*self.mu3+6*self.b1*c2*self.b3+6*c3*self.b1**2*self.mu1-2*self.b3**2*c2+2*c1*self.b1*self.b3-6*c2*self.b3*self.b1*self.mu1-6*c3*self.b3*self.b1*self.mu1-6*c1*self.b3**2*self.mu3+6*self.b1*self.b2*c3-6*c3*self.b2*self.b1*self.mu1+4*self.b1*c3*self.b3+6*c1*self.b3**2-6*self.b1*c3*self.b3*self.mu3+2*c3*self.b1**2-2*self.b2*self.b3*c3+6*self.b1*c1*self.b3*self.mu1-6*c2*self.b3**2*self.mu3)*L1-2*c1*self.b1*self.b3-2*self.b1*self.b2*c3-2*self.b1*c2*self.b3)*L2+c1*self.b2*self.b3+self.b1*c2*self.b3-0.5*(self.b1*self.b2*c3-3*self.b2*self.b3*c3*self.mu3+3*self.b2*self.b3*c3+self.b1*c2*self.b3+3*self.b3**2*c2-3*c2*self.b3**2*self.mu3+3*c2*self.b3*self.b1*self.mu1+3*c3*self.b2*self.b1*self.mu1)*L1**2-0.5*(4*c1*self.b2*self.b3+4*self.b1*c2*self.b3)*L1,
#                   -(-3*c2*self.b1*self.mu1+3*c2*self.b1*self.mu2-4*c2*self.b1-4*self.b2*c1-3*c1*self.b2*self.mu1+3*c1*self.b2*self.mu2)*L3**2+(-(-6*c3*self.b2*self.mu1+4*self.b2*c1-8*self.b2*c3+4*c2*self.b1+6*c3*self.b2*self.mu2+6*c1*self.b2*self.mu1-6*c2*self.b2*self.mu1+6*c2*self.b3*self.mu2+6*c2*self.b1*self.mu1+6*c2*self.b1*self.mu2-6*c2*self.b3*self.mu1+6*c1*self.b2*self.mu2-8*c2*self.b3-6*c2*self.b2*self.mu2+4*c2*self.b2)*L1-(-6*c3*self.b1*self.mu1+6*c1*self.b1*self.mu2-6*c1*self.b2*self.mu1-6*c2*self.b1*self.mu2+4*c1*self.b1+6*c1*self.b3*self.mu2+6*c1*self.b1*self.mu1-6*c1*self.b3*self.mu1-8*c3*self.b1-8*self.b3*c1+4*c2*self.b1-6*c2*self.b1*self.mu1+4*self.b2*c1+6*c3*self.b1*self.mu2-6*c1*self.b2*self.mu2)*L2-4*c2*self.b2+4*c3*self.b1+4*self.b3*c1)*L3-(-3*c3*self.b1*self.mu2+2*self.b3*c1-3*c1*self.b3*self.mu1-3*c3*self.b1*self.mu1+2*c3*self.b1-3*c1*self.b3*self.mu2)*L2**2+(-(4*c3*self.b1+4*self.b3*c1+4*c2*self.b3+6*c3*self.b3*self.mu2-8*c3*self.b3+6*c3*self.b1*self.mu1-6*c2*self.b3*self.mu2-6*c2*self.b3*self.mu1+6*c1*self.b3*self.mu1-6*c3*self.b3*self.mu1+6*c1*self.b3*self.mu2-6*c3*self.b2*self.mu1+4*self.b2*c3+6*c3*self.b1*self.mu2-6*c3*self.b2*self.mu2)*L1-4*c2*self.b3-4*self.b2*c3)*L2+c2*self.b3-self.b3*c1-c3*self.b1+self.b2*c3-(2*c2*self.b3+3*c3*self.b2*self.mu1+3*c3*self.b2*self.mu2+2*self.b2*c3+3*c2*self.b3*self.mu2+3*c2*self.b3*self.mu1)*L1**2+4*c3*self.b3*L1,
#                   0.5*(-c1**2*self.b2-3*c2*self.b1*c1*self.mu1-3*c1*c2*self.b2*self.mu2+3*c1*c2*self.b2-c1*c2*self.b1-3*c2**2*self.b1*self.mu2+3*c2**2*self.b1-3*self.b2*c1**2*self.mu1)*L3**2+(0.5*(2*c1*c2*self.b1+4*c1*c2*self.b2-6*c3*self.b2*c2*self.mu2+6*self.b2*c1**2*self.mu1-2*c2**2*self.b1+2*c1**2*self.b2+6*c2**2*self.b3+6*c3*self.b2*c2-6*c2*self.b3*c1*self.mu1-2*c1*self.b2*c3-6*c2**2*self.b1*self.mu2-6*c2*self.b2*c1*self.mu1+2*c2**2*self.b2+6*c2*self.b1*c1*self.mu1-6*c3*self.b2*c1*self.mu1-2*c1*c2*self.b3-6*c2**2*self.b3*self.mu2+6*c2**2*self.b2*self.mu2-6*c1*c2*self.b2*self.mu2)*L1+0.5*(4*c1*c2*self.b1+6*c1*c2*self.b3+6*c2**2*self.b1*self.mu2-6*c3*self.b1*c2*self.mu2+6*c1**2*self.b2-6*c3*self.b1*c1*self.mu1+6*c3*c2*self.b1-2*c1*c3*self.b1-2*c1**2*self.b3-6*self.b3*c1**2*self.mu1+6*self.b1*c1**2*self.mu1-6*self.b2*c1**2*self.mu1-6*c1*c2*self.b3*self.mu2+6*c1*c2*self.b2*self.mu2-6*c2*self.b1*c1*self.mu1+2*c1*c2*self.b2-6*c1*c2*self.b1*self.mu2+2*c2**2*self.b1+2*self.b1*c1**2)*L2+2*c1*c2*self.b3+2*c1*c2*self.b2+2*c3*c2*self.b1)*L3+0.5*(-3*c3*self.b1*c1*self.mu1+3*c1*c2*self.b3*self.mu2+3*c1**2*self.b3+c3*c2*self.b1+3*c1*c3*self.b1+c1*c2*self.b3-3*self.b3*c1**2*self.mu1+3*c3*self.b1*c2*self.mu2)*L2**2+(0.5*(-2*c3*c2*self.b1+4*c1*c2*self.b3-2*c1*c3*self.b3+2*c1**2*self.b3+6*c3*self.b3*c2+6*c1*self.b2*c3-6*c2*self.b3*c1*self.mu1+2*c1*c3*self.b1+6*c3*self.b1*c1*self.mu1-6*c3*self.b3*c1*self.mu1-6*c3*self.b2*c1*self.mu1-6*c3*self.b1*c2*self.mu2+6*c3*self.b2*c2*self.mu2+2*c3*self.b2*c2+2*c2**2*self.b3+6*self.b3*c1**2*self.mu1+6*c2**2*self.b3*self.mu2-6*c1*c2*self.b3*self.mu2-6*c3*c2*self.b3*self.mu2)*L1+2*c1*self.b2*c3+2*c1*c2*self.b3)*L2-c1*self.b2*c3-c1*c2*self.b3+0.5*(c1*c2*self.b3+3*c2*self.b3*c1*self.mu1-c3*self.b2*c2+3*c3*self.b2*c1*self.mu1-3*c2**2*self.b3*self.mu2-c2**2*self.b3-3*c3*self.b2*c2*self.mu2+c1*self.b2*c3)*L1**2+2*c3*self.b3*c2*L1,
#                   -0.5*(-3*self.b1*c2*self.b2*self.mu2-c2*self.b1**2+3*c1*self.b2**2-3*c1*self.b2**2*self.mu2-c1*self.b1*self.b2-3*c2*self.b1**2*self.mu1-3*self.b1*c1*self.b2*self.mu1+3*self.b1*c2*self.b2)*L3**2+(-0.5*(-6*c1*self.b2**2*self.mu2+2*c2*self.b1**2+2*c2*self.b2**2-2*c1*self.b2**2-2*self.b1*self.b2*c3+6*c3*self.b2**2+6*c2*self.b2**2*self.mu2+4*self.b1*c2*self.b2+6*c2*self.b1**2*self.mu1-6*c2*self.b3*self.b1*self.mu1-6*c2*self.b2*self.b1*self.mu1+6*self.b1*c1*self.b2*self.mu1-6*c3*self.b2**2*self.mu2+2*c1*self.b1*self.b2-6*self.b1*c2*self.b2*self.mu2+6*self.b2*c2*self.b3-6*self.b2*c2*self.b3*self.mu2-6*c3*self.b2*self.b1*self.mu1-2*self.b1*c2*self.b3)*L1-0.5*(2*c1*self.b2**2+6*c2*self.b1**2+2*self.b1*c2*self.b2-6*c1*self.b2*self.b3*self.mu2-6*c3*self.b1**2*self.mu1+4*c1*self.b1*self.b2+6*c1*self.b1**2*self.mu1-6*c2*self.b1**2*self.mu1-6*c3*self.b1*self.b2*self.mu2+6*self.b1*self.b2*c3-6*self.b1*c1*self.b2*self.mu1+6*c1*self.b2*self.b3+2*c1*self.b1**2-6*self.b1*c1*self.b3*self.mu1-2*c1*self.b1*self.b3-2*c3*self.b1**2-6*c1*self.b1*self.b2*self.mu2+6*self.b1*c2*self.b2*self.mu2+6*c1*self.b2**2*self.mu2)*L2-2*self.b1*c2*self.b2-2*c1*self.b2*self.b3-2*self.b1*self.b2*c3)*L3-0.5*(3*c1*self.b1*self.b3+3*c3*self.b1*self.b2*self.mu2-3*self.b1*c1*self.b3*self.mu1+3*c1*self.b2*self.b3*self.mu2-3*c3*self.b1**2*self.mu1+self.b1*self.b2*c3+3*c3*self.b1**2+c1*self.b2*self.b3)*L2**2+(-0.5*(2*c3*self.b1**2+6*self.b1*c1*self.b3*self.mu1+6*c3*self.b1**2*self.mu1-6*c1*self.b2*self.b3*self.mu2-2*c1*self.b2*self.b3-6*c2*self.b3*self.b1*self.mu1+2*c3*self.b2**2+6*self.b1*c2*self.b3+6*self.b2*c2*self.b3*self.mu2-2*self.b1*c3*self.b3+2*c1*self.b1*self.b3+6*self.b2*self.b3*c3+2*self.b2*c2*self.b3-6*c3*self.b1*self.b2*self.mu2-6*c3*self.b2*self.b3*self.mu2-6*c3*self.b3*self.b1*self.mu1+4*self.b1*self.b2*c3-6*c3*self.b2*self.b1*self.mu1+6*c3*self.b2**2*self.mu2)*L1-2*self.b1*self.b2*c3-2*self.b1*c2*self.b3)*L2+self.b1*self.b2*c3+self.b1*c2*self.b3-0.5*(self.b1*c2*self.b3-c3*self.b2**2+3*c2*self.b3*self.b1*self.mu1-3*c3*self.b2**2*self.mu2-3*self.b2*c2*self.b3*self.mu2+self.b1*self.b2*c3+3*c3*self.b2*self.b1*self.mu1-self.b2*c2*self.b3)*L1**2-2*self.b2*self.b3*c3*L1]])

        B =array([[-2*(-3*b1*b2*mu3+2*b1*b2-3*b1*b2*mu2)*L3**2+(-2*(4*b3*b2-6*b2*b3*mu2+2*b2**2-8*b1*b2-6*b2*b3*mu3+3*b2**2*mu3+3*b2**2*mu2-6*b1*b2*mu2+6*b1*b2*mu3)*L1-2*(-3*b1**2*mu2+4*b1*b2+4*b1*b3+6*b1*b2*mu2-4*b1**2+6*b1*b2*mu3-6*b1*b3*mu2+3*b1**2*mu3-6*b1*b3*mu3)*L2-8*b1*b3)*L3-2*(2*b1*b3+3*b1*b3*mu2+3*b1*b3*mu3)*L2**2+(-2*(2*b3**2-3*b3**2*mu2-8*b1*b3-3*b3**2*mu3+4*b3*b2+6*b1*b3*mu3+6*b2*b3*mu2+6*b2*b3*mu3-6*b1*b3*mu2)*L1+4*b1**2)*L2-2*b1*b2+2*b1*b3-2*(-3*b2*b3*mu2-4*b3*b2+3*b2*b3*mu3)*L1**2-2*(-4*b1*b2+2*b3**2)*L1,
                   -(3*b1*c2*b2*mu2-b1*b2*c3-3*b1*c3*b2*mu3-3*b1*c2*b2)*L3**2+(-(-c2*b2**2+6*b2*c2*b3*mu2+6*b1*c2*b2*mu2-6*b1*b2*c3+2*b1*c2*b2-6*b2*b3*c3*mu3-3*c2*b2**2*mu2+6*b1*c3*b2*mu3+3*c3*b2**2*mu3+c3*b2**2-6*b2*c2*b3-2*b2*b3*c3)*L1-(3*c2*b1**2*mu2-2*b1*c2*b2+c2*b1**2-6*b1*c2*b2*mu2+6*b1*c2*b3*mu2-6*b1*c3*b3*mu3-2*b1*c3*b3+2*b1*b2*c3+6*b1*c3*b2*mu3-3*c3*b1**2+3*c3*b1**2*mu3-6*b1*c2*b3)*L2+4*b1*c2*b3)*L3-(-3*b1*c2*b3*mu2-b1*c2*b3+b1*c3*b3+3*b1*c3*b3*mu3)*L2**2+(-(2*b1*c2*b3-3*b3**2*c2+6*b2*b3*c3*mu3-2*b2*c2*b3-6*b2*c2*b3*mu2-c3*b3**2-6*b1*c3*b3+2*b2*b3*c3+6*b1*c2*b3*mu2+6*b1*c3*b3*mu3+3*c2*b3**2*mu2-3*c3*b3**2*mu3)*L1+2*c3*b1**2)*L2-2*b1*c2*b3-(3*b2*b3*c3*mu3+b2*c2*b3+3*b2*c2*b3*mu2-3*b2*b3*c3)*L1**2-(-4*b1*b2*c3-2*b3**2*c2)*L1,
                   (-b1*b3*b2+3*b1*b2**2*mu2-3*b1*b2*b3*mu3-3*b1*b2**2)*L3**2+((6*b1*b2**2*mu2+6*b1*b2*b3*mu3-6*b2*b3**2*mu3-6*b1*b3*b2+2*b1*b2**2-2*b2*b3**2-3*b2**3*mu2-5*b3*b2**2-b2**3+3*b2**2*b3*mu3+6*b3*b2**2*mu2)*L1+(-4*b1*b3*b2+6*b1*b3*b2*mu2-6*b1*b2**2*mu2-6*b1*b3**2*mu3+b1**2*b2-2*b1*b3**2-2*b1*b2**2+3*b1**2*b3*mu3-3*b1**2*b3+3*b1**2*b2*mu2+6*b1*b2*b3*mu3)*L2-4*b1*b3*b2)*L3+(-3*b1*b3*b2*mu2-b1*b3*b2+3*b1*b3**2*mu3+b1*b3**2)*L2**2+((-b2*b3**2+6*b1*b3*b2*mu2-b3**3+2*b1*b3*b2+6*b2*b3**2*mu3-3*b3**3*mu3-6*b1*b3**2-2*b3*b2**2+6*b1*b3**2*mu3+3*b3**2*b2*mu2-6*b3*b2**2*mu2)*L1-2*b1**2*b3)*L2+2*b1*b3*b2+(3*b3*b2**2*mu2+3*b2*b3**2*mu3-3*b2*b3**2+b3*b2**2)*L1**2+(-4*b1*b3*b2-2*b2*b3**2)*L1,
                   2*(-3*b1*b2*mu3-3*b1*b2*mu1-2*b1*b2)*L3**2+(2*(-3*b2**2*mu1+6*b1*b2*mu3+6*b1*b2*mu1-4*b3*b2+4*b2**2-4*b1*b2-6*b2*b3*mu3-6*b2*b3*mu1+3*b2**2*mu3)*L1+2*(3*b1**2*mu1-6*b1*b3*mu3-2*b1**2-6*b1*b2*mu1-6*b1*b3*mu1+6*b1*b2*mu3+3*b1**2*mu3+8*b1*b2-4*b1*b3)*L2+4*b2**2)*L3+2*(4*b1*b3-3*b1*b3*mu1+3*b1*b3*mu3)*L2**2+(2*(-3*b3**2*mu3-6*b2*b3*mu1-3*b3**2*mu1-4*b1*b3+6*b2*b3*mu3+6*b1*b3*mu3-2*b3**2+8*b3*b2+6*b1*b3*mu1)*L1+8*b3*b2-4*b1**2)*L2+2*b1*b2-2*b3*b2+2*(3*b2*b3*mu3-2*b3*b2+3*b2*b3*mu1)*L1**2-8*b1*b2*L1,
                   -(-b1*b2*c3+3*b1*c1*b2*mu1+c1*b1*b2-3*b1*c3*b2*mu3)*L3**2+(-(-6*b1*c1*b2*mu1-6*b2*b3*c3*mu3-6*b1*b2*c3+c3*b2**2+2*c1*b3*b2+3*c3*b2**2*mu3+3*c1*b2**2*mu1-2*b2*b3*c3-3*c1*b2**2+6*b1*c3*b2*mu3+6*b2*b3*c1*mu1-2*c1*b1*b2)*L1-(-6*c1*b1*b2-2*b1*c3*b3-3*c3*b1**2+6*b1*c1*b3*mu1+2*b1*b2*c3-6*b1*c3*b3*mu3+6*b1*c3*b2*mu3-c1*b1**2+2*c1*b1*b3-3*c1*b1**2*mu1+6*b1*c1*b2*mu1+3*c3*b1**2*mu3)*L2+2*c1*b2**2)*L3-(-3*c1*b1*b3+3*b1*c1*b3*mu1+b1*c3*b3+3*b1*c3*b3*mu3)*L2**2+(-(6*b2*b3*c3*mu3-c3*b3**2-6*b1*c1*b3*mu1-6*c1*b3*b2-3*c3*b3**2*mu3+6*b2*b3*c1*mu1+3*c1*b3**2*mu1+6*b1*c3*b3*mu3-6*b1*c3*b3-2*c1*b1*b3+2*b2*b3*c3+c1*b3**2)*L1+4*c1*b3*b2+2*c3*b1**2)*L2-2*b1*b2*c3-(-3*b2*b3*c3-c1*b3*b2+3*b2*b3*c3*mu3-3*b2*b3*c1*mu1)*L1**2+4*b1*b2*c3*L1,
                   -(-b1**2*b2+3*b1*b2*b3*mu3+b1*b3*b2-3*b2*b1**2*mu1)*L3**2+(-(6*b2*b1**2*mu1-6*b1*b2*b3*mu3-b3*b2**2+2*b1**2*b2+4*b1*b3*b2+2*b2*b3**2-6*b2*b3*b1*mu1-3*b2**2*b3*mu3-3*b2**2*b1*mu1+6*b2*b3**2*mu3+3*b1*b2**2)*L1-(-6*b1*b2*b3*mu3+b1**3+2*b1*b3**2-6*b3*b1**2*mu1-2*b1*b3*b2+6*b1*b3**2*mu3+b1**2*b3-3*b1**2*b3*mu3+6*b1**2*b2+3*b1**3*mu1-6*b2*b1**2*mu1)*L2-2*b1*b2**2)*L3-(-3*b3*b1**2*mu1+3*b1**2*b3-3*b1*b3**2*mu3-b1*b3**2)*L2**2+(-(6*b3*b1**2*mu1-6*b1*b3**2*mu3+3*b3**3*mu3+6*b1*b3*b2-2*b2*b3**2+b3**3+2*b1**2*b3+5*b1*b3**2-6*b2*b3**2*mu3-3*b3**2*b1*mu1-6*b2*b3*b1*mu1)*L1-4*b1*b3*b2-2*b1**2*b3)*L2+2*b1*b3*b2-(3*b2*b3**2+3*b2*b3*b1*mu1+b1*b3*b2-3*b2*b3**2*mu3)*L1**2-4*b1*b3*b2*L1,
                   -2*(-4*b1*b2+3*b1*b2*mu2-3*b1*b2*mu1)*L3**2+(-2*(4*b1*b2+6*b2*b3*mu2-3*b2**2*mu1-6*b2*b3*mu1-8*b3*b2+2*b2**2+6*b1*b2*mu1-3*b2**2*mu2+6*b1*b2*mu2)*L1-2*(2*b1**2-8*b1*b3-6*b1*b2*mu1+4*b1*b2+3*b1**2*mu2+3*b1**2*mu1-6*b1*b2*mu2+6*b1*b3*mu2-6*b1*b3*mu1)*L2-4*b2**2+8*b1*b3)*L3-2*(-3*b1*b3*mu2+2*b1*b3-3*b1*b3*mu1)*L2**2+(-2*(-6*b2*b3*mu1-6*b2*b3*mu2+6*b1*b3*mu2+4*b3*b2+3*b3**2*mu2+4*b1*b3-3*b3**2*mu1+6*b1*b3*mu1-4*b3**2)*L1-8*b3*b2)*L2-2*b1*b3+2*b3*b2-2*(2*b3*b2+3*b2*b3*mu2+3*b2*b3*mu1)*L1**2+4*b3**2*L1,
                   (-3*b1*c1*b2*mu1-c1*b1*b2-3*b1*c2*b2*mu2+3*b1*c2*b2)*L3**2+((-6*b2*c2*b3*mu2-6*b2*b3*c1*mu1+6*b2*c2*b3+6*b1*c1*b2*mu1-3*c1*b2**2*mu1-2*b1*c2*b2-2*c1*b3*b2-6*b1*c2*b2*mu2+c2*b2**2+3*c2*b2**2*mu2+2*c1*b1*b2+3*c1*b2**2)*L1+(2*b1*c2*b2-6*b1*c1*b3*mu1-6*b1*c2*b3*mu2-2*c1*b1*b3+3*c1*b1**2*mu1+6*c1*b1*b2-c2*b1**2+6*b1*c2*b3+6*b1*c2*b2*mu2+c1*b1**2-3*c2*b1**2*mu2-6*b1*c1*b2*mu1)*L2+4*b1*c2*b3+2*c1*b2**2)*L3+(-3*b1*c1*b3*mu1+3*b1*c2*b3*mu2+3*c1*b1*b3+b1*c2*b3)*L2**2+((-6*b2*b3*c1*mu1-2*b1*c2*b3-6*b1*c2*b3*mu2-3*c2*b3**2*mu2-3*c1*b3**2*mu1+2*c1*b1*b3+2*b2*c2*b3-c1*b3**2+6*c1*b3*b2+6*b1*c1*b3*mu1+6*b2*c2*b3*mu2+3*b3**2*c2)*L1+4*c1*b3*b2)*L2-2*c1*b3*b2+(c1*b3*b2-3*b2*c2*b3*mu2-b2*c2*b3+3*b2*b3*c1*mu1)*L1**2+2*b3**2*c2*L1,
                   -(-b1**2*b2-3*b2*b1**2*mu1-3*b1*b2**2*mu2+3*b1*b2**2)*L3**2+(-(2*b1**2*b2+6*b2*b1**2*mu1+b1*b2**2-3*b2**2*b1*mu1+3*b2**3*mu2-6*b3*b2**2*mu2-2*b1*b3*b2-6*b2*b3*b1*mu1+6*b3*b2**2-6*b1*b2**2*mu2+b2**3)*L1-(-6*b3*b1**2*mu1+6*b1*b2**2*mu2-3*b1**2*b2*mu2-6*b2*b1**2*mu1+b1**3-2*b1**2*b3+5*b1**2*b2-6*b1*b3*b2*mu2+3*b1**3*mu1+2*b1*b2**2+6*b1*b3*b2)*L2-2*b1*b2**2-4*b1*b3*b2)*L3-(3*b1**2*b3+b1*b3*b2+3*b1*b3*b2*mu2-3*b3*b1**2*mu1)*L2**2+(-(-6*b1*b3*b2*mu2+2*b3*b2**2+2*b1**2*b3+3*b2*b3**2+6*b3*b2**2*mu2-b1*b3**2-3*b3**2*b1*mu1+6*b3*b1**2*mu1-3*b3**2*b2*mu2+4*b1*b3*b2-6*b2*b3*b1*mu1)*L1-4*b1*b3*b2)*L2+2*b1*b3*b2-(-b3*b2**2-3*b3*b2**2*mu2+3*b2*b3*b1*mu1+b1*b3*b2)*L1**2-2*b2*b3**2*L1],
                  [2*(3*c1*c2*mu3+3*c1*c2*mu2-2*c2*c1)*L3**2+(2*(-3*c2**2*mu2-2*c2**2+8*c2*c1+6*c1*c2*mu2+6*c2*c3*mu3+6*c2*c3*mu2-6*c1*c2*mu3-3*c2**2*mu3-4*c3*c2)*L1+2*(-4*c3*c1-3*c1**2*mu3-6*c1*c2*mu2+3*c1**2*mu2+4*c1**2-4*c2*c1+6*c1*c3*mu3-6*c1*c2*mu3+6*c1*c3*mu2)*L2-8*c3*c1)*L3+2*(-2*c3*c1-3*c1*c3*mu2-3*c1*c3*mu3)*L2**2+(2*(-6*c1*c3*mu3-6*c2*c3*mu2+3*c3**2*mu2+6*c1*c3*mu2-6*c2*c3*mu3+3*c3**2*mu3-4*c3*c2+8*c3*c1-2*c3**2)*L1+4*c1**2)*L2+2*c3*c1-2*c2*c1+2*(-3*c2*c3*mu3+4*c3*c2+3*c2*c3*mu2)*L1**2+2*(4*c2*c1-2*c3**2)*L1,
                   (3*c1*c2**2+c1*c3*c2-3*c1*c2**2*mu2+3*c1*c2*c3*mu3)*L3**2+((c2**3-6*c1*c2*c3*mu3-6*c1*c2**2*mu2-3*c2**2*c3*mu3+3*c2**3*mu2+6*c2*c3**2*mu3+5*c2**2*c3+6*c1*c3*c2-6*c2**2*c3*mu2-2*c1*c2**2+2*c2*c3**2)*L1+(-6*c1*c2*c3*mu3+6*c1*c3**2*mu3-3*c1**2*c3*mu3-3*c1**2*c2*mu2+6*c1*c2**2*mu2+4*c1*c3*c2-c1**2*c2+2*c1*c2**2+2*c1*c3**2-6*c1*c2*c3*mu2+3*c1**2*c3)*L2+4*c1*c3*c2)*L3+(-c1*c3**2+3*c1*c2*c3*mu2+c1*c3*c2-3*c1*c3**2*mu3)*L2**2+((-6*c2*c3**2*mu3+6*c2**2*c3*mu2+c3**3-6*c1*c3**2*mu3-3*c2*c3**2*mu2-2*c1*c3*c2+6*c1*c3**2+2*c2**2*c3+3*c3**3*mu3+c2*c3**2-6*c1*c2*c3*mu2)*L1+2*c1**2*c3)*L2-2*c1*c3*c2+(-c2**2*c3-3*c2*c3**2*mu3-3*c2**2*c3*mu2+3*c2*c3**2)*L1**2+(4*c1*c3*c2+2*c2*c3**2)*L1,
                   (3*c1*c2*b2*mu2-3*c1*c2*b3*mu3-c1*c2*b3-3*c1*c2*b2)*L3**2+((6*c1*c2*b2*mu2+3*c2**2*b3*mu3+6*c1*c2*b3*mu3-c2**2*b2+c2**2*b3+6*c3*b2*c2*mu2-6*c1*c2*b3+2*c1*c2*b2-6*c2*b3*c3*mu3-2*c3*b3*c2-6*c3*b2*c2-3*c2**2*b2*mu2)*L1+(-3*c1**2*b3-6*c1*b2*c3+2*c1*c2*b3+3*c1**2*b2*mu2-6*c1*c2*b2*mu2-6*c1*c3*b3*mu3-2*c1*c3*b3+6*c1*c3*b2*mu2-2*c1*c2*b2+c1**2*b2+6*c1*c2*b3*mu3+3*c1**2*b3*mu3)*L2-4*c1*b2*c3)*L3+(-3*c1*c3*b2*mu2+c1*c3*b3-c1*b2*c3+3*c1*c3*b3*mu3)*L2**2+((-6*c3*b2*c2*mu2+6*c1*c3*b3*mu3-c3**2*b3-2*c3*b2*c2+2*c1*b2*c3+2*c3*b3*c2+6*c1*c3*b2*mu2-3*c3**2*b3*mu3-6*c1*c3*b3+6*c2*b3*c3*mu3+3*c3**2*b2*mu2-3*c3**2*b2)*L1-2*c1**2*b3)*L2+2*c1*b2*c3+(-3*c3*b3*c2+3*c3*b2*c2*mu2+3*c2*b3*c3*mu3+c3*b2*c2)*L1**2+(-4*c1*c2*b3-2*c3**2*b2)*L1,
                   -2*(3*c1*c2*mu3+2*c2*c1+3*c1*c2*mu1)*L3**2+(-2*(6*c2*c3*mu3-4*c2**2-6*c1*c2*mu3-3*c2**2*mu3+4*c3*c2+3*c2**2*mu1-6*c1*c2*mu1+6*c2*c3*mu1+4*c2*c1)*L1-2*(6*c1*c2*mu1-6*c1*c2*mu3+6*c1*c3*mu3+6*c1*c3*mu1-3*c1**2*mu1+4*c3*c1-8*c2*c1-3*c1**2*mu3+2*c1**2)*L2+4*c2**2)*L3-2*(3*c1*c3*mu1-4*c3*c1-3*c1*c3*mu3)*L2**2+(-2*(3*c3**2*mu1+3*c3**2*mu3-6*c1*c3*mu1+6*c2*c3*mu1-6*c1*c3*mu3+4*c3*c1-8*c3*c2-6*c2*c3*mu3+2*c3**2)*L1-4*c1**2+8*c3*c2)*L2+2*c2*c1-2*c3*c2-2*(-3*c2*c3*mu1-3*c2*c3*mu3+2*c3*c2)*L1**2-8*c2*c1*L1,
                   (c1*c3*c2-c1**2*c2-3*c1**2*c2*mu1+3*c1*c2*c3*mu3)*L3**2+((6*c1**2*c2*mu1-c2**2*c3-6*c2*c3*c1*mu1-3*c2**2*c3*mu3-3*c1*c2**2*mu1+6*c2*c3**2*mu3+4*c1*c3*c2+2*c2*c3**2-6*c1*c2*c3*mu3+3*c1*c2**2+2*c1**2*c2)*L1+(-2*c1*c3*c2+6*c1**2*c2+3*c1**3*mu1-6*c1*c2*c3*mu3+2*c1*c3**2+c1**2*c3+6*c1*c3**2*mu3+c1**3-6*c1**2*c3*mu1-6*c1**2*c2*mu1-3*c1**2*c3*mu3)*L2+2*c1*c2**2)*L3+(3*c1**2*c3-3*c1*c3**2*mu3-3*c1**2*c3*mu1-c1*c3**2)*L2**2+((5*c1*c3**2+3*c3**3*mu3-6*c2*c3**2*mu3-6*c1*c3**2*mu3+6*c1**2*c3*mu1+c3**3-6*c2*c3*c1*mu1+6*c1*c3*c2-2*c2*c3**2-3*c1*c3**2*mu1+2*c1**2*c3)*L1+4*c1*c3*c2+2*c1**2*c3)*L2-2*c1*c3*c2+(3*c2*c3**2+c1*c3*c2+3*c2*c3*c1*mu1-3*c2*c3**2*mu3)*L1**2+4*c1*c3*c2*L1,
                   (3*c2*b1*c1*mu1-c1*c2*b3+c1*c2*b1-3*c1*c2*b3*mu3)*L3**2+((-2*c1*c2*b1+3*c2**2*b3*mu3+6*c1*c2*b3*mu3-6*c2*b1*c1*mu1+3*c2**2*b1*mu1+2*c3*c2*b1+6*c2*c3*b1*mu1-3*c2**2*b1+c2**2*b3-6*c2*b3*c3*mu3-6*c1*c2*b3-2*c3*b3*c2)*L1+(-c1**2*b1+2*c1*c3*b1-3*c1**2*b1*mu1+6*c2*b1*c1*mu1-6*c1*c2*b1-3*c1**2*b3-6*c1*c3*b3*mu3+6*c1*c2*b3*mu3+2*c1*c2*b3-2*c1*c3*b3+6*c3*b1*c1*mu1+3*c1**2*b3*mu3)*L2-2*c2**2*b1)*L3+(3*c3*b1*c1*mu1+c1*c3*b3+3*c1*c3*b3*mu3-3*c1*c3*b1)*L2**2+((3*c3**2*b1*mu1+c3**2*b1-6*c3*b1*c1*mu1+6*c1*c3*b3*mu3-2*c1*c3*b1+6*c2*b3*c3*mu3-6*c3*c2*b1+2*c3*b3*c2-c3**2*b3-3*c3**2*b3*mu3-6*c1*c3*b3+6*c2*c3*b1*mu1)*L1-2*c1**2*b3-4*c3*c2*b1)*L2+2*c1*c2*b3+(3*c2*b3*c3*mu3-c3*c2*b1-3*c3*b3*c2-3*c2*c3*b1*mu1)*L1**2-4*c1*c2*b3*L1,
                   2*(-3*c1*c2*mu2+4*c2*c1+3*c1*c2*mu1)*L3**2+(2*(6*c2*c3*mu1-2*c2**2+3*c2**2*mu2-6*c2*c3*mu2+3*c2**2*mu1-6*c1*c2*mu1-4*c2*c1-6*c1*c2*mu2+8*c3*c2)*L1+2*(-2*c1**2+8*c3*c1-6*c1*c3*mu2-3*c1**2*mu1-4*c2*c1-3*c1**2*mu2+6*c1*c2*mu1+6*c1*c2*mu2+6*c1*c3*mu1)*L2-4*c2**2+8*c3*c1)*L3+2*(-2*c3*c1+3*c1*c3*mu2+3*c1*c3*mu1)*L2**2+(2*(-3*c3**2*mu2+3*c3**2*mu1-6*c1*c3*mu1+6*c2*c3*mu1-6*c1*c3*mu2-4*c3*c2+4*c3**2-4*c3*c1+6*c2*c3*mu2)*L1-8*c3*c2)*L2-2*c3*c1+2*c3*c2+2*(-3*c2*c3*mu2-2*c3*c2-3*c2*c3*mu1)*L1**2+4*c3**2*L1,
                   -(3*c1*c2**2*mu2-3*c1*c2**2+3*c1**2*c2*mu1+c1**2*c2)*L3**2+(-(-2*c1**2*c2-6*c2**2*c3+6*c2**2*c3*mu2-c1*c2**2+2*c1*c3*c2-6*c1**2*c2*mu1-3*c2**3*mu2+3*c1*c2**2*mu1+6*c2*c3*c1*mu1+6*c1*c2**2*mu2-c2**3)*L1-(-5*c1**2*c2-3*c1**3*mu1+2*c1**2*c3-c1**3+6*c1*c2*c3*mu2-2*c1*c2**2+6*c1**2*c3*mu1-6*c1*c3*c2+6*c1**2*c2*mu1-6*c1*c2**2*mu2+3*c1**2*c2*mu2)*L2+2*c1*c2**2+4*c1*c3*c2)*L3-(-3*c1**2*c3-c1*c3*c2-3*c1*c2*c3*mu2+3*c1**2*c3*mu1)*L2**2+(-(6*c2*c3*c1*mu1-6*c1**2*c3*mu1+c1*c3**2-2*c1**2*c3-3*c2*c3**2-2*c2**2*c3+6*c1*c2*c3*mu2+3*c2*c3**2*mu2+3*c1*c3**2*mu1-4*c1*c3*c2-6*c2**2*c3*mu2)*L1+4*c1*c3*c2)*L2-2*c1*c3*c2-(-3*c2*c3*c1*mu1+3*c2**2*c3*mu2+c2**2*c3-c1*c3*c2)*L1**2+2*c2*c3**2*L1,
                   (3*c1*c2*b2*mu2-3*c1*c2*b2+c1*c2*b1+3*c2*b1*c1*mu1)*L3**2+((-2*c1*c2*b1+6*c3*b2*c2*mu2-3*c2**2*b2*mu2-6*c3*b2*c2-6*c2*b1*c1*mu1-c2**2*b2+2*c1*c2*b2+3*c2**2*b1*mu1+2*c3*c2*b1+6*c2*c3*b1*mu1-3*c2**2*b1+6*c1*c2*b2*mu2)*L1+(6*c3*b1*c1*mu1-c1**2*b1+2*c1*c3*b1-6*c1*c2*b1-3*c1**2*b1*mu1-6*c1*c2*b2*mu2-6*c1*b2*c3-2*c1*c2*b2+3*c1**2*b2*mu2+6*c1*c3*b2*mu2+6*c2*b1*c1*mu1+c1**2*b2)*L2-2*c2**2*b1-4*c1*b2*c3)*L3+(-3*c1*c3*b2*mu2+3*c3*b1*c1*mu1-c1*b2*c3-3*c1*c3*b1)*L2**2+((3*c3**2*b2*mu2+6*c1*c3*b2*mu2+6*c2*c3*b1*mu1-2*c1*c3*b1+c3**2*b1-2*c3*b2*c2+2*c1*b2*c3-3*c3**2*b2-6*c3*b1*c1*mu1+3*c3**2*b1*mu1-6*c3*c2*b1-6*c3*b2*c2*mu2)*L1-4*c3*c2*b1)*L2+2*c3*c2*b1+(c3*b2*c2+3*c3*b2*c2*mu2-c3*c2*b1-3*c2*c3*b1*mu1)*L1**2-2*c3**2*b2*L1],
                  [2.*(-(2*b2*c1-3*c2*b1*mu3+2*c2*b1-3*c1*b2*mu2-3*c2*b1*mu2-3*c1*b2*mu3)*L3**2+(-(-6*c1*b2*mu2+4*c2*b2-8*b2*c1-6*c2*b1*mu2+6*c2*b2*mu3+6*c2*b2*mu2+6*c2*b1*mu3+6*c1*b2*mu3-8*c2*b1-6*c3*b2*mu2+4*c2*b3+4*b2*c3-6*c2*b3*mu2-6*c2*b3*mu3-6*c3*b2*mu3)*L1-(6*c1*b1*mu3+6*c2*b1*mu3+4*b3*c1+6*c2*b1*mu2+6*c1*b2*mu3-6*c1*b3*mu2-6*c3*b1*mu2-6*c1*b1*mu2+4*c3*b1-8*c1*b1+6*c1*b2*mu2-6*c1*b3*mu3+4*b2*c1+4*c2*b1-6*c3*b1*mu3)*L2-4*c3*b1-4*b3*c1)*L3-(3*c3*b1*mu3+2*b3*c1+3*c1*b3*mu3+3*c3*b1*mu2+2*c3*b1+3*c1*b3*mu2)*L2**2+(-(-8*c3*b1-8*b3*c1+6*c1*b3*mu3-6*c1*b3*mu2-6*c3*b3*mu3+6*c2*b3*mu3+4*c3*b3+4*c2*b3+6*c3*b1*mu3+4*b2*c3+6*c3*b2*mu3-6*c3*b1*mu2+6*c3*b2*mu2+6*c2*b3*mu2-6*c3*b3*mu2)*L1+4*c1*b1)*L2-b2*c1+c3*b1-c2*b1+b3*c1-(-4*b2*c3-3*c3*b2*mu2+3*c2*b3*mu3-4*c2*b3+3*c3*b2*mu3-3*c2*b3*mu2)*L1**2-(4*c3*b3-4*b2*c1-4*c2*b1)*L1),
                   2.*(-0.5*(-3*c2**2*b1+3*c2**2*b1*mu2-3*c2*b1*c3*mu3-3*c1*c3*b2*mu3-c1*b2*c3-c3*c2*b1-3*c1*c2*b2+3*c1*c2*b2*mu2)*L3**2+(-0.5*(6*c2*c3*b2*mu3-2*c3*b3*c2-6*c3*c2*b1-2*c2**2*b2+2*c2**2*b1-6*c2**2*b2*mu2+6*c1*c3*b2*mu3+2*c1*c2*b2-2*c3**2*b2-6*c2**2*b3+6*c2*b1*c3*mu3+6*c2**2*b1*mu2-4*c3*b2*c2-6*c3**2*b2*mu3+6*c1*c2*b2*mu2-6*c1*b2*c3+6*c2**2*b3*mu2-6*c2*b3*c3*mu3+6*c3*b2*c2*mu2)*L1-0.5*(6*c1*c2*b1*mu2-4*c3*c2*b1-2*c1*c3*b3+6*c3*b1*c2*mu2-6*c1*c3*b1-2*c2**2*b1+6*c2*b1*c3*mu3+6*c1*c3*b1*mu3-2*c3**2*b1-6*c1*c2*b3-6*c1*c3*b3*mu3+6*c1*c3*b2*mu3-2*c1*c2*b2+6*c1*c2*b3*mu2-6*c2**2*b1*mu2-6*c1*c2*b2*mu2+2*c1*c2*b1-6*c3**2*b1*mu3+2*c1*b2*c3)*L2+2*c1*c2*b3+2*c3*c2*b1)*L3-0.5*(3*c1*c3*b3*mu3+c3**2*b1-3*c3*b1*c2*mu2-c3*c2*b1-3*c1*c2*b3*mu2+c1*c3*b3-c1*c2*b3+3*c3**2*b1*mu3)*L2**2+(-0.5*(-6*c3**2*b1-6*c2**2*b3*mu2+6*c1*c3*b3*mu3-6*c1*c3*b3+2*c1*c2*b3-2*c3**2*b3-6*c3**2*b3*mu3-4*c3*b3*c2-6*c3*b2*c2*mu2+6*c3*b1*c2*mu2+6*c3**2*b1*mu3+6*c3**2*b2*mu3-2*c2**2*b3+6*c1*c2*b3*mu2+2*c3**2*b2+2*c3*c2*b1-2*c3*b2*c2+6*c2*b3*c3*mu3+6*c3*c2*b3*mu2)*L1+2*c1*c3*b1)*L2-c3*c2*b1-c1*c2*b3-0.5*(3*c3**2*b2*mu3+3*c3*b2*c2*mu2+c3*b2*c2+3*c2*b3*c3*mu3+3*c2**2*b3*mu2-3*c3*b3*c2-3*c3**2*b2+c2**2*b3)*L1**2-0.5*(-4*c3*b3*c2-4*c3*c2*b1-4*c1*b2*c3)*L1),
                   2.*(0.5*(-3*b1*c2*b2-3*c1*b2**2-c1*b2*b3+3*c1*b2**2*mu2+3*b1*c2*b2*mu2-b1*c2*b3-3*c1*b2*b3*mu3-3*c2*b1*b3*mu3)*L3**2+(0.5*(2*c1*b2**2+2*b1*c2*b2-4*b2*c2*b3-6*c1*b2*b3-2*c2*b2**2+6*c1*b2**2*mu2-6*c3*b2**2-6*c2*b2**2*mu2+6*b2*c2*b3*mu2+6*c3*b2**2*mu2-6*b2*b3*c3*mu3-2*b2*b3*c3-6*c2*b3**2*mu3+6*c2*b2*b3*mu3+6*b1*c2*b2*mu2-2*b3**2*c2+6*c2*b1*b3*mu3+6*c1*b2*b3*mu3-6*b1*c2*b3)*L1+0.5*(-4*c1*b2*b3-2*c1*b2**2-6*c1*b1*b3+6*c1*b1*b2*mu2+6*c3*b1*b2*mu2+2*c1*b1*b2+6*c1*b2*b3*mu2-2*c1*b3**2-6*b1*b2*c3-6*b1*c3*b3*mu3-6*c1*b3**2*mu3-2*b1*c3*b3+6*c2*b1*b3*mu3-6*c1*b2**2*mu2+2*b1*c2*b3+6*c1*b2*b3*mu3-6*b1*c2*b2*mu2-2*b1*c2*b2+6*c1*b1*b3*mu3)*L2-2*b1*b2*c3-2*c1*b2*b3)*L3+0.5*(-3*c3*b1*b2*mu2+b1*c3*b3+c1*b3**2-b1*b2*c3+3*b1*c3*b3*mu3-3*c1*b2*b3*mu2+3*c1*b3**2*mu3-c1*b2*b3)*L2**2+(0.5*(-2*c3*b3**2-6*b1*c3*b3-6*b2*c2*b3*mu2+6*b1*c3*b3*mu3-6*c3*b3**2*mu3-4*b2*b3*c3+6*c3*b2*b3*mu2+6*c1*b2*b3*mu2+6*c1*b3**2*mu3-6*c3*b2**2*mu2+2*b1*b2*c3-6*c1*b3**2+6*c3*b1*b2*mu2+6*c2*b3**2*mu3+6*b2*b3*c3*mu3-2*c3*b2**2+2*b3**2*c2+2*c1*b2*b3-2*b2*c2*b3)*L1-2*c1*b1*b3)*L2+b1*b2*c3+c1*b2*b3+0.5*(3*b2*c2*b3*mu2+3*b2*b3*c3*mu3+3*c3*b2**2*mu2-3*b2*b3*c3-3*b3**2*c2+c3*b2**2+3*c2*b3**2*mu3+b2*c2*b3)*L1**2+0.5*(-4*b1*c2*b3-4*c1*b2*b3-4*b2*b3*c3)*L1),
                   2.*((-2*c2*b1-3*c1*b2*mu3-3*c2*b1*mu3-3*c2*b1*mu1-3*c1*b2*mu1-2*b2*c1)*L3**2+((6*c2*b2*mu3-6*c3*b2*mu1-4*c2*b3-4*c2*b1-6*c2*b2*mu1-4*b2*c1+6*c1*b2*mu3+6*c2*b1*mu3-6*c2*b3*mu1+8*c2*b2+6*c1*b2*mu1-6*c2*b3*mu3-4*b2*c3-6*c3*b2*mu3+6*c2*b1*mu1)*L1+(-4*c3*b1+6*c2*b1*mu3+6*c1*b2*mu3-4*b3*c1-6*c3*b1*mu3-6*c2*b1*mu1-6*c3*b1*mu1-6*c1*b3*mu3-6*c1*b2*mu1-6*c1*b3*mu1+6*c1*b1*mu3+8*b2*c1+8*c2*b1+6*c1*b1*mu1-4*c1*b1)*L2+4*c2*b2)*L3+(-3*c1*b3*mu1-3*c3*b1*mu1+4*b3*c1+3*c1*b3*mu3+3*c3*b1*mu3+4*c3*b1)*L2**2+((-6*c3*b3*mu3-4*b3*c1+8*b2*c3-4*c3*b1+6*c3*b2*mu3+6*c3*b1*mu3-6*c2*b3*mu1+6*c1*b3*mu3-4*c3*b3+6*c3*b1*mu1-6*c3*b2*mu1+6*c2*b3*mu3-6*c3*b3*mu1+8*c2*b3+6*c1*b3*mu1)*L1-4*c1*b1+4*b2*c3+4*c2*b3)*L2-c2*b3+c2*b1-b2*c3+b2*c1+(3*c2*b3*mu1+3*c3*b2*mu3-2*b2*c3+3*c2*b3*mu3-2*c2*b3+3*c3*b2*mu1)*L1**2+(-4*b2*c1-4*c2*b1)*L1),
                   2.*(0.5*(-c1**2*b2+c3*c2*b1+3*c2*b1*c3*mu3-c1*c2*b1+c1*b2*c3+3*c1*c3*b2*mu3-3*b2*c1**2*mu1-3*c2*b1*c1*mu1)*L3**2+(0.5*(-6*c1*c3*b2*mu3+6*b2*c1**2*mu1+2*c1*c2*b1+6*c1*c2*b2+2*c3*b3*c2-2*c3*b2*c2-2*c1*c2*b3+4*c1*b2*c3+6*c3*c2*b1-6*c2*b2*c1*mu1+2*c1**2*b2-6*c3*b2*c1*mu1+2*c3**2*b2-6*c2*b1*c3*mu3-6*c2*b3*c1*mu1+6*c2*b3*c3*mu3-6*c2*c3*b2*mu3+6*c3**2*b2*mu3+6*c2*b1*c1*mu1)*L1+0.5*(-2*c3*c2*b1+6*c1*c2*b1+6*c1*c3*b3*mu3+2*c1*c3*b3+4*c1*c3*b1-2*c1*b2*c3+2*c3**2*b1-6*c3*b1*c1*mu1-6*b2*c1**2*mu1-6*c2*b1*c1*mu1-6*b3*c1**2*mu1-6*c1*c3*b2*mu3+6*c3**2*b1*mu3+6*c1**2*b2-6*c1*c3*b1*mu3+2*b1*c1**2+6*b1*c1**2*mu1-6*c2*b1*c3*mu3-2*c1**2*b3)*L2+2*c1*c2*b2)*L3+0.5*(3*c1**2*b3-c1*c3*b3+3*c1*c3*b1-3*c3*b1*c1*mu1-3*c3**2*b1*mu3-3*c1*c3*b3*mu3-3*b3*c1**2*mu1-c3**2*b1)*L2**2+(0.5*(6*c1*c2*b3+6*c1*b2*c3-6*c3*b3*c1*mu1+6*c3**2*b1-2*c3*b3*c2+6*c3*b1*c1*mu1+2*c1*c3*b1-6*c3**2*b2*mu3+2*c1**2*b3-6*c1*c3*b3*mu3-6*c3*b2*c1*mu1-2*c3**2*b2+2*c3**2*b3+6*b3*c1**2*mu1+6*c3**2*b3*mu3+4*c1*c3*b3-6*c2*b3*c1*mu1-6*c2*b3*c3*mu3-6*c3**2*b1*mu3)*L1+2*c1*b2*c3+2*c1*c3*b1+2*c1*c2*b3)*L2-c1*b2*c3-c3*c2*b1+0.5*(c1*c2*b3+c1*b2*c3+3*c3*b2*c1*mu1+3*c2*b3*c1*mu1-3*c3**2*b2*mu3+3*c3**2*b2+3*c3*b3*c2-3*c2*b3*c3*mu3)*L1**2+0.5*(4*c1*b2*c3+4*c3*c2*b1)*L1),
                   2.*(-0.5*(c1*b2*b3-c2*b1**2+b1*c2*b3+3*c1*b2*b3*mu3-3*b1*c1*b2*mu1-3*c2*b1**2*mu1+3*c2*b1*b3*mu3-c1*b1*b2)*L3**2+(-0.5*(2*c1*b1*b2-6*c1*b2*b3*mu3+6*c1*b2*b3-2*b2*c2*b3+2*b2*b3*c3+4*b1*c2*b3+6*b1*c2*b2-6*c2*b1*b3*mu3-6*c2*b2*b3*mu3+6*c2*b3**2*mu3-2*b1*b2*c3+6*b1*c1*b2*mu1+2*b3**2*c2-6*c3*b2*b1*mu1-6*c2*b3*b1*mu1+2*c2*b1**2+6*b2*b3*c3*mu3-6*c2*b2*b1*mu1+6*c2*b1**2*mu1)*L1-0.5*(-2*b1*c2*b3+6*c2*b1**2-6*c1*b2*b3*mu3+6*c1*b1**2*mu1+6*b1*c3*b3*mu3-6*c2*b1**2*mu1-6*c2*b1*b3*mu3+2*b1*c3*b3+6*c1*b1*b2-2*c1*b2*b3-6*c3*b1**2*mu1+6*c1*b3**2*mu3+2*c1*b1**2+4*c1*b1*b3-6*c1*b1*b3*mu3+2*c1*b3**2-6*b1*c1*b2*mu1-6*b1*c1*b3*mu1-2*c3*b1**2)*L2-2*b1*c2*b2)*L3-0.5*(-3*c3*b1**2*mu1-3*c1*b3**2*mu3-c1*b3**2+3*c1*b1*b3-3*b1*c3*b3*mu3-3*b1*c1*b3*mu1+3*c3*b1**2-b1*c3*b3)*L2**2+(-0.5*(2*c3*b3**2+6*c3*b3**2*mu3-6*b2*b3*c3*mu3+6*b1*c2*b3+6*c3*b1**2*mu1-2*b3**2*c2+2*c1*b1*b3-6*c2*b3*b1*mu1-6*c3*b3*b1*mu1-6*c1*b3**2*mu3+6*b1*b2*c3-6*c3*b2*b1*mu1+4*b1*c3*b3+6*c1*b3**2-6*b1*c3*b3*mu3+2*c3*b1**2-2*b2*b3*c3+6*b1*c1*b3*mu1-6*c2*b3**2*mu3)*L1-2*c1*b1*b3-2*b1*b2*c3-2*b1*c2*b3)*L2+c1*b2*b3+b1*c2*b3-0.5*(b1*b2*c3-3*b2*b3*c3*mu3+3*b2*b3*c3+b1*c2*b3+3*b3**2*c2-3*c2*b3**2*mu3+3*c2*b3*b1*mu1+3*c3*b2*b1*mu1)*L1**2-0.5*(4*c1*b2*b3+4*b1*c2*b3)*L1),
                   2.*(-(-3*c2*b1*mu1+3*c2*b1*mu2-4*c2*b1-4*b2*c1-3*c1*b2*mu1+3*c1*b2*mu2)*L3**2+(-(-6*c3*b2*mu1+4*b2*c1-8*b2*c3+4*c2*b1+6*c3*b2*mu2+6*c1*b2*mu1-6*c2*b2*mu1+6*c2*b3*mu2+6*c2*b1*mu1+6*c2*b1*mu2-6*c2*b3*mu1+6*c1*b2*mu2-8*c2*b3-6*c2*b2*mu2+4*c2*b2)*L1-(-6*c3*b1*mu1+6*c1*b1*mu2-6*c1*b2*mu1-6*c2*b1*mu2+4*c1*b1+6*c1*b3*mu2+6*c1*b1*mu1-6*c1*b3*mu1-8*c3*b1-8*b3*c1+4*c2*b1-6*c2*b1*mu1+4*b2*c1+6*c3*b1*mu2-6*c1*b2*mu2)*L2-4*c2*b2+4*c3*b1+4*b3*c1)*L3-(-3*c3*b1*mu2+2*b3*c1-3*c1*b3*mu1-3*c3*b1*mu1+2*c3*b1-3*c1*b3*mu2)*L2**2+(-(4*c3*b1+4*b3*c1+4*c2*b3+6*c3*b3*mu2-8*c3*b3+6*c3*b1*mu1-6*c2*b3*mu2-6*c2*b3*mu1+6*c1*b3*mu1-6*c3*b3*mu1+6*c1*b3*mu2-6*c3*b2*mu1+4*b2*c3+6*c3*b1*mu2-6*c3*b2*mu2)*L1-4*c2*b3-4*b2*c3)*L2+c2*b3-b3*c1-c3*b1+b2*c3-(2*c2*b3+3*c3*b2*mu1+3*c3*b2*mu2+2*b2*c3+3*c2*b3*mu2+3*c2*b3*mu1)*L1**2+4*c3*b3*L1),
                   2.*(0.5*(-c1**2*b2-3*c2*b1*c1*mu1-3*c1*c2*b2*mu2+3*c1*c2*b2-c1*c2*b1-3*c2**2*b1*mu2+3*c2**2*b1-3*b2*c1**2*mu1)*L3**2+(0.5*(2*c1*c2*b1+4*c1*c2*b2-6*c3*b2*c2*mu2+6*b2*c1**2*mu1-2*c2**2*b1+2*c1**2*b2+6*c2**2*b3+6*c3*b2*c2-6*c2*b3*c1*mu1-2*c1*b2*c3-6*c2**2*b1*mu2-6*c2*b2*c1*mu1+2*c2**2*b2+6*c2*b1*c1*mu1-6*c3*b2*c1*mu1-2*c1*c2*b3-6*c2**2*b3*mu2+6*c2**2*b2*mu2-6*c1*c2*b2*mu2)*L1+0.5*(4*c1*c2*b1+6*c1*c2*b3+6*c2**2*b1*mu2-6*c3*b1*c2*mu2+6*c1**2*b2-6*c3*b1*c1*mu1+6*c3*c2*b1-2*c1*c3*b1-2*c1**2*b3-6*b3*c1**2*mu1+6*b1*c1**2*mu1-6*b2*c1**2*mu1-6*c1*c2*b3*mu2+6*c1*c2*b2*mu2-6*c2*b1*c1*mu1+2*c1*c2*b2-6*c1*c2*b1*mu2+2*c2**2*b1+2*b1*c1**2)*L2+2*c1*c2*b3+2*c1*c2*b2+2*c3*c2*b1)*L3+0.5*(-3*c3*b1*c1*mu1+3*c1*c2*b3*mu2+3*c1**2*b3+c3*c2*b1+3*c1*c3*b1+c1*c2*b3-3*b3*c1**2*mu1+3*c3*b1*c2*mu2)*L2**2+(0.5*(-2*c3*c2*b1+4*c1*c2*b3-2*c1*c3*b3+2*c1**2*b3+6*c3*b3*c2+6*c1*b2*c3-6*c2*b3*c1*mu1+2*c1*c3*b1+6*c3*b1*c1*mu1-6*c3*b3*c1*mu1-6*c3*b2*c1*mu1-6*c3*b1*c2*mu2+6*c3*b2*c2*mu2+2*c3*b2*c2+2*c2**2*b3+6*b3*c1**2*mu1+6*c2**2*b3*mu2-6*c1*c2*b3*mu2-6*c3*c2*b3*mu2)*L1+2*c1*b2*c3+2*c1*c2*b3)*L2-c1*b2*c3-c1*c2*b3+0.5*(c1*c2*b3+3*c2*b3*c1*mu1-c3*b2*c2+3*c3*b2*c1*mu1-3*c2**2*b3*mu2-c2**2*b3-3*c3*b2*c2*mu2+c1*b2*c3)*L1**2+2*c3*b3*c2*L1),
                   2.*(-0.5*(-3*b1*c2*b2*mu2-c2*b1**2+3*c1*b2**2-3*c1*b2**2*mu2-c1*b1*b2-3*c2*b1**2*mu1-3*b1*c1*b2*mu1+3*b1*c2*b2)*L3**2+(-0.5*(-6*c1*b2**2*mu2+2*c2*b1**2+2*c2*b2**2-2*c1*b2**2-2*b1*b2*c3+6*c3*b2**2+6*c2*b2**2*mu2+4*b1*c2*b2+6*c2*b1**2*mu1-6*c2*b3*b1*mu1-6*c2*b2*b1*mu1+6*b1*c1*b2*mu1-6*c3*b2**2*mu2+2*c1*b1*b2-6*b1*c2*b2*mu2+6*b2*c2*b3-6*b2*c2*b3*mu2-6*c3*b2*b1*mu1-2*b1*c2*b3)*L1-0.5*(2*c1*b2**2+6*c2*b1**2+2*b1*c2*b2-6*c1*b2*b3*mu2-6*c3*b1**2*mu1+4*c1*b1*b2+6*c1*b1**2*mu1-6*c2*b1**2*mu1-6*c3*b1*b2*mu2+6*b1*b2*c3-6*b1*c1*b2*mu1+6*c1*b2*b3+2*c1*b1**2-6*b1*c1*b3*mu1-2*c1*b1*b3-2*c3*b1**2-6*c1*b1*b2*mu2+6*b1*c2*b2*mu2+6*c1*b2**2*mu2)*L2-2*b1*c2*b2-2*c1*b2*b3-2*b1*b2*c3)*L3-0.5*(3*c1*b1*b3+3*c3*b1*b2*mu2-3*b1*c1*b3*mu1+3*c1*b2*b3*mu2-3*c3*b1**2*mu1+b1*b2*c3+3*c3*b1**2+c1*b2*b3)*L2**2+(-0.5*(2*c3*b1**2+6*b1*c1*b3*mu1+6*c3*b1**2*mu1-6*c1*b2*b3*mu2-2*c1*b2*b3-6*c2*b3*b1*mu1+2*c3*b2**2+6*b1*c2*b3+6*b2*c2*b3*mu2-2*b1*c3*b3+2*c1*b1*b3+6*b2*b3*c3+2*b2*c2*b3-6*c3*b1*b2*mu2-6*c3*b2*b3*mu2-6*c3*b3*b1*mu1+4*b1*b2*c3-6*c3*b2*b1*mu1+6*c3*b2**2*mu2)*L1-2*b1*b2*c3-2*b1*c2*b3)*L2+b1*b2*c3+b1*c2*b3-0.5*(b1*c2*b3-c3*b2**2+3*c2*b3*b1*mu1-3*c3*b2**2*mu2-3*b2*c2*b3*mu2+b1*b2*c3+3*c3*b2*b1*mu1-b2*c2*b3)*L1**2-2*b2*b3*c3*L1)]])
        
        return B, 1, 0 
    def FormT(self, L1, L2, L3):
        L3=1-L1-L2
        T = array([L1, 0, 0, L2, 0, 0, L3, 0, 0])
        return T
    def FormX(self, L1, L2, L3):
        L3=1-L1-L2
        X = array([L1, L2, L3])
        return X
    def JacoD(self, r, s, t):
        return 1
    
class SH4(Element):
    """ Shell element 4 nodes
    """
    def LengV(self, Vec):                               # bring Vec to length 1
        LL = sqrt(Vec[0]**2+Vec[1]**2+Vec[2]**2)
        LL = 1./LL                                      # division by zero not catched !
        Vec[0]= LL*Vec[0]
        Vec[1]= LL*Vec[1]
        Vec[2]= LL*Vec[2]
    def CompNoNor(self, i0, i1, i2):                    # returns normal to i0 -> i1 and i0- -> i2
        ax = self.XX[0,i1] - self.XX[0,i0]
        ay = self.XX[1,i1] - self.XX[1,i0]
        az = self.XX[2,i1] - self.XX[2,i0]
        bx = self.XX[0,i2] - self.XX[0,i0]
        by = self.XX[1,i2] - self.XX[1,i0]
        bz = self.XX[2,i2] - self.XX[2,i0]
        nx = ay*bz - az*by
        ny = az*bx - ax*bz
        nz = ax*by - ay*bx
        ll = sqrt(nx*nx+ny*ny+nz*nz)
        if ll<ZeroD: raise NameError("ConFemElements::SHX.__ini__::VecPro: something wrong with geometry")
        return nx/ll, ny/ll, nz/ll
    def ComputeGG(self):
        for i in xrange(3): 
            for j in xrange(4): self.gg[0,i,j] = -self.a[j]*self.V2[i,j]/2.     # 1st index direction, 2nd index node 
            for j in xrange(4): self.gg[1,i,j] =  self.a[j]*self.V1[i,j]/2.
    def ComputeEdgeDir(self):                           # direction of 1st edge for later use
        self.EdgeDir[0] = self.XX[0,1]-self.XX[0,0]
        self.EdgeDir[1] = self.XX[1,1]-self.XX[1,0]
        self.EdgeDir[2] = self.XX[2,1]-self.XX[2,0]
        self.LengV(self.EdgeDir)
    def CompleteTriad(self, dd):                        # complete director system with V1, V2
        Tol = 0.01
        V1 = zeros((3),dtype=float)
        V2 = zeros((3),dtype=float)
        if abs(dd[2])>Tol or abs(dd[0])>Tol: V1[0]= dd[2];              V1[2]=-dd[0] # seems to regard on direction / unit vector of global coordinate system
        else:                                V1[0]=-dd[1]; V1[1]= dd[0]
        self.LengV(V1)
        V2[0] = dd[1]*V1[2]
        V2[1] = dd[2]*V1[0]-dd[0]*V1[2]
        V2[2] =            -dd[1]*V1[0]
        return V1, V2
    def ComputeTransLocal(self, i, nn, TTW):            # 
        V1_, V2_ = self.CompleteTriad( nn)              # second approach for director triad with input data for nodes
        TT = zeros((2,3),dtype=float)                   # coordinate transformation matrix for rotations regarding current node
        TT[0,0] = dot(V1_,self.V1[:,i])
        TT[0,1] = dot(V1_,self.V2[:,i])
        TT[0,2] = dot(V1_,self.Vg[:,i])
        TT[1,0] = dot(V2_,self.V1[:,i])
        TT[1,1] = dot(V2_,self.V2[:,i])
        TT[1,2] = dot(V2_,self.Vg[:,i])
        TTW[i]  = copy(TT)
        for j in xrange(3): self.V1[j,i] = V1_[j]       # set rest of the director triad annew
        for j in xrange(3): self.V2[j,i] = V2_[j]       #
    def ComputeTransLocalAll(self, TTW):
        base, base_ = 0, 0
        for i in xrange(4):                             # loop over nodes of element  
            for j in xrange(3):          self.Trans[base_+j,base+j] = 1.
            if self.SixthDoF[i]:
                for j in xrange(2):
                    for jj in xrange(3): self.Trans[base_+3+j,base+3+jj] = TTW[i][j][jj] 
#   uhc             XX = dot(transpose(TTW[i]),TTW[i])
#   uhc             YY = dot(TTW[i],transpose(TTW[i]))
#                print i, TTW[i],'\n', XX, '\n', YY
            else:
                for j in xrange(2):      self.Trans[base_+3+j,base+3+j] = 1.
            base = base + self.DofN[i]
            base_= base_+ self.DofNini[i]            
#        for j in xrange(self.Trans.shape[0]):
#            for jj in xrange(self.Trans.shape[1]): sys.stdout.write('%6.2f'%(self.Trans[j,jj]))
#            sys.stdout.write('\n')
#        raise NameError ("Exit")
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, ShellSec, StateV, NData, RCFlag):
        # Element.__init__((self, TypeVal,nNodVal,DofEVal,nFieVal, IntTVal,nIntVal,nIntLVal, DofTVal,DofNVal, dimVal,NLGeomIVal)
        # Element.__init__(self,"SH4",4,20,3, 4,2,16, [set([1, 2, 3, 4, 5]),set([1, 2, 3, 4, 5]),set([1, 2, 3, 4, 5]),set([1, 2, 3, 4, 5])], (5,5,5,5), 21, True) # four integration points over cross section height
        Element.__init__(self,"SH4",4,20,3, 4,5,20, [set([1, 2, 3, 4, 5]),set([1, 2, 3, 4, 5]),set([1, 2, 3, 4, 5]),set([1, 2, 3, 4, 5])], (5,5,5,5), 21, True) # five integration points over cross section height
        self.PlSt = True                                    # flag for plane stress  - should be true, otherwise inconsistencies in e.g. MISES
        self.Label = Label                                  # element number in input
        self.RotM= True                                     # Flag for elements with local coordinate system for materials
        self.TensStiff = False                              # flag for tension stiffening
        self.MatN = MatName                                 # name of material
        self.Set = SetLabel                                 # label of corresponding element set
        self.ElemUpdate = True                              # element update might be required in case of NLGEOM
        self.ShellRCFlag = RCFlag
        self.InzList = [InzList[0], InzList[1], InzList[2], InzList[3]]
        self.a = array([ShellSec.Height,ShellSec.Height,ShellSec.Height,ShellSec.Height]) # shell thickness
        nRe = len(ShellSec.Reinf)                           # number of reinforcement layers
        self.Geom = zeros( (2+nRe,5), dtype=double)
        self.Geom[0,0] = 1                                  # dummy for Area / Jacobi determinant used instead
        self.Geom[1,0] = 1                                  # dummy for height / thickness
        if RCFlag and nRe>0:
            if   self.nIntL==16: i1 = 1; i2 = 16
            elif self.nIntL==20: i1 = 4; i2 = 20
            else: raise NameError("ConFemElements::SH4.__ini__: integration order not implemented for RC")
            import ConFEM2D_Basics                             # to modify sampling points for numerical integration
            for j in xrange(nRe):
                self.Geom[2+j,0] = ShellSec.Reinf[j][0]      # reinforcement cross section
                self.Geom[2+j,1] = ShellSec.Reinf[j][1]      # " lever arm
                self.Geom[2+j,2] = ShellSec.Reinf[j][2]      # " effective reinforcement ratio for tension stiffening
                self.Geom[2+j,3] = ShellSec.Reinf[j][3]      # " parameter 0<beta<=1 for tension stiffening, beta=0 no tension stiffening
                self.Geom[2+j,4] = ShellSec.Reinf[j][4]      # " direction
                tt = 2.*ShellSec.Reinf[j][1]/ShellSec.Height # local t-coordinate for reinforcement
 #               i1 = 5
                ConFEM2D_Basics.SamplePointsRCShell[SetLabel, 4, i1, i2 + 4 * j + 0] =[-0.577350269189626, -0.577350269189626, tt] # every reinforcement layer / j gets consecutive indices in base plane
                ConFEM2D_Basics.SamplePointsRCShell[SetLabel, 4, i1, i2 + 4 * j + 1] =[-0.577350269189626, 0.577350269189626, tt] #
                ConFEM2D_Basics.SamplePointsRCShell[SetLabel, 4, i1, i2 + 4 * j + 2] =[0.577350269189626, -0.577350269189626, tt] #
                ConFEM2D_Basics.SamplePointsRCShell[SetLabel, 4, i1, i2 + 4 * j + 3] =[0.577350269189626, 0.577350269189626, tt] #
                ConFEM2D_Basics.SampleWeightRCShell[SetLabel, 4, i1, i2 + 4 * j + 0]= 2. * ShellSec.Reinf[j][0] / ShellSec.Height
                ConFEM2D_Basics.SampleWeightRCShell[SetLabel, 4, i1, i2 + 4 * j + 1]= 2. * ShellSec.Reinf[j][0] / ShellSec.Height
                ConFEM2D_Basics.SampleWeightRCShell[SetLabel, 4, i1, i2 + 4 * j + 2]= 2. * ShellSec.Reinf[j][0] / ShellSec.Height
                ConFEM2D_Basics.SampleWeightRCShell[SetLabel, 4, i1, i2 + 4 * j + 3]= 2. * ShellSec.Reinf[j][0] / ShellSec.Height
                self.nIntL = self.nIntL+4                    # four more integration points in base area
        self.Data = zeros((self.nIntL,NData), dtype=float)   # storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)   # storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])      # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        i2 = FindIndexByLabel( NoList, self.InzList[2])      # find node index from node label
        i3 = FindIndexByLabel( NoList, self.InzList[3])      # find node index from node label
        self.Inzi = [ i0, i1, i2, i3]                        # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0]) # attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        NoList[i2].DofT = NoList[i2].DofT.union(self.DofT[2])
        NoList[i3].DofT = NoList[i3].DofT.union(self.DofT[3])
        self.XX =      array([[NoList[i0].XCo, NoList[i1].XCo, NoList[i2].XCo, NoList[i3].XCo],
                              [NoList[i0].YCo, NoList[i1].YCo, NoList[i2].YCo, NoList[i3].YCo],
                              [NoList[i0].ZCo, NoList[i1].ZCo, NoList[i2].ZCo, NoList[i3].ZCo]])  # collect nodal coordinates in a compact form for later use
        self.EdgeDir = zeros((3),dtype=float)           # initialize direction of 1st edge for later use
        self.ComputeEdgeDir()                           # direction of 1st edge for later use
        nn = zeros((4,3), dtype = float)
        nn[0,:] = self.CompNoNor( 0, 1, 3)              # roughly unit normals to shell surface at nodes
        nn[1,:] = self.CompNoNor( 1, 2, 0)
        nn[2,:] = self.CompNoNor( 2, 3, 1)
        nn[3,:] = self.CompNoNor( 3, 0, 2)
        self.V1 = zeros((3,4),dtype=float)              # initialize director triad
        self.V2 = zeros((3,4),dtype=float)
        self.Vn = zeros((3,4),dtype=float)
        self.Vg = zeros((3,4),dtype=float)              # director as defined per node via input data, might not be the actually used director as is ruled in the following
        self.VD = zeros((3,4),dtype=float)              # director increment in time increment
        TTW = [None, None, None, None]                  # for temporal storage of transformation coefficients
        self.SixthDoF = [False, False, False, False]
        for i in xrange(4):                             # loop over nodes of element  
            ii = self.Inzi[i]
            LL = sqrt(NoList[ii].XDi**2+NoList[ii].YDi**2+NoList[ii].ZDi**2) # length of directors
            self.Vg[0,i] = NoList[ii].XDi/LL                   # values given with input data
            self.Vg[1,i] = NoList[ii].YDi/LL                   # "
            self.Vg[2,i] = NoList[ii].ZDi/LL                   # "
            self.V1[:,i], self.V2[:,i] = self.CompleteTriad(self.Vg[:,i])
            if dot(nn[i],self.Vg[:,i])<0.8:
                print "ConFemElements::SH4.__ini__::ControlGeom: unfavorable shell director element %s local node index %s"%(str(self.Label),str(i))
                self.SixthDoF[i] = True
                NoList[ii].DofT = NoList[ii].DofT.union(set([6])) # extend types of dof for this node
                self.DofT[i] = self.DofT[i].union(set([6])) # extend types of dof for this element
                self.DofE = self.DofE + 1               # adapt number of dofs of whole element
                DofN_ = list(self.DofN)                 # transform tuple into list to make it changeable
                DofN_[i] = 6                            # one more degree of freedom for local node i
                self.DofN = tuple(DofN_)                # transform back into tuple
                self.Rot = True                         # Element has formally to be rotated as a whole
                self.RotG= True                         # geometric stiffness has also to be rotated for nonlinear geometry
                self.Trans = zeros((self.DofEini, self.DofE), dtype=float)# initialize rotation / transformation matrix for element / coordinate transformation matrix, not quadratic anymore!
                self.ComputeTransLocal(i, nn[i], TTW)      # transformation matrix for modified director system
                self.Vn[0,i] = nn[i,0]                  # final director
                self.Vn[1,i] = nn[i,1]
                self.Vn[2,i] = nn[i,2]
            else:
                self.Vn[0,i] = self.Vg[0,i]             # final director
                self.Vn[1,i] = self.Vg[1,i]
                self.Vn[2,i] = self.Vg[2,i]
        if self.Rot: self.ComputeTransLocalAll(TTW)     # build rotation / transformation matrix for element / coordinate transformation matrix, not quadratic anymore!
        self.XX0 = copy(self.XX)                        # retain initial values for NLGEOM
        self.DofI = array([[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1]],dtype=int)
        self.gg = zeros((2,3,4), dtype=float)
        self.ComputeGG()                                # scaled axes for rotational degrees of freedom
        #
        AA = 0.
        for i in xrange(16):                            # volume by numerical integration for characteristic length  
            r = SamplePoints[4,1,i][0]
            s = SamplePoints[4,1,i][1]
            f = self.JacoD(r,s,0)*SampleWeight[4,1,i]
            AA = AA + f
        self.Lch_ = sqrt(AA/(0.25*(self.a[0]+self.a[1]+self.a[2]+self.a[3]))) #/self.nInt   # characteristic length -> side length of square of same area of shell area
        self.Lch  = self.Lch_/self.nInt                 # presumably not correct
        if MaList[self.MatN].RType ==2:                 # find scaling factor for band width regularization
            x = MaList[self.MatN].bw/self.Lch_          # ratio of crack band width to characteristic length
            CrX, CrY = MaList[self.MatN].CrX, MaList[self.MatN].CrY # support points for tensile softening scaling factor 
            i = bisect_left(CrX, x) 
            if i>0 and i<(MaList[self.MatN].CrBwN+1): 
                self.CrBwS = CrY[i-1] + (x-CrX[i-1])/(CrX[i]-CrX[i-1])*(CrY[i]-CrY[i-1]) # scling factor by linear interpolation
#                print 'YYY', self.Label, self.Lch_, x, self.CrBwS  
            else:
                print 'ZZZ', AA, self.Lch_,x,'\n', CrX, '\n', CrY, i, MaList[self.MatN].CrBwN  
                raise NameError("ConFemElem:SH4.Ini2: RType 2 - element char length exceeds scaling factor interpolation")
#    def ControlGeom(self, i, nn, dd, tol):
#        if dot(nn,dd)<tol:
#            print "ConFemElements::SH4.__ini__::ControlGeom: unfavorable shell director element %s node index %s"%(str(self.Label),str(self.Inzi[i]))
#            self.Trans = zeros((self.DofE, self.DofE), dtype=float)# coordinate transformation matrix
#            for i in xrange(self.DofE): self.Trans[i,i] = 1.# fill transformation axis
#            return True
#        else:
#            if self.Rot == True: return True
#            return False
    def Lists1(self):                               # indices for first integration points in base area
        if self.nInt==1:                            # integration order   
            Lis = [0,1, 2,3]
        elif self.nInt==2: 
            Lis = [0,4, 8,12]                       # indices for first integration points in base area, 4 Gaussian integration points over cross section height
        elif self.nInt>=5: 
            Lis = [0,5,10,15]                       #   " 5 "
        return Lis
    def Lists2(self, nRe, j):                       # integration point indices specific for base point
        if self.nInt==1:
            Lis2 = [0]
        elif self.nInt==2: 
            Lis2, Lis3 = [0,1,2,3], []              # local counter for loop over cross section height, indices of reinforcement layers in the actual base plane integration point 
            j_ = j//4                               # floor division -> integer value, consecutive counter for point in base plane
            Offs2 = 16
        elif self.nInt>=5: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
            Lis2, Lis3 = [0,1,2,3,4], []            # "
            j_ = j//5                               # "
            Offs2 = 20
        for k in xrange(nRe): Lis3.append(Offs2+k*4+j_-j) # indices for reinforcement layers only, 4 should be for number of base plane integration points, -j compensates for adding j in next loop
        Lis2.extend(Lis3)                           # appends indices for reinforcement layers to local height counter 
        return Lis2, Lis3
    def FormN(self, r, s, t):
        N = array([[(1-r)*(1-s)*0.25, 0,0,0,0, (1+r)*(1-s)*0.25, 0,0,0,0, (1+r)*(1+s)*0.25, 0,0,0,0, (1-r)*(1+s)*0.25, 0,0,0,0],
                   [0,(1-r)*(1-s)*0.25, 0,0,0,0, (1+r)*(1-s)*0.25, 0,0,0,0, (1+r)*(1+s)*0.25, 0,0,0,0, (1-r)*(1+s)*0.25, 0,0,0],
                   [0,0,(1-r)*(1-s)*0.25, 0,0,0,0, (1+r)*(1-s)*0.25, 0,0,0,0, (1+r)*(1+s)*0.25, 0,0,0,0, (1-r)*(1+s)*0.25, 0,0]])
        return N
    def Basics(self, r, s, t):
        N =  array([(1-r)*(1-s)*0.25, (1+r)*(1-s)*0.25, (1+r)*(1+s)*0.25, (1-r)*(1+s)*0.25])
        br = array([(-1+s)*0.25,      ( 1-s)*0.25,     ( 1+s)*0.25,      -( 1+s)*0.25])
        bs = array([(-1+r)*0.25,     -( 1+r)*0.25,     ( 1+r)*0.25,       ( 1-r)*0.25])
        JJ = zeros((3,3),dtype=float)
        for k in xrange(3): JJ[k,0]=br[0]*(self.XX[k,0]+t/2.*self.a[0]*self.Vn[k,0])+br[1]*(self.XX[k,1]+t/2.*self.a[1]*self.Vn[k,1])+br[2]*(self.XX[k,2]+t/2.*self.a[2]*self.Vn[k,2])+br[3]*(self.XX[k,3]+t/2.*self.a[3]*self.Vn[k,3])
        for k in xrange(3): JJ[k,1]=bs[0]*(self.XX[k,0]+t/2.*self.a[0]*self.Vn[k,0])+bs[1]*(self.XX[k,1]+t/2.*self.a[1]*self.Vn[k,1])+bs[2]*(self.XX[k,2]+t/2.*self.a[2]*self.Vn[k,2])+bs[3]*(self.XX[k,3]+t/2.*self.a[3]*self.Vn[k,3])
        for k in xrange(3): JJ[k,2]=N[0]*(1/2.*self.a[0]*self.Vn[k,0])+N[1]*(1/2.*self.a[1]*self.Vn[k,1])+N[2]*(1/2.*self.a[2]*self.Vn[k,2])+N[3]*(1/2.*self.a[3]*self.Vn[k,3])
        JI = inv(JJ)
        ll=sqrt(JJ[0,2]**2+JJ[1,2]**2+JJ[2,2]**2)   
        vv = array([[0.,0.,JJ[0,2]/ll],[0.,0.,JJ[1,2]/ll],[0.,0.,JJ[2,2]/ll]]) # unit normal of local coordinate system, 3rd column
        LoC = False                                                        
        if LoC:                                         # local right handed coordinate system with 1st direction / column aligned to element edge
            x0 = self.EdgeDir[1]*vv[2,2]-self.EdgeDir[2]*vv[1,2]
            x1 = self.EdgeDir[2]*vv[0,2]-self.EdgeDir[0]*vv[2,2]
            x2 = self.EdgeDir[0]*vv[1,2]-self.EdgeDir[1]*vv[0,2]
            xx = sqrt(x0**2+x1**2+x2**2)
            vv[0,1] = -x0/xx                            # 2nd column, approx perp. to element edge, reversed in sign to preserve right handedness
            vv[1,1] = -x1/xx
            vv[2,1] = -x2/xx 
            x0 = vv[1,2]*vv[2,1]-vv[2,2]*vv[1,1]
            x1 = vv[2,2]*vv[0,1]-vv[0,2]*vv[2,1]
            x2 = vv[0,2]*vv[1,1]-vv[1,2]*vv[0,1]
            xx = sqrt(x0**2+x1**2+x2**2)
            vv[0,0] = -x0/xx                            # 1st column, approx aligned to element edge, sign reversal of 2nd column is implicitely corrected
            vv[1,0] = -x1/xx
            vv[2,0] = -x2/xx 
        else:                                           # local coordinate system with one axis aligned to global axis
            if abs(vv[1,2])<0.99:                       # local coordinate system V_1 from cross product of V_n and e_y ( V_1 in e_x - e_z plane) if V_n is not to much aligned to e_y  
                ll = sqrt(vv[2,2]**2+vv[0,2]**2)        # length of V_1
                vv[0,0] = vv[2,2]/ll                    # V_1[0] normalized;  V1[1] = 0
                vv[2,0] =-vv[0,2]/ll                    # V_1[2] normalized

                vv[0,1] = vv[1,2]*vv[2,0]               # as V_n and V_1 are orthogonal and both have unit length V_2 also should have unit length
                vv[1,1] = vv[2,2]*vv[0,0]-vv[0,2]*vv[2,0]
                vv[2,1] =-vv[1,2]*vv[0,0]
            else:                                       # local coordinate system V_1 from cross product of V_n and e_x ( V_1 in e_y - e_z plane)
                ll = sqrt(vv[2,2]**2+vv[1,2]**2)        # length of V_1
                vv[1,0] =-vv[2,2]/ll                    # V_1[0] normalized;  V1[0] = 0
                vv[2,0] = vv[1,2]/ll                    # V_1[2] normalized

                vv[0,1] = vv[1,2]*vv[2,0]-vv[2,2]*vv[1,0] # as V_n and V_1 are orthogonal and both have unit length V_2 also should have unit length
                vv[1,1] =-vv[0,2]*vv[2,0]
                vv[2,1] = vv[0,2]*vv[1,0]
        return N, br, bs, JJ, JI, vv
    def FormB(self, r, s, t_, NLg):
        t = t_
        N, br, bs, JJ, JI, vv = self.Basics( r, s, t)
        det = JJ[0,0]*JJ[1,1]*JJ[2,2]-JJ[0,0]*JJ[1,2]*JJ[2,1]-JJ[1,0]*JJ[0,1]*JJ[2,2]+JJ[1,0]*JJ[0,2]*JJ[2,1]+JJ[2,0]*JJ[0,1]*JJ[1,2]-JJ[2,0]*JJ[0,2]*JJ[1,1]
        HH = zeros((3,2,4),dtype=float)
        for k in xrange(4):
            HH[0,0,k]=JJ[0,0]*self.gg[0,0,k]+JJ[1,0]*self.gg[0,1,k]+JJ[2,0]*self.gg[0,2,k]
            HH[0,1,k]=JJ[0,0]*self.gg[1,0,k]+JJ[1,0]*self.gg[1,1,k]+JJ[2,0]*self.gg[1,2,k]
            HH[1,0,k]=JJ[0,1]*self.gg[0,0,k]+JJ[1,1]*self.gg[0,1,k]+JJ[2,1]*self.gg[0,2,k]
            HH[1,1,k]=JJ[0,1]*self.gg[1,0,k]+JJ[1,1]*self.gg[1,1,k]+JJ[2,1]*self.gg[1,2,k]
            HH[2,0,k]=JJ[0,2]*self.gg[0,0,k]+JJ[1,2]*self.gg[0,1,k]+JJ[2,2]*self.gg[0,2,k]
            HH[2,1,k]=JJ[0,2]*self.gg[1,0,k]+JJ[1,2]*self.gg[1,1,k]+JJ[2,2]*self.gg[1,2,k]
        BB = zeros((6,20),dtype=float)
        for k in xrange(4):
            BB[0,k*5+0]=JJ[0,0]*br[k]
            BB[0,k*5+1]=JJ[1,0]*br[k]
            BB[0,k*5+2]=JJ[2,0]*br[k]
            BB[0,k*5+3]=               t*br[k]*HH[0,0,k]
            BB[0,k*5+4]=               t*br[k]*HH[0,1,k]
            BB[1,k*5+0]=JJ[0,1]*bs[k]
            BB[1,k*5+1]=JJ[1,1]*bs[k]
            BB[1,k*5+2]=JJ[2,1]*bs[k]
            BB[1,k*5+3]=               t*bs[k]*HH[1,0,k]
            BB[1,k*5+4]=               t*bs[k]*HH[1,1,k]
            BB[2,k*5+3]=N[k]*HH[2,0,k]
            BB[2,k*5+4]=N[k]*HH[2,1,k]
            BB[5,k*5+0]=JJ[0,0]*bs[k]  +JJ[0,1]*br[k]
            BB[5,k*5+1]=JJ[1,0]*bs[k]  +JJ[1,1]*br[k]
            BB[5,k*5+2]=JJ[2,0]*bs[k]  +JJ[2,1]*br[k]
            BB[5,k*5+3]=     t*(bs[k]*HH[0,0,k]+br[k]*HH[1,0,k])
            BB[5,k*5+4]=     t*(bs[k]*HH[0,1,k]+br[k]*HH[1,1,k])
        flag = True    #  True --> Assumed-Natural-Strain-Method to avoid transverse shear locking
        if not flag:
            for k in xrange(4):
                BB[3,(k-0)*5+0]=JJ[0,2]*bs[k]
                BB[3,(k-0)*5+1]=JJ[1,2]*bs[k]
                BB[3,(k-0)*5+2]=JJ[2,2]*bs[k]
                BB[3,(k-0)*5+3]=N[k]*HH[1,0,k]+t*bs[k]*HH[2,0,k]
                BB[3,(k-0)*5+4]=N[k]*HH[1,1,k]+t*bs[k]*HH[2,1,k]
                BB[4,(k-0)*5+0]=JJ[0,2]*br[k]
                BB[4,(k-0)*5+1]=JJ[1,2]*br[k]
                BB[4,(k-0)*5+2]=JJ[2,2]*br[k]
                BB[4,(k-0)*5+3]=N[k]*HH[0,0,k]+t*br[k]*HH[2,0,k]
                BB[4,(k-0)*5+4]=N[k]*HH[0,1,k]+t*br[k]*HH[2,1,k]
        else:
#                t = 0
                J01D=-0.5*(self.XX[0,1]-self.XX[0,2])-0.25*t*(self.a[1]*self.Vn[0,1]-self.a[2]*self.Vn[0,2])
                J11D=-0.5*(self.XX[1,1]-self.XX[1,2])-0.25*t*(self.a[1]*self.Vn[1,1]-self.a[2]*self.Vn[1,2])
                J21D=-0.5*(self.XX[2,1]-self.XX[2,2])-0.25*t*(self.a[1]*self.Vn[2,1]-self.a[2]*self.Vn[2,2])
                J01B=-0.5*(self.XX[0,0]-self.XX[0,3])-0.25*t*(self.a[0]*self.Vn[0,0]-self.a[3]*self.Vn[0,3])
                J11B=-0.5*(self.XX[1,0]-self.XX[1,3])-0.25*t*(self.a[0]*self.Vn[1,0]-self.a[3]*self.Vn[1,3])
                J21B=-0.5*(self.XX[2,0]-self.XX[2,3])-0.25*t*(self.a[0]*self.Vn[2,0]-self.a[3]*self.Vn[2,3])
                J00A= 0.5*(self.XX[0,2]-self.XX[0,3])+0.25*t*(self.a[2]*self.Vn[0,2]-self.a[3]*self.Vn[0,3])
                J10A= 0.5*(self.XX[1,2]-self.XX[1,3])+0.25*t*(self.a[2]*self.Vn[1,2]-self.a[3]*self.Vn[1,3])
                J20A= 0.5*(self.XX[2,2]-self.XX[2,3])+0.25*t*(self.a[2]*self.Vn[2,2]-self.a[3]*self.Vn[2,3])
                J00C=-0.5*(self.XX[0,0]-self.XX[0,1])-0.25*t*(self.a[0]*self.Vn[0,0]-self.a[1]*self.Vn[0,1])
                J10C=-0.5*(self.XX[1,0]-self.XX[1,1])-0.25*t*(self.a[0]*self.Vn[1,0]-self.a[1]*self.Vn[1,1])
                J20C=-0.5*(self.XX[2,0]-self.XX[2,1])-0.25*t*(self.a[0]*self.Vn[2,0]-self.a[1]*self.Vn[2,1])
                J02D=0.25*(self.a[1]*self.Vn[0,1]+self.a[2]*self.Vn[0,2])
                J12D=0.25*(self.a[1]*self.Vn[1,1]+self.a[2]*self.Vn[1,2])
                J22D=0.25*(self.a[1]*self.Vn[2,1]+self.a[2]*self.Vn[2,2])
                J02B=0.25*(self.a[0]*self.Vn[0,0]+self.a[3]*self.Vn[0,3])
                J12B=0.25*(self.a[0]*self.Vn[1,0]+self.a[3]*self.Vn[1,3])
                J22B=0.25*(self.a[0]*self.Vn[2,0]+self.a[3]*self.Vn[2,3])
                J02A=0.25*(self.a[2]*self.Vn[0,2]+self.a[3]*self.Vn[0,3])
                J12A=0.25*(self.a[2]*self.Vn[1,2]+self.a[3]*self.Vn[1,3])
                J22A=0.25*(self.a[2]*self.Vn[2,2]+self.a[3]*self.Vn[2,3])
                J02C=0.25*(self.a[0]*self.Vn[0,0]+self.a[1]*self.Vn[0,1])
                J12C=0.25*(self.a[0]*self.Vn[1,0]+self.a[1]*self.Vn[1,1])
                J22C=0.25*(self.a[0]*self.Vn[2,0]+self.a[1]*self.Vn[2,1])
                BB[3,0] =-0.25*(1-r)*J02B
                BB[3,1] =-0.25*(1-r)*J12B
                BB[3,2] =-0.25*(1-r)*J22B
                BB[3,3] = 0.25*(1-r)*((J01B*self.gg[0,0,0]+J11B*self.gg[0,1,0]+J21B*self.gg[0,2,0])-t*(J02B*self.gg[0,0,0]+J12B*self.gg[0,1,0]+J22B*self.gg[0,2,0]))
                BB[3,4] = 0.25*(1-r)*((J01B*self.gg[1,0,0]+J11B*self.gg[1,1,0]+J21B*self.gg[1,2,0])-t*(J02B*self.gg[1,0,0]+J12B*self.gg[1,1,0]+J22B*self.gg[1,2,0]))
                BB[3,5] =-0.25*(1+r)*J02D 
                BB[3,6] =-0.25*(1+r)*J12D
                BB[3,7] =-0.25*(1+r)*J22D
                BB[3,8] = 0.25*(1+r)*((J01D*self.gg[0,0,1]+J11D*self.gg[0,1,1]+J21D*self.gg[0,2,1])-t*(J02D*self.gg[0,0,1]+J12D*self.gg[0,1,1]+J22D*self.gg[0,2,1]))
                BB[3,9]= 0.25*(1+r)*((J01D*self.gg[1,0,1]+J11D*self.gg[1,1,1]+J21D*self.gg[1,2,1])-t*(J02D*self.gg[1,0,1]+J12D*self.gg[1,1,1]+J22D*self.gg[1,2,1]))
                BB[3,10]= 0.25*(1+r)*J02D 
                BB[3,11]= 0.25*(1+r)*J12D
                BB[3,12]= 0.25*(1+r)*J22D
                BB[3,13]= 0.25*(1+r)*((J01D*self.gg[0,0,2]+J11D*self.gg[0,1,2]+J21D*self.gg[0,2,2])+t*(J02D*self.gg[0,0,2]+J12D*self.gg[0,1,2]+J22D*self.gg[0,2,2]))
                BB[3,14]= 0.25*(1+r)*((J01D*self.gg[1,0,2]+J11D*self.gg[1,1,2]+J21D*self.gg[1,2,2])+t*(J02D*self.gg[1,0,2]+J12D*self.gg[1,1,2]+J22D*self.gg[1,2,2]))
                BB[3,15]= 0.25*(1-r)*J02B
                BB[3,16]= 0.25*(1-r)*J12B
                BB[3,17]= 0.25*(1-r)*J22B
                BB[3,18]= 0.25*(1-r)*((J01B*self.gg[0,0,3]+J11B*self.gg[0,1,3]+J21B*self.gg[0,2,3])+t*(J02B*self.gg[0,0,3]+J12B*self.gg[0,1,3]+J22B*self.gg[0,2,3]))
                BB[3,19]= 0.25*(1-r)*((J01B*self.gg[1,0,3]+J11B*self.gg[1,1,3]+J21B*self.gg[1,2,3])+t*(J02B*self.gg[1,0,3]+J12B*self.gg[1,1,3]+J22B*self.gg[1,2,3]))
                BB[4,0] =-0.25*(1-s)*J02C
                BB[4,1] =-0.25*(1-s)*J12C
                BB[4,2] =-0.25*(1-s)*J22C
                BB[4,3] = 0.25*(1-s)*((J00C*self.gg[0,0,0]+J10C*self.gg[0,1,0]+J20C*self.gg[0,2,0])-t*(J02C*self.gg[0,0,0]+J12C*self.gg[0,1,0]+J22C*self.gg[0,2,0]))
                BB[4,4] = 0.25*(1-s)*((J00C*self.gg[1,0,0]+J10C*self.gg[1,1,0]+J20C*self.gg[1,2,0])-t*(J02C*self.gg[1,0,0]+J12C*self.gg[1,1,0]+J22C*self.gg[1,2,0]))
                BB[4,5] = 0.25*(1-s)*J02C 
                BB[4,6] = 0.25*(1-s)*J12C
                BB[4,7] = 0.25*(1-s)*J22C
                BB[4,8] = 0.25*(1-s)*((J00C*self.gg[0,0,1]+J10C*self.gg[0,1,1]+J20C*self.gg[0,2,1])+t*(J02C*self.gg[0,0,1]+J12C*self.gg[0,1,1]+J22C*self.gg[0,2,1]))
                BB[4,9] = 0.25*(1-s)*((J00C*self.gg[1,0,1]+J10C*self.gg[1,1,1]+J20C*self.gg[1,2,1])+t*(J02C*self.gg[1,0,1]+J12C*self.gg[1,1,1]+J22C*self.gg[1,2,1]))
                BB[4,10]= 0.25*(1+s)*J02A 
                BB[4,11]= 0.25*(1+s)*J12A 
                BB[4,12]= 0.25*(1+s)*J22A
                BB[4,13]= 0.25*(1+s)*((J00A*self.gg[0,0,2]+J10A*self.gg[0,1,2]+J20A*self.gg[0,2,2])+t*(J02A*self.gg[0,0,2]+J12A*self.gg[0,1,2]+J22A*self.gg[0,2,2]))
                BB[4,14]= 0.25*(1+s)*((J00A*self.gg[1,0,2]+J10A*self.gg[1,1,2]+J20A*self.gg[1,2,2])+t*(J02A*self.gg[1,0,2]+J12A*self.gg[1,1,2]+J22A*self.gg[1,2,2]))
                BB[4,15]=-0.25*(1+s)*J02A 
                BB[4,16]=-0.25*(1+s)*J12A
                BB[4,17]=-0.25*(1+s)*J22A
                BB[4,18]= 0.25*(1+s)*((J00A*self.gg[0,0,3]+J10A*self.gg[0,1,3]+J20A*self.gg[0,2,3])-t*(J02A*self.gg[0,0,3]+J12A*self.gg[0,1,3]+J22A*self.gg[0,2,3]))
                BB[4,19]= 0.25*(1+s)*((J00A*self.gg[1,0,3]+J10A*self.gg[1,1,3]+J20A*self.gg[1,2,3])-t*(J02A*self.gg[1,0,3]+J12A*self.gg[1,1,3]+J22A*self.gg[1,2,3]))

#       td=array([[JI[0,0]*vv[0,0]+JI[0,1]*vv[0,1]+JI[0,2]*vv[0,2],JI[0,0]*vv[1,0]+JI[0,1]*vv[1,1]+JI[0,2]*vv[1,2],JI[0,0]*vv[2,0]+JI[0,1]*vv[2,1]+JI[0,2]*vv[2,2]],
#                 [JI[1,0]*vv[0,0]+JI[1,1]*vv[0,1]+JI[1,2]*vv[0,2],JI[1,0]*vv[1,0]+JI[1,1]*vv[1,1]+JI[1,2]*vv[1,2],JI[1,0]*vv[2,0]+JI[1,1]*vv[2,1]+JI[1,2]*vv[2,2]],
#                 [JI[2,0]*vv[0,0]+JI[2,1]*vv[0,1]+JI[2,2]*vv[0,2],JI[2,0]*vv[1,0]+JI[2,1]*vv[1,1]+JI[2,2]*vv[1,2],JI[2,0]*vv[2,0]+JI[2,1]*vv[2,1]+JI[2,2]*vv[2,2]]])
        td=array([[JI[0,0]*vv[0,0]+JI[0,1]*vv[1,0]+JI[0,2]*vv[2,0],JI[0,0]*vv[0,1]+JI[0,1]*vv[1,1]+JI[0,2]*vv[2,1],JI[0,0]*vv[0,2]+JI[0,1]*vv[1,2]+JI[0,2]*vv[2,2]],
                  [JI[1,0]*vv[0,0]+JI[1,1]*vv[1,0]+JI[1,2]*vv[2,0],JI[1,0]*vv[0,1]+JI[1,1]*vv[1,1]+JI[1,2]*vv[2,1],JI[1,0]*vv[0,2]+JI[1,1]*vv[1,2]+JI[1,2]*vv[2,2]],
                  [JI[2,0]*vv[0,0]+JI[2,1]*vv[1,0]+JI[2,2]*vv[2,0],JI[2,0]*vv[0,1]+JI[2,1]*vv[1,1]+JI[2,2]*vv[2,1],JI[2,0]*vv[0,2]+JI[2,1]*vv[1,2]+JI[2,2]*vv[2,2]]])
        TD=array([[td[0,0]**2,         td[1,0]**2,       td[2,0]**2,     td[1,0]*td[2,0],                td[0,0]*td[2,0],                td[0,0]*td[1,0]],
                  [td[0,1]**2,         td[1,1]**2,       td[2,1]**2,     td[1,1]*td[2,1],                td[0,1]*td[2,1],                td[0,1]*td[1,1]],
                  [td[0,2]**2,         td[1,2]**2,       td[2,2]**2,     td[1,2]*td[2,2],                td[0,2]*td[2,2],                td[0,2]*td[1,2]],
                  [2*td[0,1]*td[0,2],2*td[1,1]*td[1,2],2*td[2,1]*td[2,2],td[1,1]*td[2,2]+td[2,1]*td[1,2],td[0,1]*td[2,2]+td[0,2]*td[2,1],td[0,1]*td[1,2]+td[1,1]*td[0,2]],
                  [2*td[0,0]*td[0,2],2*td[1,0]*td[1,2],2*td[2,0]*td[2,2],td[1,0]*td[2,2]+td[2,0]*td[1,2],td[0,0]*td[2,2]+td[0,2]*td[2,0],td[0,0]*td[1,2]+td[1,0]*td[0,2]],
                  [2*td[0,0]*td[0,1],2*td[1,0]*td[1,1],2*td[2,0]*td[2,1],td[1,0]*td[2,1]+td[2,0]*td[1,1],td[0,0]*td[2,1]+td[0,1]*td[2,0],td[0,0]*td[1,1]+td[1,0]*td[0,1]]])
#            for kk in xrange(3): sys.stdout.write('%10.4f'%(JJ[k,kk]))
#            sys.stdout.write('\n')  
#        for k in xrange(3): 
#            for kk in xrange(3): sys.stdout.write('%10.4f'%(vv[k,kk]))
#            sys.stdout.write('\n')  
#        for k in xrange(3): 
#            for kk in xrange(3): sys.stdout.write('%10.4f'%(td[k,kk]))
#            sys.stdout.write('\n')  
        return BB, det, TD
    
    def FormT(self, r, s, t):                               # interpolation on temperature - currently not used
        T = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return T
    def FormX(self, r, s, t):                               # interpolation on geometry
        X = array([(1-r)*(1-s)*0.25, (1+r)*(1-s)*0.25, (1+r)*(1+s)*0.25, (1-r)*(1+s)*0.25])
        return X
    def UpdateElemData(self):
        for i in xrange(4):                                 # loop over nodes
            self.Vn[:,i] = self.Vn[:,i] + self.VD[:,i] 
            self.Vg[:,i] = self.Vg[:,i] + self.VD[:,i]
            self.V1[:,i], self.V2[:,i] = self.CompleteTriad(self.Vg[:,i]) # may be overwritten in the following in case of a local system
        if self.Rot:
            TTW = [None, None, None, None]                  # for temporal storage of transformation coefficients
            for i in xrange(4):                             # loop over nodes of element  
                if self.SixthDoF[i]: self.ComputeTransLocal(i,self.Vn[:,i],TTW)  # transformation matrix for modified director system, completed director triad for local system if required
            self.ComputeTransLocalAll(TTW)                  # build rotation / transformation matrix for element / coordinate transformation matrix, not quadratic anymore!
        self.ComputeGG()                                    # scaled axes for rotational degrees of freedom
    def UpdateCoord(self, dis, ddis ):
        # length of dis, ddis must be 20 here!
        self.XX[0,0] = self.XX0[0,0] + dis[0]               # x-displacement 1st node
        self.XX[1,0] = self.XX0[1,0] + dis[1]               # y-displacement
        self.XX[2,0] = self.XX0[2,0] + dis[2]               # z-displacement
        self.XX[0,1] = self.XX0[0,1] + dis[5]               # x-displacement 2nd node
        self.XX[1,1] = self.XX0[1,1] + dis[6]               # y-displacement
        self.XX[2,1] = self.XX0[2,1] + dis[7]
        self.XX[0,2] = self.XX0[0,2] + dis[10]              # x-displacement 3rd node
        self.XX[1,2] = self.XX0[1,2] + dis[11]              # y-displacement
        self.XX[2,2] = self.XX0[2,2] + dis[12]
        self.XX[0,3] = self.XX0[0,3] + dis[15]              # x-displacement 4th node
        self.XX[1,3] = self.XX0[1,3] + dis[16]              # y-displacement
        self.XX[2,3] = self.XX0[2,3] + dis[17]
        self.ComputeEdgeDir()
        self.VD[0,0] =  - self.V2[0,0]*ddis[3]  + self.V1[0,0]*ddis[4] # ddis is increment of rotation angle in time increment; V1, V2 are displaced director triad at beginning of time increment
        self.VD[1,0] =  - self.V2[1,0]*ddis[3]  + self.V1[1,0]*ddis[4]
        self.VD[2,0] =  - self.V2[2,0]*ddis[3]  + self.V1[2,0]*ddis[4]
        self.VD[0,1] =  - self.V2[0,1]*ddis[8]  + self.V1[0,1]*ddis[9]
        self.VD[1,1] =  - self.V2[1,1]*ddis[8]  + self.V1[1,1]*ddis[9]
        self.VD[2,1] =  - self.V2[2,1]*ddis[8]  + self.V1[2,1]*ddis[9]
        self.VD[0,2] =  - self.V2[0,2]*ddis[13] + self.V1[0,2]*ddis[14]
        self.VD[1,2] =  - self.V2[1,2]*ddis[13] + self.V1[1,2]*ddis[14]
        self.VD[2,2] =  - self.V2[2,2]*ddis[13] + self.V1[2,2]*ddis[14]
        self.VD[0,3] =  - self.V2[0,3]*ddis[18] + self.V1[0,3]*ddis[19]
        self.VD[1,3] =  - self.V2[1,3]*ddis[18] + self.V1[1,3]*ddis[19]
        self.VD[2,3] =  - self.V2[2,3]*ddis[18] + self.V1[2,3]*ddis[19]
        for i in xrange(4): self.LengV(self.Vn[:,i])
    def GeomStiff(self, r, s, t, sig):
        GeomK = zeros((20, 20), dtype=float)
        N =  array([(1-r)*(1-s)*0.25, (1+r)*(1-s)*0.25, (1+r)*(1+s)*0.25, (1-r)*(1+s)*0.25])
        br = array([(-1+s)*0.25,      ( 1-s)*0.25,     ( 1+s)*0.25,      -( 1+s)*0.25])
        bs = array([(-1+r)*0.25,     -( 1+r)*0.25,     ( 1+r)*0.25,       ( 1-r)*0.25])
        t2 = t*t
#        bi = 0                                              # base for dof index
        for i in xrange(4):                                 # loop over nodes
            bi = i*5                                        # index for dofs
#            bj = 0                                          # base for dof index
            for j in xrange(4):                             # loop over nodes
                bj = j*5                                    # index for dofs
                S11   = sig[0] * br[i]*br[j]                #  s_11 sequence according to voigt notation
                S22   = sig[1] * bs[i]*bs[j]                #  s_22
                S33   = sig[2] * N[i] *N[j]                 #  s_33
                S23si = sig[3] * bs[i]*N[j]                 #  s_23
                S23sj = sig[3] * N[i] *bs[j]                #  s_23
                S13ri = sig[4] * br[i]*N[j]                 #  s_13
                S13rj = sig[4] * N[i] *br[j]                #  s_13
                S12   = sig[5] *(br[i]*bs[j]+bs[i]*br[j])   #  s_12     
                ggg00 = (self.gg[0,0,i]*self.gg[0,0,j] + self.gg[0,1,i]*self.gg[0,1,j] + self.gg[0,2,i]*self.gg[0,2,j])
                ggg11 = (self.gg[1,0,i]*self.gg[1,0,j] + self.gg[1,1,i]*self.gg[1,1,j] + self.gg[1,2,i]*self.gg[1,2,j])
                ggg01 = (self.gg[0,0,i]*self.gg[1,0,j] + self.gg[0,1,i]*self.gg[1,1,j] + self.gg[0,2,i]*self.gg[1,2,j])
                ggg10 = (self.gg[1,0,i]*self.gg[0,0,j] + self.gg[1,1,i]*self.gg[0,1,j] + self.gg[1,2,i]*self.gg[0,2,j])
                
                if False:
                    GeomK[bi,  bj]   = S11+S22+S12
                    GeomK[bi+1,bj+1] = S11+S22+S12
                    GeomK[bi+2,bj+2] = S11+S22+S12
                    GeomK[bi+3,bj+3] = (S11*t2 + S22*t2 + S12*t2 + S33 + S13ri*t + S13rj*t + S23si*t + S23sj*t)*ggg00
                    GeomK[bi+4,bj+4] = (S11*t2 + S22*t2 + S12*t2 + S33 + S13ri*t + S13rj*t + S23si*t + S23sj*t)*ggg11
                    GeomK[bi,  bj+1] = 0. 
                    GeomK[bi,  bj+2] = 0.
                    GeomK[bi,  bj+3] = (S11*t + S22*t + S13ri + S23si + S12*t)*self.gg[0,0,j]
                    GeomK[bi,  bj+4] = (S11*t + S22*t + S13ri + S23si + S12*t)*self.gg[1,0,j]
                    GeomK[bi+1,bj]   = 0.
                    GeomK[bi+1,bj+2] = 0.
                    GeomK[bi+1,bj+3] = (S11*t + S22*t + S13ri + S23si + S12*t)*self.gg[0,1,j]
                    GeomK[bi+1,bj+4] = (S11*t + S22*t + S13ri + S23si + S12*t)*self.gg[1,1,j]
                    GeomK[bi+2,bj]   = 0.
                    GeomK[bi+2,bj+1] = 0.
                    GeomK[bi+2,bj+3] = (S11*t + S22*t + S12*t + S13ri + S23si)*self.gg[0,2,j]
                    GeomK[bi+2,bj+4] = (S11*t + S22*t + S12*t + S13ri + S23si)*self.gg[1,2,j]
                    GeomK[bi+3,bj]   = (S11*t + S22*t + S12*t + S13rj + S23sj)*self.gg[0,0,i]
                    GeomK[bi+3,bj+1] = (S11*t + S22*t + S12*t + S13rj + S23sj)*self.gg[0,1,i]
                    GeomK[bi+3,bj+2] = (S11*t + S22*t + S12*t + S13rj + S23sj)*self.gg[0,2,i]
                    GeomK[bi+3,bj+4] = (S11*t2 + S22*t2 + S12*t2 + S33 + S13ri*t + S13rj*t + S23si*t + S23sj*t)*ggg01
                    GeomK[bi+4,bj]   = (S11*t + S22*t + S12*t + S13rj + S23sj)*self.gg[1,0,i]
                    GeomK[bi+4,bj+1] = (S11*t + S22*t + S12*t + S13rj + S23sj)*self.gg[1,1,i]
                    GeomK[bi+4,bj+2] = (S11*t + S22*t + S12*t + S13rj + S23sj)*self.gg[1,2,i]
                    GeomK[bi+4,bj+3] = (S11*t2 + S22*t2 + S12*t2 + S33 + S13ri*t + S13rj*t + S23si*t + S23sj*t)*ggg10
                else:
                    GeomK[bi,  bj]   = S11+S22+S12
                    GeomK[bi+1,bj+1] = S11+S22+S12
                    GeomK[bi+2,bj+2] = S11+S22+S12
                    GeomK[bi+3,bj+3] = (S11*t2 + S22*t2 + S12*t2 + S33)*ggg00
                    GeomK[bi+4,bj+4] = (S11*t2 + S22*t2 + S12*t2 + S33)*ggg11
                    GeomK[bi,  bj+1] = 0. 
                    GeomK[bi,  bj+2] = 0.
                    GeomK[bi,  bj+3] = (S11*t + S22*t + S12*t)*self.gg[0,0,j]
                    GeomK[bi,  bj+4] = (S11*t + S22*t + S12*t)*self.gg[1,0,j]
                    GeomK[bi+1,bj]   = 0.
                    GeomK[bi+1,bj+2] = 0.
                    GeomK[bi+1,bj+3] = (S11*t + S22*t + S12*t)*self.gg[0,1,j]
                    GeomK[bi+1,bj+4] = (S11*t + S22*t + S12*t)*self.gg[1,1,j]
                    GeomK[bi+2,bj]   = 0.
                    GeomK[bi+2,bj+1] = 0.
                    GeomK[bi+2,bj+3] = (S11*t + S22*t + S12*t)*self.gg[0,2,j]
                    GeomK[bi+2,bj+4] = (S11*t + S22*t + S12*t)*self.gg[1,2,j]
                    GeomK[bi+3,bj]   = (S11*t + S22*t + S12*t)*self.gg[0,0,i]
                    GeomK[bi+3,bj+1] = (S11*t + S22*t + S12*t)*self.gg[0,1,i]
                    GeomK[bi+3,bj+2] = (S11*t + S22*t + S12*t)*self.gg[0,2,i]
                    GeomK[bi+3,bj+4] = (S11*t2 + S22*t2 + S12*t2 + S33)*ggg01
                    GeomK[bi+4,bj]   = (S11*t + S22*t + S12*t)*self.gg[1,0,i]
                    GeomK[bi+4,bj+1] = (S11*t + S22*t + S12*t)*self.gg[1,1,i]
                    GeomK[bi+4,bj+2] = (S11*t + S22*t + S12*t)*self.gg[1,2,i]
                    GeomK[bi+4,bj+3] = (S11*t2 + S22*t2 + S12*t2 + S33)*ggg10
                    if True:
                        GeomK[bi+3,bj+3] += (S13ri*t + S13rj*t + S23si*t + S23sj*t)*ggg00
                        GeomK[bi+4,bj+4] += (S13ri*t + S13rj*t + S23si*t + S23sj*t)*ggg11
                        GeomK[bi,  bj+3] += (S13ri + S23si)*self.gg[0,0,j]
                        GeomK[bi,  bj+4] += (S13ri + S23si)*self.gg[1,0,j]
                        GeomK[bi+1,bj+3] += (S13ri + S23si)*self.gg[0,1,j]
                        GeomK[bi+1,bj+4] += (S13ri + S23si)*self.gg[1,1,j]
                        GeomK[bi+2,bj+3] += (S13ri + S23si)*self.gg[0,2,j]
                        GeomK[bi+2,bj+4] += (S13ri + S23si)*self.gg[1,2,j]
                        GeomK[bi+3,bj]   += (S13rj + S23sj)*self.gg[0,0,i]
                        GeomK[bi+3,bj+1] += (S13rj + S23sj)*self.gg[0,1,i]
                        GeomK[bi+3,bj+2] += (S13rj + S23sj)*self.gg[0,2,i]
                        GeomK[bi+3,bj+4] += (S13ri*t + S13rj*t + S23si*t + S23sj*t)*ggg01
                        GeomK[bi+4,bj]   += (S13rj + S23sj)*self.gg[1,0,i]
                        GeomK[bi+4,bj+1] += (S13rj + S23sj)*self.gg[1,1,i]
                        GeomK[bi+4,bj+2] += (S13rj + S23sj)*self.gg[1,2,i]
                        GeomK[bi+4,bj+3] += (S13ri*t + S13rj*t + S23si*t + S23sj*t)*ggg10

        return GeomK
    def JacoD(self, r, s, t):
        N = array([(1-r)*(1-s)*0.25, (1+r)*(1-s)*0.25, (1+r)*(1+s)*0.25, (1-r)*(1+s)*0.25])
        br = array([(-1+s)*0.25,( 1-s)*0.25,( 1+s)*0.25,-( 1+s)*0.25])
        bs = array([(-1+r)*0.25,-( 1+r)*0.25,( 1+r)*0.25,( 1-r)*0.25])
        JJ = zeros((3,3),dtype=float)
        for k in xrange(3): JJ[k,0]=br[0]*(self.XX[k,0]+t/2.*self.a[0]*self.Vn[k,0])+br[1]*(self.XX[k,1]+t/2.*self.a[1]*self.Vn[k,1])+br[2]*(self.XX[k,2]+t/2.*self.a[2]*self.Vn[k,2])+br[3]*(self.XX[k,3]+t/2.*self.a[3]*self.Vn[k,3])
        for k in xrange(3): JJ[k,1]=bs[0]*(self.XX[k,0]+t/2.*self.a[0]*self.Vn[k,0])+bs[1]*(self.XX[k,1]+t/2.*self.a[1]*self.Vn[k,1])+bs[2]*(self.XX[k,2]+t/2.*self.a[2]*self.Vn[k,2])+bs[3]*(self.XX[k,3]+t/2.*self.a[3]*self.Vn[k,3])
        for k in xrange(3): JJ[k,2]=N[0]*(1/2.*self.a[0]*self.Vn[k,0])+N[1]*(1/2.*self.a[1]*self.Vn[k,1])+N[2]*(1/2.*self.a[2]*self.Vn[k,2])+N[3]*(1/2.*self.a[3]*self.Vn[k,3])
        det = JJ[0,0]*JJ[1,1]*JJ[2,2]-JJ[0,0]*JJ[1,2]*JJ[2,1]-JJ[1,0]*JJ[0,1]*JJ[2,2]+JJ[1,0]*JJ[0,2]*JJ[2,1]+JJ[2,0]*JJ[0,1]*JJ[1,2]-JJ[2,0]*JJ[0,2]*JJ[1,1]
        return det
    
class SH3( SH4 ):
    """ Shell element 3 nodes
    """
    def ComputeGG(self):
        for i in xrange(3): 
            for j in xrange(3): self.gg[0,i,j] = -self.a[j]*self.V2[i,j]/2.     # 1st index direction, 2nd index node 
            for j in xrange(3): self.gg[1,i,j] =  self.a[j]*self.V1[i,j]/2.
    def ComputeTransLocalAll(self, TTW):
        base, base_ = 0, 0
        for i in xrange(3):                                 # loop over nodes of element  
            for j in xrange(3):          self.Trans[base_+j,base+j] = 1.
            if False:
#            if self.SixthDoF[i]:
                for j in xrange(2):
                    for jj in xrange(3): self.Trans[base_+3+j,base+3+jj] = TTW[i][j][jj] 
#   uhc             XX = dot(transpose(TTW[i]),TTW[i])
#   uhc             YY = dot(TTW[i],transpose(TTW[i]))
#                print i, TTW[i],'\n', XX, '\n', YY
            else:
                for j in xrange(2):      self.Trans[base_+3+j,base+3+j] = 1.
            base = base + self.DofN[i]
            base_= base_+ self.DofNini[i]            
#        for j in xrange(self.Trans.shape[0]):
#            for jj in xrange(self.Trans.shape[1]): sys.stdout.write('%6.2f'%(self.Trans[j,jj]))
#            sys.stdout.write('\n')
#        raise NameError ("Exit")
    def __init__(self, Label, SetLabel, InzList, MatName, NoList, ShellSec, StateV, NData, RCFlag):
        # Element.__init__((self, TypeVal,nNodVal,DofEVal,nFieVal, IntTVal,nIntVal,nIntLVal, DofTVal,DofNVal, dimVal,NLGeomIVal)
        Element.__init__(self,"SH3",3,15,3, 5,2,15, [set([1, 2, 3, 4, 5]),set([1, 2, 3, 4, 5]),set([1, 2, 3, 4, 5])], (5,5,5), 21, False) # five integration points over cross section height
        self.PlSt = True                                    # flag for plane stress  - should be true, otherwise inconsistencies in e.g. MISES
        self.Label = Label                                  # element number in input
        self.RotM= True                                     # Flag for elements with local coordinate system for materials
        self.TensStiff = False                              # flag for tension stiffening
        self.MatN = MatName                                 # name of material
        self.Set = SetLabel                                 # label of corresponding element set
#        self.ElemUpdate = True                              # element update might be required in case of NLGEOM
        self.ShellRCFlag = RCFlag
        self.InzList = [InzList[0], InzList[1], InzList[2]]
        self.a = array([ShellSec.Height,ShellSec.Height,ShellSec.Height]) # shell thickness
        nRe = len(ShellSec.Reinf)                           # number of reinforcement layers
        self.Geom = zeros( (2+nRe,5), dtype=double)
        self.Geom[0,0] = 1                                  # dummy for Area / Jacobi determinant used instead
        self.Geom[1,0] = 1                                  # dummy for height / thickness
        if False:
#        if RCFlag and nRe>0:
            if   self.nIntL==16: i1 = 1; i2 = 16
            elif self.nIntL==20: i1 = 4; i2 = 20
            else: raise NameError("ConFemElements::SH4.__ini__: integration order not implemented for RC")
            import ConFEM2D_Basics                             # to modify sampling points for numerical integration
            for j in xrange(nRe):
                self.Geom[2+j,0] = ShellSec.Reinf[j][0]     # reinforcement cross section
                self.Geom[2+j,1] = ShellSec.Reinf[j][1]     # " lever arm
                self.Geom[2+j,2] = ShellSec.Reinf[j][2]     # " effective reinforcement ratio for tension stiffening
                self.Geom[2+j,3] = ShellSec.Reinf[j][3]     # " parameter 0<beta<=1 for tension stiffening, beta=0 no tension stiffening
                self.Geom[2+j,4] = ShellSec.Reinf[j][4]     # " direction
                tt = 2.*ShellSec.Reinf[j][1]/ShellSec.Height # local t-coordinate for reinforcement
 #               i1 = 5
                ConFEM2D_Basics.SamplePointsRCShell[SetLabel, 4, i1, i2 + 4 * j + 0] =[-0.577350269189626, -0.577350269189626, tt] # every reinforcement layer / j gets consecutive indices in base plane
                ConFEM2D_Basics.SamplePointsRCShell[SetLabel, 4, i1, i2 + 4 * j + 1] =[-0.577350269189626, 0.577350269189626, tt] #
                ConFEM2D_Basics.SamplePointsRCShell[SetLabel, 4, i1, i2 + 4 * j + 2] =[0.577350269189626, -0.577350269189626, tt] #
                ConFEM2D_Basics.SamplePointsRCShell[SetLabel, 4, i1, i2 + 4 * j + 3] =[0.577350269189626, 0.577350269189626, tt] #
                ConFEM2D_Basics.SampleWeightRCShell[SetLabel, 4, i1, i2 + 4 * j + 0]= 2. * ShellSec.Reinf[j][0] / ShellSec.Height
                ConFEM2D_Basics.SampleWeightRCShell[SetLabel, 4, i1, i2 + 4 * j + 1]= 2. * ShellSec.Reinf[j][0] / ShellSec.Height
                ConFEM2D_Basics.SampleWeightRCShell[SetLabel, 4, i1, i2 + 4 * j + 2]= 2. * ShellSec.Reinf[j][0] / ShellSec.Height
                ConFEM2D_Basics.SampleWeightRCShell[SetLabel, 4, i1, i2 + 4 * j + 3]= 2. * ShellSec.Reinf[j][0] / ShellSec.Height
                self.nIntL = self.nIntL+4                   # four more integration points in base area
        self.Data = zeros((self.nIntL,NData), dtype=float)  # storage for element data
        self.DataP= zeros((self.nIntL,NData), dtype=float)  # storage for element data of previous step
        if StateV<>None:
            self.StateVar = zeros((self.nIntL,StateV), dtype=float)
            self.StateVarN= zeros((self.nIntL,StateV), dtype=float)
    def Lists1(self):                                       # indices for first integration points in base area
        if self.nInt==2:                                    # integration order 
            Lis = [0,5,10]                                  # indices for first integration points in base area, 4 Gaussian integration points over cross section height
        return Lis
    def Lists2(self, nRe, j):                               # integration point indices specific for base point
        if self.nInt==2: 
            Lis2, Lis3 = [0,1,2,3,4], []            
        return Lis2, Lis3                                   # RC not yet implemented
    def Ini2(self, NoList, MaList):
        i0 = FindIndexByLabel( NoList, self.InzList[0])     # find node index from node label
        i1 = FindIndexByLabel( NoList, self.InzList[1])
        i2 = FindIndexByLabel( NoList, self.InzList[2])     # find node index from node label
        self.Inzi = [ i0, i1, i2]                           # indices of nodes belonging to element
        NoList[i0].DofT = NoList[i0].DofT.union(self.DofT[0]) # attach element dof types to node dof types
        NoList[i1].DofT = NoList[i1].DofT.union(self.DofT[1])
        NoList[i2].DofT = NoList[i2].DofT.union(self.DofT[2])
        self.XX =      array([[NoList[i0].XCo, NoList[i1].XCo, NoList[i2].XCo],
                              [NoList[i0].YCo, NoList[i1].YCo, NoList[i2].YCo],
                              [NoList[i0].ZCo, NoList[i1].ZCo, NoList[i2].ZCo]])  # collect nodal coordinates in a compact form for later use
        self.EdgeDir = zeros((3),dtype=float)               # initialize direction of 1st edge for later use
        self.ComputeEdgeDir()                               # direction of 1st edge for later use
        nn = zeros((4,3), dtype = float)                    # 1st index for nodes, 2nd for directions
        nn[0,:] = self.CompNoNor( 0, 1, 2)                  # roughly unit normals to shell surface at nodes
        nn[1,:] = self.CompNoNor( 1, 2, 0)
        nn[2,:] = self.CompNoNor( 2, 0, 1)
        self.V1 = zeros((3,3),dtype=float)                  # initialize director triad, 1st index for directions, 2nd index for nodes
        self.V2 = zeros((3,3),dtype=float)
        self.Vn = zeros((3,3),dtype=float)
        self.Vg = zeros((3,3),dtype=float)                  # director as defined per node via input data, might not be the actually used director as is ruled in the following
#        self.VD = zeros((3,4),dtype=float)                  # director increment in time increment
        TTW = [None, None, None, None]                      # for temporal storage of transformation coefficients
        self.SixthDoF = [False, False, False]
        for i in xrange(3):                                 # loop over nodes of element  
            ii = self.Inzi[i]
            LL = sqrt(NoList[ii].XDi**2+NoList[ii].YDi**2+NoList[ii].ZDi**2) # length of directors
            self.Vg[0,i] = NoList[ii].XDi/LL                # values given with input data
            self.Vg[1,i] = NoList[ii].YDi/LL                # "
            self.Vg[2,i] = NoList[ii].ZDi/LL                # "
            self.V1[:,i], self.V2[:,i] = self.CompleteTriad(self.Vg[:,i])
            if False:
#            if dot(nn[i],self.Vg[:,i])<0.8:
                print "ConFemElements::SH4.__ini__::ControlGeom: unfavorable shell director element %s local node index %s"%(str(self.Label),str(i))
                self.SixthDoF[i] = True
                NoList[ii].DofT = NoList[ii].DofT.union(set([6])) # extend types of dof for this node
                self.DofT[i] = self.DofT[i].union(set([6])) # extend types of dof for this element
                self.DofE = self.DofE + 1               # adapt number of dofs of whole element
                DofN_ = list(self.DofN)                 # transform tuple into list to make it changeable
                DofN_[i] = 6                            # one more degree of freedom for local node i
                self.DofN = tuple(DofN_)                # transform back into tuple
                self.Rot = True                         # Element has formally to be rotated as a whole
                self.RotG= True                         # geometric stiffness has also to be rotated for nonlinear geometry
                self.Trans = zeros((self.DofEini, self.DofE), dtype=float)# initialize rotation / transformation matrix for element / coordinate transformation matrix, not quadratic anymore!
                self.ComputeTransLocal(i, nn[i], TTW)      # transformation matrix for modified director system
                self.Vn[0,i] = nn[i,0]                  # final director
                self.Vn[1,i] = nn[i,1]
                self.Vn[2,i] = nn[i,2]
            else:
                self.Vn[0,i] = self.Vg[0,i]                 # final director
                self.Vn[1,i] = self.Vg[1,i]
                self.Vn[2,i] = self.Vg[2,i]
        if self.Rot: self.ComputeTransLocalAll(TTW)     # ??? build rotation / transformation matrix for element / coordinate transformation matrix, not quadratic anymore!
        self.XX0 = copy(self.XX)                        # retain initial values for NLGEOM
        self.DofI = array([[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1]],dtype=int)
        self.gg = zeros((2,3,3), dtype=float)           # initialization of scaled rotation axes; 1st index for rotation dofs, 2nd index for directions of rotation axes, 3rd for nodes 
        self.ComputeGG()                                # scaled axes for rotational degrees of freedom
        AA = 0.
        for i in xrange(15):                            # volume by numerical integration for characteristic length  
            r = SamplePoints[5,1,i][0]
            s = SamplePoints[5,1,i][1]
            f = self.JacoD(r,s,0)*SampleWeight[5,1,i]
            AA = AA + f
        self.Lch_ = sqrt(AA/(0.25*(self.a[0]+self.a[1]+self.a[2]))) #/self.nInt   # characteristic length -> side length of square of same area of shell area
        # --> common method with sh4
        if MaList[self.MatN].RType ==2:                 # find scaling factor for band width regularization
            x = MaList[self.MatN].bw/self.Lch_          # ratio of crack band width to characteristic length
            CrX, CrY = MaList[self.MatN].CrX, MaList[self.MatN].CrY # support points for tensile softening scaling factor 
            i = bisect_left(CrX, x) 
            if i>0 and i<(MaList[self.MatN].CrBwN+1): 
                self.CrBwS = CrY[i-1] + (x-CrX[i-1])/(CrX[i]-CrX[i-1])*(CrY[i]-CrY[i-1]) # scaling factor by linear interpolation
            else:
                print 'ZZZ', AA, self.Lch_,x,'\n', CrX, '\n', CrY, i, MaList[self.MatN].CrBwN  
                raise NameError("ConFemElem:SH3.Ini2: RType 2 - element char length exceeds scaling factor interpolation")

    def Basics(self, r, s, t):
        N =  array([ 1. -r-s, r, s])
        br = array([-1., 1., 0.])
        bs = array([-1., 0., 1.])
        JJ = zeros((3,3),dtype=float)
        # following may be simplified due to br[2]=bs[1]=0 and so on
        for k in xrange(3): JJ[k,0]=br[0]*(self.XX[k,0]+t/2.*self.a[0]*self.Vn[k,0])+br[1]*(self.XX[k,1]+t/2.*self.a[1]*self.Vn[k,1])+br[2]*(self.XX[k,2]+t/2.*self.a[2]*self.Vn[k,2]) #+br[3]*(self.XX[k,3]+t/2.*self.a[3]*self.Vn[k,3])
        for k in xrange(3): JJ[k,1]=bs[0]*(self.XX[k,0]+t/2.*self.a[0]*self.Vn[k,0])+bs[1]*(self.XX[k,1]+t/2.*self.a[1]*self.Vn[k,1])+bs[2]*(self.XX[k,2]+t/2.*self.a[2]*self.Vn[k,2]) #+bs[3]*(self.XX[k,3]+t/2.*self.a[3]*self.Vn[k,3])
        for k in xrange(3): JJ[k,2]=N[0]*(1/2.*self.a[0]*self.Vn[k,0])+N[1]*(1/2.*self.a[1]*self.Vn[k,1])+N[2]*(1/2.*self.a[2]*self.Vn[k,2]) #+N[3]*(1/2.*self.a[3]*self.Vn[k,3])
        JI = inv(JJ)
        ll=sqrt(JJ[0,2]**2+JJ[1,2]**2+JJ[2,2]**2)   
        vv = array([[0.,0.,JJ[0,2]/ll],[0.,0.,JJ[1,2]/ll],[0.,0.,JJ[2,2]/ll]]) # normal of local coordinate system, 3RD COLUMN
        Loc = False
        if Loc:                                         # local right handed coordinate system with 1st direction / column aligned to element edge
            x0 = self.EdgeDir[1]*vv[2,2]-self.EdgeDir[2]*vv[1,2]
            x1 = self.EdgeDir[2]*vv[0,2]-self.EdgeDir[0]*vv[2,2]
            x2 = self.EdgeDir[0]*vv[1,2]-self.EdgeDir[1]*vv[0,2]
            xx = sqrt(x0**2+x1**2+x2**2)
            vv[0,1] = -x0/xx                            # 2ND COLUMN, approx perp. to element edge, reversed in sign to preserve right handedness
            vv[1,1] = -x1/xx
            vv[2,1] = -x2/xx 
            x0 = vv[1,2]*vv[2,1]-vv[2,2]*vv[1,1]
            x1 = vv[2,2]*vv[0,1]-vv[0,2]*vv[2,1]
            x2 = vv[0,2]*vv[1,1]-vv[1,2]*vv[0,1]
            xx = sqrt(x0**2+x1**2+x2**2)
            vv[0,0] = -x0/xx                            # 1ST COLUMN, approx aligned to element edge, sign reversal of 2nd column is implicitly corrected
            vv[1,0] = -x1/xx
            vv[2,0] = -x2/xx 
        else:
            if abs(vv[1,2])<0.99:                       # local coordinate system V_1 from cross product of V_n and e_y ( V_1 in e_x - e_z plane) if V_n is not to much aligned to e_y  
                ll = sqrt(vv[2,2]**2+vv[0,2]**2)        # length of V_1
                vv[0,0] = vv[2,2]/ll                    # V_1[0] normalized;  V1[1] = 0
                vv[2,0] =-vv[0,2]/ll                    # V_1[2] normalized

                vv[0,1] = vv[1,2]*vv[2,0]               # as V_n and V_1 are orthogonal and both have unit length V_2 also should have unit length
                vv[1,1] = vv[2,2]*vv[0,0]-vv[0,2]*vv[2,0]
                vv[2,1] =-vv[1,2]*vv[0,0]
            else:                                       # local coordinate system V_1 from cross product of V_n and e_x ( V_1 in e_y - e_z plane)
                ll = sqrt(vv[2,2]**2+vv[1,2]**2)        # length of V_1
                vv[1,0] =-vv[2,2]/ll                    # V_1[0] normalized;  V1[0] = 0
                vv[2,0] = vv[1,2]/ll                    # V_1[2] normalized

                vv[0,1] = vv[1,2]*vv[2,0]-vv[2,2]*vv[1,0] # as V_n and V_1 are orthogonal and both have unit length V_2 also should have unit length
                vv[1,1] =-vv[0,2]*vv[2,0]
                vv[2,1] = vv[0,2]*vv[1,0]
        return N, br, bs, JJ, JI, vv

    def FormB(self, r, s, t, NLg):
        N, br, bs, JJ, JI, vv = self.Basics( r, s, t)
        det = JJ[0,0]*JJ[1,1]*JJ[2,2]-JJ[0,0]*JJ[1,2]*JJ[2,1]-JJ[1,0]*JJ[0,1]*JJ[2,2]+JJ[1,0]*JJ[0,2]*JJ[2,1]+JJ[2,0]*JJ[0,1]*JJ[1,2]-JJ[2,0]*JJ[0,2]*JJ[1,1]
        HH = zeros((3,2,3),dtype=float)
        for k in xrange(3):
            HH[0,0,k]=JJ[0,0]*self.gg[0,0,k]+JJ[1,0]*self.gg[0,1,k]+JJ[2,0]*self.gg[0,2,k]
            HH[0,1,k]=JJ[0,0]*self.gg[1,0,k]+JJ[1,0]*self.gg[1,1,k]+JJ[2,0]*self.gg[1,2,k]
            HH[1,0,k]=JJ[0,1]*self.gg[0,0,k]+JJ[1,1]*self.gg[0,1,k]+JJ[2,1]*self.gg[0,2,k]
            HH[1,1,k]=JJ[0,1]*self.gg[1,0,k]+JJ[1,1]*self.gg[1,1,k]+JJ[2,1]*self.gg[1,2,k]
            HH[2,0,k]=JJ[0,2]*self.gg[0,0,k]+JJ[1,2]*self.gg[0,1,k]+JJ[2,2]*self.gg[0,2,k]
            HH[2,1,k]=JJ[0,2]*self.gg[1,0,k]+JJ[1,2]*self.gg[1,1,k]+JJ[2,2]*self.gg[1,2,k]
        BB = zeros((6,15),dtype=float)
        BB[0,0] =-JJ[0,0]
        BB[0,1] =-JJ[1,0]
        BB[0,2] =-JJ[2,0]
        BB[0,3] =-t*HH[0,0,0]
        BB[0,4] =-t*HH[0,1,0]
        BB[0,5] = JJ[0,0]
        BB[0,6] = JJ[1,0]
        BB[0,7] = JJ[2,0]
        BB[0,8] = t*HH[0,0,1]
        BB[0,9] = t*HH[0,1,1]
        BB[0,10]= 0
        BB[0,11]= 0
        BB[0,12]= 0
        BB[0,13]= 0
        BB[0,14]= 0
        BB[1,0] =-JJ[0,1]
        BB[1,1] =-JJ[1,1]
        BB[1,2] =-JJ[2,1]
        BB[1,3] =-t*HH[1,0,0]
        BB[1,4] =-t*HH[1,1,0]
        BB[1,5] = 0
        BB[1,6] = 0
        BB[1,7] = 0
        BB[1,8] = 0
        BB[1,9] = 0
        BB[1,10]= JJ[0,1]
        BB[1,11]= JJ[1,1]
        BB[1,12]= JJ[2,1]
        BB[1,13]= t*HH[1,0,2]
        BB[1,14]= t*HH[1,1,2]
        BB[2,0] = 0
        BB[2,1] = 0
        BB[2,2] = 0
        BB[2,3] = N[0]*HH[2,0,0]
        BB[2,4] = N[0]*HH[2,1,0]
        BB[2,5] = 0
        BB[2,6] = 0
        BB[2,7] = 0
        BB[2,8] = N[1]*HH[2,0,1]
        BB[2,9] = N[1]*HH[2,1,1]
        BB[2,10]= 0
        BB[2,11]= 0
        BB[2,12]= 0
        BB[2,13]= N[2]*HH[2,0,2]
        BB[2,14]= N[2]*HH[2,1,2]
        BB[5,0] =-JJ[0,0]-JJ[0,1]
        BB[5,1] =-JJ[1,0]-JJ[1,1]
        BB[5,2] =-JJ[2,0]-JJ[2,1]
        BB[5,3] = t*(-HH[0,0,0]-HH[1,0,0])
        BB[5,4] = t*(-HH[0,1,0]-HH[1,1,0])
        BB[5,5] = JJ[0,1]
        BB[5,6] = JJ[1,1]
        BB[5,7] = JJ[2,1]
        BB[5,8] = t*HH[1,0,1]
        BB[5,9] = t*HH[1,1,1]
        BB[5,10]= JJ[0,0]
        BB[5,11]= JJ[1,0]
        BB[5,12]= JJ[2,0]
        BB[5,13]= t*HH[0,0,2]
        BB[5,14]= t*HH[0,1,2]
        BB[3,0] =-(1.-r)*JJ[0,2]-r*JJ[0,2]+s*(JJ[0,2]-JJ[0,2])
        BB[3,1] =-(1.-r)*JJ[1,2]-r*JJ[1,2]+s*(JJ[1,2]-JJ[1,2])
        BB[3,2] =-(1.-r)*JJ[2,2]-r*JJ[2,2]+s*(JJ[2,2]-JJ[2,2])
        BB[3,3] = (1.-r)*(0.5*HH[1,0,0]-t*HH[2,0,0])-r*t*HH[2,0,0]+s*(t*HH[2,0,0]+0.5*HH[1,0,0]-t*HH[2,0,0])
        BB[3,4] = (1.-r)*(0.5*HH[1,1,0]-t*HH[2,1,0])-r*t*HH[2,1,0]+s*(t*HH[2,1,0]+0.5*HH[1,1,0]-t*HH[2,1,0])
        BB[3,5] = 0
        BB[3,6] = 0
        BB[3,7] = 0
        BB[3,8] = 0.5*r*HH[1,0,1]-0.5*s*HH[1,0,1]
        BB[3,9] = 0.5*r*HH[1,1,1]-0.5*s*HH[1,1,1]
        BB[3,10]= (1-r)*JJ[0,2]+r*JJ[0,2]+s*(-JJ[0,2]+JJ[0,2])
        BB[3,11]= (1-r)*JJ[1,2]+r*JJ[1,2]+s*(-JJ[1,2]+JJ[1,2])
        BB[3,12]= (1-r)*JJ[2,2]+r*JJ[2,2]+s*(-JJ[2,2]+JJ[2,2])
        BB[3,13]= (1-r)*(0.5*HH[1,0,2]+t*HH[2,0,2])-r*(-0.5*HH[1,0,2]-t*HH[2,0,2])+s*(-0.5*HH[1,0,2]-t*HH[2,0,2]+0.5*HH[1,0,2]+t*HH[2,0,2])
        BB[3,14]= (1-r)*(0.5*HH[1,1,2]+t*HH[2,1,2])-r*(-0.5*HH[1,1,2]-t*HH[2,1,2])+s*(-0.5*HH[1,1,2]-t*HH[2,1,2]+0.5*HH[1,1,2]+t*HH[2,1,2])
        BB[4,0] =-r*(-JJ[0,2]+JJ[0,2])-(1-s)*JJ[0,2]-s*JJ[0,2]
        BB[4,1] =-r*(-JJ[1,2]+JJ[1,2])-(1-s)*JJ[1,2]-s*JJ[1,2]
        BB[4,2] =-r*(-JJ[2,2]+JJ[2,2])-(1-s)*JJ[2,2]-s*JJ[2,2]
        BB[4,3] =-r*(-t*HH[2,0,0]-0.5*HH[0,0,0]+t*HH[2,0,0])+(1-s)*(0.5*HH[0,0,0]-t*HH[2,0,0])-s*t*HH[2,0,0]
        BB[4,4] =-r*(-t*HH[2,1,0]-0.5*HH[0,1,0]+t*HH[2,1,0])+(1-s)*(0.5*HH[0,1,0]-t*HH[2,1,0])-s*t*HH[2,1,0]
        BB[4,5] =-r*(JJ[0,2]-JJ[0,2])+(1-s)*JJ[0,2]+s*JJ[0,2]
        BB[4,6] =-r*(JJ[1,2]-JJ[1,2])+(1-s)*JJ[1,2]+s*JJ[1,2]
        BB[4,7] =-r*(JJ[2,2]-JJ[2,2])+(1-s)*JJ[2,2]+s*JJ[2,2]
        BB[4,8] =-r*(0.5*HH[0,0,1]+t*HH[2,0,1]-0.5*HH[0,0,1]-t*HH[2,0,1])+(1-s)*(0.5*HH[0,0,1]+t*HH[2,0,1])+s*(0.5*HH[0,0,1]+t*HH[2,0,1])
        BB[4,9] =-r*(0.5*HH[0,1,1]+t*HH[2,1,1]-0.5*HH[0,1,1]-t*HH[2,1,1])+(1-s)*(0.5*HH[0,1,1]+t*HH[2,1,1])+s*(0.5*HH[0,1,1]+t*HH[2,1,1])
        BB[4,10]= 0
        BB[4,11]= 0
        BB[4,12]= 0
        BB[4,13]=-0.5*r*HH[0,0,2]+0.5*s*HH[0,0,2]
        BB[4,14]=-0.5*r*HH[0,1,2]+0.5*s*HH[0,1,2]

        td=array([[JI[0,0]*vv[0,0]+JI[0,1]*vv[1,0]+JI[0,2]*vv[2,0],JI[0,0]*vv[0,1]+JI[0,1]*vv[1,1]+JI[0,2]*vv[2,1],JI[0,0]*vv[0,2]+JI[0,1]*vv[1,2]+JI[0,2]*vv[2,2]],
                  [JI[1,0]*vv[0,0]+JI[1,1]*vv[1,0]+JI[1,2]*vv[2,0],JI[1,0]*vv[0,1]+JI[1,1]*vv[1,1]+JI[1,2]*vv[2,1],JI[1,0]*vv[0,2]+JI[1,1]*vv[1,2]+JI[1,2]*vv[2,2]],
                  [JI[2,0]*vv[0,0]+JI[2,1]*vv[1,0]+JI[2,2]*vv[2,0],JI[2,0]*vv[0,1]+JI[2,1]*vv[1,1]+JI[2,2]*vv[2,1],JI[2,0]*vv[0,2]+JI[2,1]*vv[1,2]+JI[2,2]*vv[2,2]]])
        TD=array([[td[0,0]**2,         td[1,0]**2,       td[2,0]**2,     td[1,0]*td[2,0],                td[0,0]*td[2,0],                td[0,0]*td[1,0]],
                  [td[0,1]**2,         td[1,1]**2,       td[2,1]**2,     td[1,1]*td[2,1],                td[0,1]*td[2,1],                td[0,1]*td[1,1]],
                  [td[0,2]**2,         td[1,2]**2,       td[2,2]**2,     td[1,2]*td[2,2],                td[0,2]*td[2,2],                td[0,2]*td[1,2]],
                  [2*td[0,1]*td[0,2],2*td[1,1]*td[1,2],2*td[2,1]*td[2,2],td[1,1]*td[2,2]+td[2,1]*td[1,2],td[0,1]*td[2,2]+td[0,2]*td[2,1],td[0,1]*td[1,2]+td[1,1]*td[0,2]],
                  [2*td[0,0]*td[0,2],2*td[1,0]*td[1,2],2*td[2,0]*td[2,2],td[1,0]*td[2,2]+td[2,0]*td[1,2],td[0,0]*td[2,2]+td[0,2]*td[2,0],td[0,0]*td[1,2]+td[1,0]*td[0,2]],
                  [2*td[0,0]*td[0,1],2*td[1,0]*td[1,1],2*td[2,0]*td[2,1],td[1,0]*td[2,1]+td[2,0]*td[1,1],td[0,0]*td[2,1]+td[0,1]*td[2,0],td[0,0]*td[1,1]+td[1,0]*td[0,1]]])

        return BB, det, TD
    def FormT(self, r, s, t):                               # interpolation on temperature - currently not used
        T = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return T
    def FormX(self, r, s, t):                               # interpolation on geometry
        X = array([ 1.-r-s, r, s])
        return X
    def JacoD(self, r, s, t):
        N =  array([ 1. -r-s, r, s])
        br = array([-1., 1., 0.])
        bs = array([-1., 0., 1.])
        JJ = zeros((3,3),dtype=float)
        # following may be simplified due to br[2]=bs[1]=0 and so on
        for k in xrange(3): JJ[k,0]=br[0]*(self.XX[k,0]+t/2.*self.a[0]*self.Vn[k,0])+br[1]*(self.XX[k,1]+t/2.*self.a[1]*self.Vn[k,1])+br[2]*(self.XX[k,2]+t/2.*self.a[2]*self.Vn[k,2])#+br[3]*(self.XX[k,3]+t/2.*self.a[3]*self.Vn[k,3])
        for k in xrange(3): JJ[k,1]=bs[0]*(self.XX[k,0]+t/2.*self.a[0]*self.Vn[k,0])+bs[1]*(self.XX[k,1]+t/2.*self.a[1]*self.Vn[k,1])+bs[2]*(self.XX[k,2]+t/2.*self.a[2]*self.Vn[k,2])#+bs[3]*(self.XX[k,3]+t/2.*self.a[3]*self.Vn[k,3])
        for k in xrange(3): JJ[k,2]=N[0]*(1/2.*self.a[0]*self.Vn[k,0])+N[1]*(1/2.*self.a[1]*self.Vn[k,1])+N[2]*(1/2.*self.a[2]*self.Vn[k,2])#+N[3]*(1/2.*self.a[3]*self.Vn[k,3])
        det = JJ[0,0]*JJ[1,1]*JJ[2,2]-JJ[0,0]*JJ[1,2]*JJ[2,1]-JJ[1,0]*JJ[0,1]*JJ[2,2]+JJ[1,0]*JJ[0,2]*JJ[2,1]+JJ[2,0]*JJ[0,1]*JJ[1,2]-JJ[2,0]*JJ[0,2]*JJ[1,1]
        return det

