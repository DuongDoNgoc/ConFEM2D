# ConFem2D_Basics
import sys
import numpy as np
from math import pi, sqrt
try:
    import libElemC
    reload(libElemC)
    from libElemC import *
    ConFemElemCFlag = True    
except ImportError:
    ConFemElemCFlag = False
import logging

SamplePoints={}
SampleWeight={}
# Sample points and weighting factors for Gauss Quadrature 1D
SamplePoints[0,0,0]=[0.,0.,0.]
SampleWeight[0,0,0]= 2.
SamplePoints[0,1,0]=[-0.577350269189626, 0., 0.]
SamplePoints[0,1,1]=[ 0.577350269189626, 0., 0.]
SampleWeight[0,1,0]= 1.
SampleWeight[0,1,1]= 1.
SamplePoints[0,2,0]=[-0.774596669241483, 0., 0.]
SamplePoints[0,2,1]=[ 0., 0., 0.]
SamplePoints[0,2,2]=[ 0.774596669241483, 0., 0.]
SampleWeight[0,2,0]= 0.555555555555556
SampleWeight[0,2,1]= 0.888888888888888
SampleWeight[0,2,2]= 0.555555555555556
# Sample points and weighting factors for Gauss Quadrature 2D
SamplePoints[1,0,0]=[0.,0.,0.]
SampleWeight[1,0,0]= 4. # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SamplePoints[1,1,0]=[-0.577350269189626,-0.577350269189626, 0.]
SamplePoints[1,1,1]=[-0.577350269189626, 0.577350269189626, 0.]
SamplePoints[1,1,2]=[ 0.577350269189626,-0.577350269189626, 0.]
SamplePoints[1,1,3]=[ 0.577350269189626, 0.577350269189626, 0.]
SampleWeight[1,1,0]= 1.
SampleWeight[1,1,1]= 1.
SampleWeight[1,1,2]= 1.
SampleWeight[1,1,3]= 1.
# Sample points and weighting factors for Gauss Quadrature 3D
SamplePoints[2,0,0]=[0.,0.,0.]
SampleWeight[2,0,0]= 8. # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SamplePoints[2,1,0]=[-0.577350269189626,-0.577350269189626, -0.577350269189626]
SamplePoints[2,1,1]=[-0.577350269189626,-0.577350269189626,  0.577350269189626]
SamplePoints[2,1,2]=[-0.577350269189626, 0.577350269189626, -0.577350269189626]
SamplePoints[2,1,3]=[-0.577350269189626, 0.577350269189626,  0.577350269189626]
SamplePoints[2,1,4]=[ 0.577350269189626,-0.577350269189626, -0.577350269189626]
SamplePoints[2,1,5]=[ 0.577350269189626,-0.577350269189626,  0.577350269189626]
SamplePoints[2,1,6]=[ 0.577350269189626, 0.577350269189626, -0.577350269189626]
SamplePoints[2,1,7]=[ 0.577350269189626, 0.577350269189626,  0.577350269189626]
SampleWeight[2,1,0]= 1.
SampleWeight[2,1,1]= 1.
SampleWeight[2,1,2]= 1.
SampleWeight[2,1,3]= 1.
SampleWeight[2,1,4]= 1.
SampleWeight[2,1,5]= 1.
SampleWeight[2,1,6]= 1.
SampleWeight[2,1,7]= 1.
# Sample points and weighting factors for triangular -elements, vgl. Zienk I, Table 8.2
SamplePoints[3,0,0]=[ 0.333333333333333, 0.333333333333333, 0.333333333333333]
SampleWeight[3,0,0]= 1.
SamplePoints[3,1,0]=[ 0.5,               0.5,               0.0]
SamplePoints[3,1,1]=[ 0.0,               0.5,               0.5]
SamplePoints[3,1,2]=[ 0.5,               0.0,               0.5]
SampleWeight[3,1,0]= 0.333333333333333
SampleWeight[3,1,1]= 0.333333333333333
SampleWeight[3,1,2]= 0.333333333333333
SamplePoints[3,2,0]=[ 0.333333333333333, 0.333333333333333, 0.333333333333333]
SamplePoints[3,2,1]=[ 0.2,               0.6,               0.2]
SamplePoints[3,2,2]=[ 0.2,               0.2,               0.6]
SamplePoints[3,2,3]=[ 0.6,               0.2,               0.2]
SampleWeight[3,2,0]=-0.562500000000000
SampleWeight[3,2,1]= 0.520833333333333
SampleWeight[3,2,2]= 0.520833333333333
SampleWeight[3,2,3]= 0.520833333333333
# specht, int. j. num. meth. eng., 1988, 705
SamplePoints[3,3,0]=[ 0.666666666666667, 0.166666666666667, 0.166666666666667]
SamplePoints[3,3,1]=[ 0.166666666666667, 0.666666666666667, 0.166666666666667]
SamplePoints[3,3,2]=[ 0.166666666666667, 0.166666666666667, 0.666666666666667]
SampleWeight[3,3,0]= 0.333333333333333
SampleWeight[3,3,1]= 0.333333333333333
SampleWeight[3,3,2]= 0.333333333333333
# Sample points and weighting factors for shells SH3
SamplePoints[4,0,0] =[-0.577350269189626,-0.577350269189626, 0.]
SamplePoints[4,0,1] =[-0.577350269189626, 0.577350269189626, 0.]
SamplePoints[4,0,2] =[ 0.577350269189626,-0.577350269189626, 0.] 
SamplePoints[4,0,3] =[ 0.577350269189626, 0.577350269189626, 0.] 
SampleWeight[4,0,0] = 2.
SampleWeight[4,0,1] = 2. 
SampleWeight[4,0,2] = 2.
SampleWeight[4,0,3] = 2.
SamplePoints[4,1,0] =[-0.577350269189626,-0.577350269189626, -0.861136311594053] #-0.577350269189626]
SamplePoints[4,1,1] =[-0.577350269189626,-0.577350269189626, -0.339981043584856] #0.577350269189626]
SamplePoints[4,1,2] =[-0.577350269189626,-0.577350269189626,  0.339981043584856] #-0.577350269189626]
SamplePoints[4,1,3] =[-0.577350269189626,-0.577350269189626,  0.861136311594053] #0.577350269189626]
SamplePoints[4,1,4] =[-0.577350269189626, 0.577350269189626, -0.861136311594053] #-0.577350269189626]
SamplePoints[4,1,5] =[-0.577350269189626, 0.577350269189626, -0.339981043584856] #0.577350269189626]
SamplePoints[4,1,6] =[-0.577350269189626, 0.577350269189626,  0.339981043584856] #-0.577350269189626]
SamplePoints[4,1,7] =[-0.577350269189626, 0.577350269189626,  0.861136311594053] #0.577350269189626]
SamplePoints[4,1,8] =[ 0.577350269189626,-0.577350269189626, -0.861136311594053] #-0.577350269189626]
SamplePoints[4,1,9] =[ 0.577350269189626,-0.577350269189626, -0.339981043584856] #0.577350269189626]
SamplePoints[4,1,10]=[ 0.577350269189626,-0.577350269189626,  0.339981043584856] #-0.577350269189626]
SamplePoints[4,1,11]=[ 0.577350269189626,-0.577350269189626,  0.861136311594053] #0.577350269189626]
SamplePoints[4,1,12]=[ 0.577350269189626, 0.577350269189626, -0.861136311594053] #-0.577350269189626]
SamplePoints[4,1,13]=[ 0.577350269189626, 0.577350269189626, -0.339981043584856] #0.577350269189626]
SamplePoints[4,1,14]=[ 0.577350269189626, 0.577350269189626,  0.339981043584856] #-0.577350269189626]
SamplePoints[4,1,15]=[ 0.577350269189626, 0.577350269189626,  0.861136311594053] #0.577350269189626]
SampleWeight[4,1,0] = 0.347854845137454
SampleWeight[4,1,1] = 0.652145154862546 
SampleWeight[4,1,2] = 0.652145154862546
SampleWeight[4,1,3] = 0.347854845137454
SampleWeight[4,1,4] = 0.347854845137454
SampleWeight[4,1,5] = 0.652145154862546
SampleWeight[4,1,6] = 0.652145154862546
SampleWeight[4,1,7] = 0.347854845137454
SampleWeight[4,1,8] = 0.347854845137454
SampleWeight[4,1,9] = 0.652145154862546
SampleWeight[4,1,10]= 0.652145154862546
SampleWeight[4,1,11]= 0.347854845137454
SampleWeight[4,1,12]= 0.347854845137454
SampleWeight[4,1,13]= 0.652145154862546
SampleWeight[4,1,14]= 0.652145154862546
SampleWeight[4,1,15]= 0.347854845137454
SamplePoints[4,4,0] =[-0.577350269189626,-0.577350269189626, -0.9061798459386640]
SamplePoints[4,4,1] =[-0.577350269189626,-0.577350269189626, -0.5384693101056831]
SamplePoints[4,4,2] =[-0.577350269189626,-0.577350269189626,  0.0]
SamplePoints[4,4,3] =[-0.577350269189626,-0.577350269189626,  0.5384693101056831]
SamplePoints[4,4,4] =[-0.577350269189626,-0.577350269189626,  0.9061798459386640]
SamplePoints[4,4,5] =[-0.577350269189626, 0.577350269189626, -0.9061798459386640]
SamplePoints[4,4,6] =[-0.577350269189626, 0.577350269189626, -0.5384693101056831]
SamplePoints[4,4,7] =[-0.577350269189626, 0.577350269189626,  0.0]
SamplePoints[4,4,8] =[-0.577350269189626, 0.577350269189626,  0.5384693101056831]
SamplePoints[4,4,9] =[-0.577350269189626, 0.577350269189626,  0.9061798459386640]
SamplePoints[4,4,10]=[ 0.577350269189626,-0.577350269189626, -0.9061798459386640]
SamplePoints[4,4,11]=[ 0.577350269189626,-0.577350269189626, -0.5384693101056831]
SamplePoints[4,4,12]=[ 0.577350269189626,-0.577350269189626,  0.0]
SamplePoints[4,4,13]=[ 0.577350269189626,-0.577350269189626,  0.5384693101056831]
SamplePoints[4,4,14]=[ 0.577350269189626,-0.577350269189626,  0.9061798459386640]
SamplePoints[4,4,15]=[ 0.577350269189626, 0.577350269189626, -0.9061798459386640]
SamplePoints[4,4,16]=[ 0.577350269189626, 0.577350269189626, -0.5384693101056831]
SamplePoints[4,4,17]=[ 0.577350269189626, 0.577350269189626,  0.0]
SamplePoints[4,4,18]=[ 0.577350269189626, 0.577350269189626,  0.5384693101056831]
SamplePoints[4,4,19]=[ 0.577350269189626, 0.577350269189626,  0.9061798459386640]
SampleWeight[4,4,0] = 0.2369268850561891
SampleWeight[4,4,1] = 0.4786286704993665 
SampleWeight[4,4,2] = 0.5688888888888889
SampleWeight[4,4,3] = 0.4786286704993665
SampleWeight[4,4,4] = 0.2369268850561891
SampleWeight[4,4,5] = 0.2369268850561891
SampleWeight[4,4,6] = 0.4786286704993665
SampleWeight[4,4,7] = 0.5688888888888889
SampleWeight[4,4,8] = 0.4786286704993665
SampleWeight[4,4,9] = 0.2369268850561891
SampleWeight[4,4,10]= 0.2369268850561891
SampleWeight[4,4,11]= 0.4786286704993665
SampleWeight[4,4,12]= 0.5688888888888889
SampleWeight[4,4,13]= 0.4786286704993665
SampleWeight[4,4,14]= 0.2369268850561891
SampleWeight[4,4,15]= 0.2369268850561891
SampleWeight[4,4,16]= 0.4786286704993665
SampleWeight[4,4,17]= 0.5688888888888889
SampleWeight[4,4,18]= 0.4786286704993665
SampleWeight[4,4,19]= 0.2369268850561891
# Sample points and weighting factors for shells SH3
SamplePoints[5,1,0] =[.16666666666666666667, .16666666666666666667, -.906179845938664]
SamplePoints[5,1,1] =[.16666666666666666667, .16666666666666666667, -.538469310105683]
SamplePoints[5,1,2] =[.16666666666666666667, .16666666666666666667, 0.]
SamplePoints[5,1,3] =[.16666666666666666667, .16666666666666666667,  .538469310105683]
SamplePoints[5,1,4] =[.16666666666666666667, .16666666666666666667,  .906179845938664]
SamplePoints[5,1,5] =[.66666666666666666667, .16666666666666666667, -.906179845938664]
SamplePoints[5,1,6] =[.66666666666666666667, .16666666666666666667, -.538469310105683]
SamplePoints[5,1,7] =[.66666666666666666667, .16666666666666666667, 0.]
SamplePoints[5,1,8] =[.66666666666666666667, .16666666666666666667,  .538469310105683]
SamplePoints[5,1,9] =[.66666666666666666667, .16666666666666666667,  .906179845938664]
SamplePoints[5,1,10]=[.16666666666666666667, .66666666666666666667, -.906179845938664]
SamplePoints[5,1,11]=[.16666666666666666667, .66666666666666666667, -.538469310105683]
SamplePoints[5,1,12]=[.16666666666666666667, .66666666666666666667, 0.]
SamplePoints[5,1,13]=[.16666666666666666667, .66666666666666666667,  .538469310105683]
SamplePoints[5,1,14]=[.16666666666666666667, .66666666666666666667,  .906179845938664]
# divided by 8 to make unit volume to 1/2, see SH4:: 
SampleWeight[5,1,0] = 0.31590251340825200000/8.
SampleWeight[5,1,1] = 0.63817156066582133333/8.
SampleWeight[5,1,2] = 0.75851851851851866666/8.
SampleWeight[5,1,3] = 0.63817156066582133333/8.
SampleWeight[5,1,4] = 0.31590251340825200000/8.
SampleWeight[5,1,5] = 0.31590251340825200000/8.
SampleWeight[5,1,6] = 0.63817156066582133333/8.
SampleWeight[5,1,7] = 0.75851851851851866666/8.
SampleWeight[5,1,8] = 0.63817156066582133333/8.
SampleWeight[5,1,9] = 0.31590251340825200000/8.
SampleWeight[5,1,10]= 0.31590251340825200000/8.
SampleWeight[5,1,11]= 0.63817156066582133333/8.
SampleWeight[5,1,12]= 0.75851851851851866666/8.
SampleWeight[5,1,13]= 0.63817156066582133333/8.
SampleWeight[5,1,14]= 0.31590251340825200000/8.
# for reinforcement of shells, see SH4.__init__
SamplePointsRCShell={}
SampleWeightRCShell={}
# Microplane
I21Points=np.array([[1.,             0.,             0.            ],
                 [0.,             1.,             0.            ],
                 [0.,             0.,             1.            ],
                 [0.707106781187, 0.707106781187, 0.            ],
                 [0.707106781187,-0.707106781187, 0.            ],
                 [0.707106781187, 0.            , 0.707106781187],
                 [0.707106781187, 0.            ,-0.707106781187],
                 [0.            , 0.707106781187, 0.707106781187],
                 [0.            , 0.707106781187,-0.707106781187],
                 [0.387907304067, 0.387907304067, 0.836095596749],
                 [0.387907304067, 0.387907304067,-0.836095596749],
                 [0.387907304067,-0.387907304067, 0.836095596749],
                 [0.387907304067,-0.387907304067,-0.836095596749],
                 [0.387907304067, 0.836095596749, 0.387907304067],
                 [0.387907304067, 0.836095596749,-0.387907304067],
                 [0.387907304067,-0.836095596749, 0.387907304067],
                 [0.387907304067,-0.836095596749,-0.387907304067],
                 [0.836095596749, 0.387907304067, 0.387907304067],
                 [0.836095596749, 0.387907304067,-0.387907304067],
                 [0.836095596749,-0.387907304067, 0.387907304067],
                 [0.836095596749,-0.387907304067,-0.387907304067]])
I21Weights=np.array([0.0265214244093,0.0265214244093,0.0265214244093,
                  0.0199301476312,0.0199301476312,0.0199301476312,0.0199301476312,0.0199301476312,0.0199301476312,
                  0.0250712367487,0.0250712367487,0.0250712367487,0.0250712367487,0.0250712367487,0.0250712367487,0.0250712367487,0.0250712367487,0.0250712367487,0.0250712367487,0.0250712367487,0.0250712367487])

ResultTypes = {}                                        # dictionary for each element type
ZeroD = 1.e-9                                           # Smallest float for division by Zero
StrNum0 = 3.*sqrt(3.)/2.                                # strange number used by ConFemMat::IsoDamage

def FindIndexByLabel(NodeList, Key):                    # find index from label
    if len(NodeList)==0: raise NameError ("no item defined")
    for i in range( len(NodeList)):                     # loop over all nodes
        if NodeList[i].Label == Key: return i
    if i==(len(NodeList)-1):         return -1          # no node found

def PrinC( xx, yy, xy):                                 # calculation of principal values
    if ZeroD < abs(xy):
        h1 = xx+yy
        h2 = np.sqrt((xx-yy)**2+4*xy**2);
        s1 = .5*h1+.5*h2
        s2 = .5*h1-.5*h2
        h = (s1-xx)/xy
        L = np.sqrt(h**2+1)
        n11 = 1./L
        n12 = h/L
        h = (s2-xx)/xy
        L = np.sqrt(h**2+1)
        n21 = 1./L
        n22 = h/L 
    else:
        s1 = xx
        n11 = 1.
        n12 = 0
        s2 = yy
        n21 = 0
        n22 = 1. 
    return( [s1, n11, n12, s2, n21, n22] ) 

def PrinCLT(v1, v2, v3):                                # princial values, largest, direction   # used for ElasticLT material
    pp = PrinC( v1, v2, v3)                             # principal values
    if pp[0]>pp[3]:
        if abs(pp[1])>ZeroD: phi = np.arctan(pp[2]/pp[1])  # direction of larger tensile principal stress
        else: phi = pi/2
        pig = pp[0]                                     # larger principal value
        pig_= pp[3]                                     # lower principal value
    else:
        if abs(pp[4])>ZeroD: phi = np.arctan(pp[5]/pp[4])
        else: phi = pi/2
        pig = pp[3]                                     # larger principal value
        pig_= pp[0]                                     # lower principal value
    return pig, phi, pig_
def PrinCLT_(vx, vy, vxy):      # used for ElasticLT material
    a = 0.5*(vx-vy)
    b = np.sqrt(a**2+vxy**2)
    if b>ZeroD:
        c = 0.5*(vx+vy)
        v1 = c + b
        v2 = c - b
        if vxy<0.: si = -1.
        else:      si = 1.
        ang = 0.5*si*np.arccos(a/b)
    else:
        v1 = vx
        v2 = vy
        ang = 0.
    if v1>v2: return v1, ang, v2
    else:     return v2, ang+0.5*pi, v1

def AssignGlobalDof( NodeList, ElList, MatList):                 # assign dof indices
    for i in xrange(len(ElList)): 
        ElList[i].Ini2( NodeList, MatList)             # initialization of data depending on Sequence of nodes in NodeList which has been determined by Cuthill McKee
    Index = 0
    for i in xrange(len(NodeList)):                     # loop over nodes
        NodeList[i].GlobDofStart = Index                #
        Index = Index + len(NodeList[i].DofT)           # Node.DofT filled during data input / element initialization
    for i in xrange(len(ElList)):                       # loop over all elements to assign global dof index to element table if element shares this dof (which is not mandatory)
        Elem=ElList[i]
        for j in xrange(Elem.nNod):                     # loop over nodes of element
            Node = NodeList[Elem.Inzi[j]]               # global node of element
            set2 = Node.DofT                            # set of dof types of global node - not the same as element node
            set1 = Elem.DofT[j]                         # set of dof types of element node
            iterator1 = iter(set1)                      # 
            for k in xrange(len(set1)):                 # loop over number of dofs of element node
                k0 = iterator1.next()                   # dof type of element node
                iterator2 = iter(set2) 
                for l in xrange(len(set2)):             # loop over number of dofs of global node
                    l0 = iterator2.next()               # dof type of global node
                    if k0==l0:                          # element node dof found as global node dof
                        Elem.DofI[j,k] = Node.GlobDofStart + l# assign global dof index to element table
                        break
    Mask = np.ones((Index),dtype=int)                   # mask for particular dof types 
    for i in xrange(len(ElList)):       # loop over all elements
        Elem=ElList[i]
        for j in xrange(Elem.nNod):     # loop over all nodes in element
            set1 = Elem.DofT[j]
            iterator1 = iter(set1) 
            for k in xrange(len(set1)):
                k0 = iterator1.next()
                if k0==7: Mask[Elem.DofI[j,k]]=0        # Mask for dof type gradient damage
                
    Skyline = np.zeros((Index), dtype=int)
    SDiag = np.zeros((Index), dtype=int)        # array MAXA (FE Procedure-K.J.Bathe)
    for i in xrange(len(ElList)):                       # upper right skyline of system matrix
        Elem=ElList[i]
        minL = []   # define as type
        for j in Elem.DofI:
            try:    jc = j.compressed()                 # for mased array, see e.g.B23E
            except: jc = j
            minL += [min(set(jc).difference(set([-1])))] # subtracts -1 from jc, if it there (to get rid of -1 which might be there for some element types)
        min_ = min(minL)# Elem.DofI.min()
        for j in Elem.DofI:
            for k in j:
                if k>=0:
                    sl1 = k - min_          # page 986 Finite Element Procedure, K.J.Bathe
                    sl2 = min(k,sl1)+1
                    sl3 = Skyline[k]
                    if sl2>sl3: Skyline[k] = sl2
    for i in xrange(Index-1): 
        SDiag[i+1]=SDiag[i]+Skyline[i]                  # addresses of diagonal entries of system matrix in system vector
    SLen = SDiag[Index-1]+Skyline[Index-1]
#    for i in xrange(Index): print i, SDiag[i], Skyline[i]
    return Index, Mask, Skyline, SDiag, SLen             # return last global dof index

#@profile
def IntForces( N, MatList, ElList, Dt, VecC, VecU, VecS, VecT, VecI, MatK,MatG,KVecU,KVecL,Skyline,SDiag, CalcType, ff, NLGeom, SymS, Buckl):
                                                        # CalcType 0: check system 1: internal forces only 2: internal forces and tangential stiffness matrix 10: mass matrix only
    for i in xrange(N): VecI[i] = 0                     # initialization of internal nodal forces vector
    nEl = len(ElList)           # total number of elements
    for i in xrange(nEl):                       # loop over all elements
        # Initialize variables
        Elem = ElList[i]
        nint = Elem.nInt                                # integration order
        nnod = Elem.nNod                                # number of nodes per element
        matN = Elem.MatN                                # name of material
        InT  = Elem.IntT                                # integration type of element
        uvec = np.zeros((Elem.DofE), dtype=np.double)   # element displacements
        dvec = np.zeros((Elem.DofE), dtype=np.double)   # element displacement increments from last time step
        rvec = np.zeros((Elem.DofEini), dtype=np.double)# element residuals
        tvec = np.zeros((Elem.DofEini), dtype=np.double)# element temperatures
        svec = np.zeros((Elem.DofEini), dtype=np.double)# element temperature increments from last time step
        kmat = np.zeros((Elem.DofEini, Elem.DofEini), dtype=np.double)# element stiffness
        if NLGeom: gmat = np.zeros((Elem.DofEini, Elem.DofEini), dtype=np.double)# geometric stiffness

        # Determination of internal forces and element stiffness matrix
        ndof0, ndof0_ = 0, 0
        # from the global vectors VecU and VecC, compute local vectors related to the considered element : uvec and dvec
        for j in xrange(nnod):                          # local node loop
            for k in xrange(Elem.DofN[j]):              # local dof loop
                kk = Elem.DofI[j,k]                     # global index of local dof k of local node j
                uvec[ndof0+k] = VecU[kk]                # element displacements from global displacements
                dvec[ndof0+k] = VecU[kk]-VecC[kk]       # element displacement time step increments
            ndof0 = ndof0 + Elem.DofN[j]                # update entry index for element displacements
            for k in xrange(Elem.DofNini[j]):           # dof loop
                kk = Elem.DofI[j,k]                     # global index of local dof k of local node j
                tvec[ndof0_+k]= VecT[kk]                # element node temperatures
                svec[ndof0_+k]= VecT[kk]-VecS[kk]       # element node temperature time step increments
            ndof0_= ndof0_+ Elem.DofNini[j]             # update entry index for element displacements, initial values
#        for kk in xrange(7): sys.stdout.write('%14.4f'%(uvec[kk]))
#        sys.stdout.write('\n')
        if NLGeom and Elem.NLGeomI and not Elem.RotG: Elem.UpdateCoord(uvec, dvec) # update current element coordinates for all elements but sh4
        if Elem.Rot: 
            uvec = np.dot(Elem.Trans,uvec)              # transform global to local displacements
            dvec = np.dot(Elem.Trans,dvec)              # transform global to local displacement increments
        if NLGeom and Elem.NLGeomI and Elem.RotG: 
            Elem.UpdateCoord(uvec, dvec) # update current element coordinates for shell element SH4 (must be done here due to variable number of dofs per node before)
            Elem.UpdateElemData()       # uhc
        for j in xrange(Elem.nIntL):                    # build element stiffness with integration loop
            # retrieve informations of the integration point
            if Elem.ShellRCFlag and j>=Elem.nIntLi:     # for reinforcement layers
                set = Elem.Set
                r = SamplePointsRCShell[set,InT,nint-1,j][0]
                s = SamplePointsRCShell[set,InT,nint-1,j][1]
                t = SamplePointsRCShell[set,InT,nint-1,j][2]
                f = Elem.Geom[0,0]*SampleWeightRCShell[set,InT,nint-1,j] # weighting factor (GaussWeight -> Script Tab. 1.1; in SimFemBasics.py)
            else:
                r = SamplePoints[InT,nint-1,j][0]
                s = SamplePoints[InT,nint-1,j][1]
                t = SamplePoints[InT,nint-1,j][2]
                f = Elem.Geom[0,0]*SampleWeight[InT,nint-1,j] # weighting factor (GaussWeight -> Script Tab. 1.1; in SimFemBasics.py)
            if CalcType==10:                            # mass matrix
                det = Elem.JacoD(r, s, t)
                N = Elem.FormN(r, s, t)                 # shape function
                mm = MatList[matN].Mass(Elem)           # element mass in integration point
                kmat = kmat + det*f*Elem.Geom[1][0]*np.dot(np.transpose(N), np.dot(mm,N)) # element mass matrix
                continue
            if Elem.RegType ==1: 
                B, BN, det, TM = Elem.FormB(r,s,t, NLGeom) # shape function derivative for regularization 
            else:
                if ConFemElemCFlag and Elem.Type=='SH4':
                    B = np.zeros((6,20), dtype=float)
                    Data_ = np.zeros((1), dtype=float)
                    TM = np.zeros((6,6), dtype=float)
                    rc = SH4FormBC( r, s, t, B, Data_, Elem.XX, Elem.a, Elem.Vn, Elem.EdgeDir, Elem.gg[0], Elem.gg[1], TM )  # in _libElemC library
                    det = Data_[0]
                else:
                    B, det, TM = Elem.FormB(r,s,t, NLGeom) # shape function derivative
            T = Elem.FormT(r,s,t)                         # shape function for temperature interpolation
            # from uvec,dvec,tvec and svec, compute strains, strains increment of this integration point
            epsI = np.dot( B, uvec)                     # integration point strains
            dpsI = np.dot( B, dvec)                     # integration point strain increments
            tempI= np.dot( T, tvec)                     # integration point temperatures
            dtmpI= np.dot( T, svec)                     # integration point temperature increments
            if Elem.RegType==1: epsR = np.dot( BN, uvec)# integration point nonlocal equivalent strains
            if Elem.RotM:   # need to transform covariant strain to local strain
                epsI = np.dot(TM,epsI)
                dpsI = np.dot(TM,dpsI)
            # from epsI and dpsI, compute stress, stress increment of this integration point by basing in element material behaviour
            if Elem.RegType==1:
                sigI, C, sigR, CR, Data = MatList[matN].Sig( ff, CalcType, Dt,i,j, Elem, dpsI, epsI, dtmpI, tempI, epsR)# stress, stress incr, tangential stiffness 
            else:         sigI, C, Data = MatList[matN].Sig( ff, CalcType, Dt,i,j, Elem, dpsI, epsI, dtmpI, tempI, [])# stress, stress incr, tangential stiffness
#            print 'XXX', Elem.Label, j, epsI[0], sigI[0],sigI[1] #, C
            if CalcType==0: continue                    # next integration point in case of system check

            if Elem.RotM:   # transform local stress to contravariant stress
                C = np.dot(np.transpose(TM), np.dot(C,TM))
                sigI = np.dot(np.transpose(TM),sigI)
            # assemble recent results (sigI,C,Data) of integration point into data(rvec, Elem.Data, kmat) of the owner element
            for k in xrange(len(Data)): Elem.Data[j,k] = Data[k]# element data (strains, stresses etc.)
            if Elem.RegType==1: 
                rvec = rvec + det*f*Elem.Geom[1][0]*(np.dot(np.transpose(B), sigI)+np.dot(np.transpose(BN), sigR)) # element internal forces
            else:
                rvec = rvec + det*f*Elem.Geom[1][0]* np.dot(np.transpose(B), sigI) # element internal forces
            if CalcType==1: continue                    # nodal forces only
            if Elem.RegType==1: kmat = kmat + det*f*Elem.Geom[1][0]*(np.dot(np.transpose(B), np.dot(C,B))+np.dot(np.transpose(BN), np.dot(CR,BN)))# element stiffness
            else:               kmat = kmat + det*f*Elem.Geom[1][0]* np.dot(np.transpose(B), np.dot(C,B)) # element stiffness

            if NLGeom:
                if ConFemElemCFlag and Elem.Type=='SH4':
                    GeomSti = np.zeros((20,20), dtype=float)
                    DataX = np.zeros((1), dtype =float)
                    rc = SH4FormGeomC( r, s, t, GeomSti, DataX, sigI, Elem.gg[0], Elem.gg[1] )
                else: 
                    GeomSti = Elem.GeomStiff(r,s,t,sigI)
                if Elem.NLGeomCase==0: gmat = gmat + det*f*Elem.Geom[1][0]                                             *GeomSti# Elem.GeomStiff(r,s,t,sigI)
                else:                  gmat = gmat + Elem.Geom[1][0]*(SampleWeight[InT,nint-1,j]/SampleWeight[InT,0,0])*GeomSti# Elem.GeomStiff(r,s,t,sigI)
                
        if CalcType==0: continue                        # next element in case of system check
        # Transform local to global values
        if Elem.Rot: rvec = np.dot(Elem.Trans.transpose(),rvec)# transform local to global forces
        if Elem.Rot and CalcType==2: 
            kmat = np.dot(np.dot(Elem.Trans.transpose(),kmat),Elem.Trans)# transform local to global element stiffness
            if NLGeom and Elem.RotG: gmat = np.dot(np.dot(Elem.Trans.transpose(),gmat),Elem.Trans)

        # Assemble local vectors (rvec,kmat, Elem.Data) of the element into global vectors VecI, MatK, KVecU, KVecL
        ndof0 = 0
        for j0 in xrange(nnod):                         # assemble rows
            for k in xrange(Elem.DofN[j0]):             # loop over element row dofs
                kk = Elem.DofI[j0,k]                    # global dof row index
                VecI[kk] = VecI[kk] + rvec[ndof0+k]
            ndof0 = ndof0 + Elem.DofN[j0]

        if CalcType>=2:                                 # asssemble system matrices
            ndof0 = 0
            if MatK<>None:
                for j0 in xrange(nnod):                 # assemble rows
                    ndof1 = 0
                    for j1 in xrange(nnod):             # assemble columns
                        for k in xrange(Elem.DofN[j0]): # loop over element row dofs
                            kk = Elem.DofI[j0,k]        # global dof row index
                            for l in xrange(Elem.DofN[j1]): # loop over element column dofs
                                ll = Elem.DofI[j1,l]    # global dof column index
                                MatK[kk,ll] = MatK[kk,ll] + kmat[ndof0+k,ndof1+l]
                                if NLGeom: 
                                    if not Buckl: MatK[kk,ll] = MatK[kk,ll] + gmat[ndof0+k,ndof1+l]
                                    else:         MatG[kk,ll] = MatG[kk,ll] + gmat[ndof0+k,ndof1+l]
                        ndof1 = ndof1 + Elem.DofN[j1]   # update entry for element dof column index
                    ndof0 = ndof0 + Elem.DofN[j0]
            else:  # LinAlgFlag = True
                for j0 in xrange(nnod):                 # assemble rows
                    ndof1 = 0
                    for j1 in xrange(nnod):             # assemble columns
                        for k in xrange(Elem.DofN[j0]): # loop over element row dofs
                            kk = Elem.DofI[j0,k]            # global dof row index
                            for l in xrange(Elem.DofN[j1]): # loop over element column dofs
                                ll = Elem.DofI[j1,l]    # global dof column index
                                if ll>=kk:              # upper right part column >= row
                                    jj = SDiag[ll] + ll - kk    # index in stiffness vector
                                    KVecU[jj] = KVecU[jj] + kmat[ndof0+k,ndof1+l] # update stiffness vector
                                elif not SymS:          # lower left part column < row
                                    jj = SDiag[kk] + kk - ll    # index in stiffness vector
                                    KVecL[jj] = KVecL[jj] + kmat[ndof0+k,ndof1+l] # update stiffness vector
                                if NLGeom:
                                    if ll>=kk:          # upper right part column >= row
                                        jj = SDiag[ll] + ll - kk    # index in stiffness vector
                                        KVecU[jj] = KVecU[jj] + gmat[ndof0+k,ndof1+l] # update stiffness vector
                                    elif not SymS:      # lower left part column < row
                                        jj = SDiag[kk] + kk - ll    # index in stiffness vector
                                        KVecL[jj] = KVecL[jj] + gmat[ndof0+k,ndof1+l] # update stiffness vector
                        ndof1 = ndof1 + Elem.DofN[j1]   # update entry for element dof column index
                    ndof0 = ndof0 + Elem.DofN[j0]

    return 0

def ArcLength(gamma, UI, dUII, DU, DY, Mask):           # arc length control, arclength measure, load displ, residuum displ, displ step incr,
#   aa =  np.dot(UI,UI)                                 # parameter a
    aa = MaskedP(len(UI), UI,UI, Mask)                  # parameter a
    if aa<ZeroD: raise NameError("Arc length control: exit 1")
#   bb =   np.dot(DU,UI)   + np.dot(dUII,UI)            # parameter b
    bb =   MaskedP(len(UI),DU,UI,Mask) + MaskedP(len(UI),dUII,UI,Mask) # parameter b
#   cc = 2*np.dot(DU,dUII) + np.dot(dUII,dUII)          # parameter c
    cc = 2*MaskedP(len(UI),DU,dUII,Mask) + MaskedP(len(UI),dUII,dUII,Mask) # parameter c
#   dd =   np.dot(DU,DU)                                # parameter d
    dd =   MaskedP(len(UI),DU,DU,Mask)                  # parameter d
    cp = cc+dd-gamma**2
    qq = bb**2-aa*cp
    if qq<0: return -bb/(2*aa)
    else:
        L1 = (-bb-np.sqrt(qq))/aa                       # arc length solution 1
        L2 = (-bb+np.sqrt(qq))/aa                       # arc length solution 2
#       x1 = np.dot(DY,(DY+dUII))
        x1 = MaskedP(len(UI),DY,(DY+dUII),Mask)
#       x2 = np.dot(DY,UI)
        x2 = MaskedP(len(UI),DY,UI,Mask)
        g1 = x1 + L1*x2
        g2 = x1 + L2*x2
        if g1>g2: return L1
        else: return L2
def MaskedP(N, V1, V2, Ma_):
    xx = 0.
    for i in xrange(N):
        xx = xx + V1[i]*V2[i]*Ma_[i]
    return xx

def FinishEquilibIteration( MatList, ElList, ff, NLGeom, LoFl):
    Flag = False
    for Elem in ElList:                                 # loop over all elements
        for j in xrange(Elem.nIntL):                    # loop over all integration points
                for k in xrange(len(Elem.DataP[j])): Elem.DataP[j][k] = Elem.Data[j][k]   # store Eleme.Data of current converged solution as prev. solution for next step
        if len(Elem.StateVar) > 0:
            Mat = MatList[Elem.MatN] 
            if MatList[Elem.MatN].Update:   # if have UpdateStateVar in material
                Flag2 = MatList[Elem.MatN].UpdateStateVar(Elem, ff)
                if not Flag and Flag2:
                    Flag = True
                    logging.getLogger(__name__).debug('There is at least one element has UpdateStateVar. Change Flag=False->True')
            else:
                for j in xrange(Elem.StateVar.shape[0]):# loop over all integration and integration sub points
                    Elem.StateVar[j] = Elem.StateVarN[j]
# uhc        if Elem.ElemUpdate and NLGeom: Elem.UpdateElemData()
    for Elem in ElList:                                 # loop over all elements
        if len(Elem.StateVar) > 0:
            if MatList[Elem.MatN].Updat2: MatList[Elem.MatN].UpdateStat2Var(Elem, ff, Flag, LoFl)
    return 0

def CuthillMckee(NodeList, ElList):

    # Function for getting the NodeConnectionDegree
    for i in xrange(len(NodeList)):
        node = NodeList[i]
        ConNodes = set()                                                          # uhc
        for k in xrange(len(ElList)):
            elem = ElList[k]
            if node.Label in elem.InzList:                                        # uhc
                InzSet = set(elem.InzList)                                        # uhc
                ConNodes=ConNodes.union(InzSet)                                   # uhc
        node.ListOfConnectedNodes_ = list(ConNodes.difference(set([node.Label]))) # uhc
        node.LevelBereitsErhalten_ = False                                        # uhc
        # Festlegen des Vernetzungsgrades durch Zaehlen der Verknuepfungen
        node.ConnectionDegree_ = len(node.ListOfConnectedNodes_)
        
    # Suchen des geringsten Vernetzungsgrades
    NodeWithLowestConDeg_ = 100                                                    # uhc
    for i in xrange(len(NodeList)):                                                # uhc
        node = NodeList[i]                                                         # uhc
#        if node.ConnectionDegree_ == 0: node.CMLabel_ = 0                          # uhc
        if node.ConnectionDegree_ <= NodeWithLowestConDeg_ and node.ConnectionDegree_>0: # uhc
            NodeWithLowestConDeg_ = node.ConnectionDegree_                         # uhc
    
    # Auflisten der Knoten mit dem geringsten Vernetzungsgrad 
    NodeListWithLowestConDeg_ = []                                                 # uhc
    for i in xrange(len(NodeList)):                                                # uhc
        node = NodeList[i]                                                         # uhc   
        if node.ConnectionDegree_ <= NodeWithLowestConDeg_ and node.ConnectionDegree_>0: # uhc
            NodeListWithLowestConDeg_.append(node.Label)                           # uhc

    usedNodes_ = []                                                                # Knoten, welche schon einer Stufe zugeordnet wurden
    startPointLabelF_ = NodeListWithLowestConDeg_[0]
    for i in xrange(len(NodeList)):
        node = NodeList[i]
        if node.Label == startPointLabelF_: startPointNodeIndex_ = FindIndexByLabel( NodeList, node.Label) # uhc
    
    startNode = NodeList[startPointNodeIndex_]                                     # uhc
    startPointNodeLabel_ = startNode.Label                                         # uhc  
    startNode.CMLabel_ =  1                                                        # uhc
    startNode.LevelBereitsErhalten_ = True                                         # uhc
    startNode.Level_ = 0                                                           # uhc
    usedNodes_.append(startNode.Label)                                             # uhc
    currentCMLabelFVal_ = 2                                                        # uhc
    LevelList_ = [startPointNodeLabel_]                                            # uhc
    def Relabel(currentCMLabelFVal_, LevelList_, usedNodes_, level):               # uhc
        CurrentConNodes = set()                                                    # uhc
        for i in LevelList_:                                                       # uhc
            node = NodeList[FindIndexByLabel(NodeList, i)]                         # uhc
            CurrentConNodes = CurrentConNodes.union(set(node.ListOfConnectedNodes_)) # uhc
        CurrentConNodes = CurrentConNodes.difference(set(usedNodes_))              # uhc
        CurrentConNodes = list(CurrentConNodes)                                    # uhc
        CurrentConNodes =  sorted(CurrentConNodes, key=lambda t: NodeList[FindIndexByLabel(NodeList, t)].ConnectionDegree_) # uhc
        for u in CurrentConNodes:                                                  # uhc
            currentNode = NodeList[FindIndexByLabel(NodeList, u)]                  # uhc
            if not currentNode.LevelBereitsErhalten_:                              # uhc
                currentNode.CMLabel_ = currentCMLabelFVal_                         # uhc
                currentNode.Level_ = level                                         # uhc
                currentNode.LevelBereitsErhalten_ = True                           # uhc
                usedNodes_ += [currentNode.Label]                                  # uhc
                currentCMLabelFVal_ += 1                                           # uhc
        return CurrentConNodes, currentCMLabelFVal_                                # uhc
    level = 0                                                                      # uhc
    while len(LevelList_)>0:                                                       # uhc
        LevelList_, currentCMLabelFVal_ = Relabel(currentCMLabelFVal_, LevelList_, usedNodes_, level)  # uhc
        level += 1                                                                 # uhc
#    for i in xrange(len(NodeList)):
#        node = NodeList[i]
#        print 'X', i, node.Label, node.CMLabel_

def IsSymSys( MatList ):
    for i in list(MatList.values()):   # if there is at least one unsymmetric material the whole system is unsymmetric
        if (not i.Symmetric) and (i.Used):
            return False
    return True
