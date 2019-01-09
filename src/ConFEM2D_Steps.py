# ConFem2D_Steps
from numpy import transpose, sqrt, zeros, array, double, dot, cos

from ConFEM2D_Basics import FindIndexByLabel, SamplePoints, SampleWeight
from lib_Mat import MisesUniaxial

class Boundary(object):
    def __init__(self, NodeLabel, Dof, Val, NoList, AmpLbl, AddVal):
        self.Dof = Dof
        self.Val = Val
        self.NodeLabel = NodeLabel
# uhc       self.Index = FindIndexByLabel( NoList, NodeLabel)
        self.Amplitude = AmpLbl
        self.AddVal = AddVal
        self.ValOffset = 0.
class CLoad(object):
    def __init__(self, NodeLabel, Dof, Val, NoList, AmpLbl):
        self.Dof = Dof
        self.Val = Val
        self.NodeLabel = NodeLabel
# uhc        self.Index = FindIndexByLabel( NoList, NodeLabel)
        self.Amplitude = AmpLbl
class Temperature(object):
    def __init__(self, NodeLabel, Val, NoList, AmpLbl):
        self.Val = Val
        self.NodeLabel = NodeLabel
#        self.Index = FindIndexByLabel( NoList, NodeLabel)
        self.Amplitude = AmpLbl
class PreStress(object):
    def __init__(self, Type, PrePar, GeomL, AmpLbl):
        self.For = PrePar[0]                            # prestressing force
        self.Ap = PrePar[1]                             # cross section of tendon
        self.GeomL = GeomL                              # elements and geometry
        self.Amplitude = AmpLbl
        self.PreLength = 1.                             # prestressing length
        self.PreLenZ = 1.                               # length of prestressing after 1st step                      
        self.Reinf = MisesUniaxial( [PrePar[2],.2,PrePar[3],PrePar[4],PrePar[5],PrePar[6],PrePar[7],PrePar[8]], [0.,1.] )# initialization material for prestressing
        self.StateVar = zeros((1,3), dtype=float)       # state variables for mises material + tendon stress during tensioning 
        self.StateVarN= zeros((1,3), dtype=float)       # updated state variables
        self.TensStiff = False
        self.Geom = array([[0.],[1.]])
        self.dim = 1                                    # for compatibility reasons
        self.nint = 3
        if   Type == "POST-BONDED":    self.BondLess = False
        elif Type == "POST-UNBONDED": self.BondLess = True
        else: raise NameError ("Unknown type of prestressing")
        if not self.BondLess: self.PrStrain = zeros((len(GeomL),self.nint),dtype=float) # storage for strain of prestressed fiber - after tensioning 
class PreStreSt(object):                                # prestressing data related to step
    def __init__(self, Name, AmpLbl):
        self.Name = Name                                # prestressing force
        self.Amplitude = AmpLbl
class DLoad(object):
    def __init__(self, ElLabel, Dof, Val, AmpLbl):
        self.ElLabel = ElLabel
        self.Dof = Dof
        self.Val = Val
        self.Amplitude = AmpLbl
class ElFile(object):
    def __init__(self, OutTime):
        self.OutTime = OutTime                          # time interval output
class NoFile(object):
    def __init__(self, OutTime):
        self.OutTime = OutTime                          # time interval output
class Step(object):
    current = 0                                         # initialization of current step
    PreSDict = {}                                       # dictionary for all prestressings
    def __init__(self):
        self.IterTol = 1.e-3
        self.IterNum = 10
        self.TimeStep = 1.
        self.TimeTarg = 1.
        self.SolType ="NR"                              # NR: Newton Raphson, BFGS: BFGS
        self.BoundList = []                             # boundary conditions
        self.CLoadList = []                             # nodal loads
        self.DLoadList = []                             # distributed loads
        self.TempList = []                              # nodal temperatures
        self.PrestList = []                             # prestressing data with reference to step
        self.ElFilList = []                             # time inteval for writing output files for element data
        self.NoFilList = []                             # time inteval for writing output files for nodal data
        self.AmpDict = {}                               # dictionary for Amplitudes
        self.AmpDict['Default'] = [[0.,0.],[1.,1.]]
        self.ZeroD = 1.e-9                              # Smallest float for division by Zero
        self.Dyn = False                                # Flag for implicit dynamics 
        self.Buckl = False                              # Flag for buckling / modal analysis 
        self.NMbeta = 0.25                              # Newmark parameter beta
        self.NMgamma = 0.5                              # Newmark parameter gamma
        self.Damp = False                               # Flag for artificial damping
        self.RaAlph = 0.0                               # Rayleigh damping parameter with stiffness                 
        self.RaBeta = 0.0                               # Rayleigh damping parameter with mass                 
        self.ArcLen = False                             # flag for arc length control
        self.ArcLenV = 0                                # arc length parameter
        self.NLGeom= False                              # Flag for large deformations
        self.varTimeSteps= False                        # Flag for variable TimestepSizes

    def AmpVal(self, Time, Label):                      # determine value for amplitude
        Data = self.AmpDict[Label]
        nD = len(Data)
        if Time>=Data[nD-1][0]:                         # time exceeds last value
            DelT = Data[nD-1][0]-Data[nD-2][0]
            if DelT<self.ZeroD: raise NameError ("something wrong with amplitude 1")
            return (Time-Data[nD-2][0])/DelT * (Data[nD-1][1]-Data[nD-2][1]) + Data[nD-2][1]
        for i in xrange(len(Data)-1):
            if Data[i][0]<=Time and Time<=Data[i+1][0]: # time within interval
                DelT = Data[i+1][0]-Data[i][0]
                if DelT<self.ZeroD: raise NameError ("something wrong with amplitude 2")
                return (Time-Data[i][0])/DelT * (Data[i+1][1]-Data[i][1]) + Data[i][1]
        print('T',Time,Data)
        raise NameError ("something wrong with amplitude 3")

    def BoundOffset(self, NodeList, VecU):          # add offset for prescribed displacement from current displacement for OPT=ADD
        for i in self.BoundList:                        # loop over all boundary conditions of step
            if i.AddVal:
                nI = FindIndexByLabel( NodeList, i.NodeLabel) # node index of bc
                Node = NodeList[nI]
                iterator = iter(Node.DofT)              # iterator for set
                for j in xrange(len(Node.DofT)):        # loop over all dofs of node
                    jj=iterator.next()
                    if jj==i.Dof: break                 # dof type equals prescribed dof type 
                k = Node.GlobDofStart + j               # constrained dof global index
                i.ValOffset = VecU[k]
#                print 'XXX', Node.Label, i.Dof, i.ValOffset
        return 0

    def BoundCond(self, N, Time, TimeS, TimeTar, NodeList, VecU, VecI, VecP, VecP0, BCIn, BCIi, Kmat,KVecU,KVecL,Skyline,SDiag, CalcType, SymS):# introduce loads and boundary conditions into system
        for i in self.BoundList:                        # loop over all boundary conditions of step
            Found = False
            val = self.AmpVal(Time, i.Amplitude)*i.Val + i.ValOffset  # prescribed value for actual time
            valT = self.AmpVal(TimeTar, i.Amplitude)*i.Val + i.ValOffset # prescribed value for final target
            valS = self.AmpVal(TimeS, i.Amplitude)*i.Val + i.ValOffset # prescribed value for step target
            valT = valT - valS
            nI = FindIndexByLabel( NodeList, i.NodeLabel) # node index of bc
            iterator = iter(NodeList[nI].DofT)          # iterator for set
            for j in xrange(len(NodeList[nI].DofT)):    # loop over all dofs of node
                jj=iterator.next()
                if jj==i.Dof:           # dof type equals prescribed dof type 
                    Found = True
                    break 
            if not Found: raise NameError ("ConFemSteps.Step.BoundCond: missing correspondence for dof type")
            k = NodeList[nI].GlobDofStart + j           # constrained dof global index
            if CalcType==2:                             # only for NR, not for MNR, BFGS
                if Kmat<>None:
                    for j in xrange(N):                 # loop over rows and columns simul
                        VecI[j] = VecI[j] - VecU[k]*Kmat[j,k] # internal forces due to prescribed displacement
                        VecP[j] = VecP[j] - val*    Kmat[j,k] # external forces to enforce prescribed displacement
                        VecP0[j] = VecP0[j] - valT* Kmat[j,k] # external nominal forces corresponding to prescribed displacements
                        Kmat[j,k] = 0.                  # modification stiffness matrix
                        Kmat[k,j] = 0.
                    Kmat[k,k] = 1.                      # modification stiffness matrix
                elif len(KVecU)>0:
                    jj = SDiag[k]                       # same for stiffness vectors
                    # modify values connected with system matrix column
                    for j in xrange(Skyline[k]):
                        VecI[k-j]  = VecI[k-j]  - VecU[k]*KVecU[jj+j] # internal forces due to prescribed displacement
                        VecP[k-j]  = VecP[k-j]  - val    *KVecU[jj+j] # external forces to enforce prescribed displacement
                        VecP0[k-j] = VecP0[k-j] - valT   *KVecU[jj+j] # external nominal forces corresponding to prescribed displacements
                    for j in xrange(Skyline[k]): KVecU[jj+j] = 0. # upper right part of system matrix 
                    if not SymS:
                        for j in xrange(Skyline[k]): KVecL[jj+j] = 0. # lower left part of system matrix
                    # modify values connected with system matrix row -- "diff" within loops unfortunately is not constant due to banded vector storage
                    if not SymS:                        # unsymmetric system
                        for j in xrange(k,N):
                            diff = j-k
                            if diff<Skyline[j]:
                                valK = KVecL[SDiag[j]+diff]
                                VecI[j] = VecI[j] - VecU[k]*valK
                                VecP[j] = VecP[j] - val*    valK
                                VecP0[j] = VecP0[j] - valT* valK
                        for j in xrange(k,N):
                            diff = j-k
                            if diff<Skyline[j]:
                                KVecL[SDiag[j]+diff] = 0.
                    else:                               # use KVecU instead KVecL in case of symmetric system
                        for j in xrange(k,N):
                            diff = j-k
                            if diff<Skyline[j]:
                                valK = KVecU[SDiag[j]+diff]
                                VecI[j] = VecI[j] - VecU[k]*valK 
                                VecP[j] = VecP[j] - val*    valK 
                                VecP0[j] = VecP0[j] - valT* valK 
                    for j in xrange(k,N):
                        diff = j-k
                        if diff<Skyline[j]: KVecU[SDiag[j]+diff] = 0.
                    #
                    KVecU[jj] = 1.
            VecI[k] = VecU[k]                           # 
            VecP[k] = val                               # modification load vector
            VecP0[k] = valT                             #
            BCIn[k] = 0                                 # index for prescribed displacements
            BCIi[k] = 1
        return 0

    def NodalLoads(self, N, Time, TimeTar, NodeList, VecP, VecP0):      # introduce concentrated loads into system
        for i in xrange(N): VecP[i] = 0
        for i in xrange(N): VecP0[i] = 0
        for i in xrange(len(self.CLoadList)):           # loop over all concentrated loads of step
            val = self.AmpVal(Time, self.CLoadList[i].Amplitude)*self.CLoadList[i].Val# prescribed value
            valT = self.AmpVal(TimeTar, self.CLoadList[i].Amplitude)*self.CLoadList[i].Val# prescribed value
# uhc            nI = self.CLoadList[i].Index                # node index of concentrated load
            nI = FindIndexByLabel( NodeList, self.CLoadList[i].NodeLabel) # node index of concentrated load
            iterator = iter(NodeList[nI].DofT)          # iterator for set
            for j in xrange(len(NodeList[nI].DofT)):    # loop over all dofs of node
                j0 = iterator.next()                    # dof type
                if j0==self.CLoadList[i].Dof: break     # dof type equals prescribed dof type
            k = NodeList[nI].GlobDofStart + j           # loaded dof global index
            VecP[k] = VecP[k] + val                     # global load vector
            VecP0[k] = VecP0[k] + valT                  # global load vector
        return 0

    def ElementLoads(self, Time, TimeTar, ElList, NodeList, VecP, VecP0):# introduce distributed loads into system
        for i in xrange(len(self.DLoadList)):           # loop over all distributed/element loads of step
            Label = self.DLoadList[i].ElLabel
            Dof = self.DLoadList[i].Dof
            Val = self.AmpVal(Time, self.DLoadList[i].Amplitude)*self.DLoadList[i].Val# prescribed value
            ValT = self.AmpVal(TimeTar, self.DLoadList[i].Amplitude)*self.DLoadList[i].Val# prescribed value
            for j in xrange(len(ElList)):
                Elem = ElList[j]
                if Elem.Set==Label or str(Elem.Label)==Label:
                    nfie = Elem.nFie                    # number of loading degrees of freedom
                    if Elem.Rot: TraM = Elem.Trans[0:nfie,0:nfie]
                    ev = zeros((nfie), dtype=double)
                    ev0 = zeros((nfie), dtype=double)
                    if Elem.Type=='SB3':
                        ev[1] = Val
                        ev0[1] = ValT
                    else:
                        ev[Dof-1] = Val
                        ev0[Dof-1] = ValT
                    if Elem.Rot:
                        ev = dot(TraM,ev)
                        ev0 = dot(TraM,ev0)
                    pv=zeros(( Elem.DofE ),dtype=double)# element nodal loads
                    pv0=zeros(( Elem.DofE ),dtype=double)# element nodal loads
                    InT  = Elem.IntT                    # integration type for element
                    nint = Elem.nInt                    # integration order
                    nintL= Elem.nIntLi                   # total number of integration points
                    for k in xrange(nintL):             # integration point loop
                        r = SamplePoints[InT,nint-1,k][0]
                        s = SamplePoints[InT,nint-1,k][1]
                        t = SamplePoints[InT,nint-1,k][2]
                        if Elem.Type=='SH4': t=0
                        f = Elem.JacoD(r,s,t)*Elem.Geom[0,0]*SampleWeight[InT,nint-1,k]# weighting factor, Elem.Geom[0,0] is a measure for length, area ...
                        N = Elem.FormN(r,s, t)          # shape function
                        pv = pv + f*dot( N.T, ev)
                        pv0 = pv0 + f*dot( N.T, ev0)
                    if Elem.Rot:
                        pv = dot(Elem.Trans.transpose(),pv)# transform local to global forces (Script Eq. (3.68))
                        pv0 = dot(Elem.Trans.transpose(),pv0)# transform local to global forces (Script Eq. (3.68))
                    ndof = 0
                    for k in xrange(Elem.nNod):         # assemble
                        for l in xrange(Elem.DofN[k]):
                            ii = Elem.DofI[k,l]
                            VecP[ii] = VecP[ii] + pv[ndof+l] 
                            VecP0[ii] = VecP0[ii] + pv0[ndof+l] 
                        ndof = ndof + Elem.DofN[k]
        return 0

    def NodalPrestress(self, N, Time, ElList, NodeList, VecP, VecU, NLg):  # introduce nodal prestress
        for i in xrange(len(self.PrestList)):           # loop over all prestressings of step
            pName = self.PrestList[i].Name
            if self.PreSDict[pName].BondLess: pF = self.PreSDict[pName].PreLength/self.PreSDict[pName].PreLenZ # relative change of prestressing length
            else: pF = 1
            Val =  self.AmpVal(Time, self.PrestList[i].Amplitude) * self.PreSDict[pName].For # prescribed value
            Emod = 2.e5
            if self.current == 0: self.PreSDict[pName].StateVarN[0,2] = Val/(self.PreSDict[pName].Ap*Emod) # prestressing strain during tensioning
            GeomL = self.PreSDict[pName].GeomL          # geometric an other prestressing data
            LP = 0.                                     # initialization arc length of prestressing
            for j in xrange(len(GeomL)):                # loop over all elements of prestressing
                Elem = ElList[GeomL[j][0]]
                uvec = zeros((Elem.DofE), dtype=double) # element displacements
                ndof0 = 0
                for k in xrange(Elem.nNod):             # loop over all element nodes to determine element displacements
                    for k1 in xrange(Elem.DofN[k]): uvec[ndof0+k1] = VecU[Elem.DofI[k,k1]]# element displacements from global displacements
                    ndof0 = ndof0 + Elem.DofN[k]        # update entry index for element displacements
                nfie = Elem.nFie                        # number of loading degrees of freedom
                if Elem.Rot: TraM = Elem.Trans[0:nfie,0:nfie]
                pv=zeros(( Elem.DofE ),dtype=double)    # element nodal loads
                InT  = Elem.IntT                        # integration type for element
                nint = Elem.nInt                        # integration order
                nintL= Elem.nIntLi                       # total number of integration points
                for k in xrange(nintL):                 # integration point loop
                    r = SamplePoints[InT,nint-1,k][0]
                    f =  Elem.Geom[0,0]*SampleWeight[InT,nint-1,k]# weighting factor
                    B, jd, TM = Elem.FormB(r, 0, 0, NLg)       # shape function derivative
                    epsI = dot( B, uvec)                # integration point strains (Script Eq. (1.6))
                    P = Elem.FormP(r, 0)                # form function for prestressing 
                    DatP1 = dot(P[0,:],GeomL[j][1])     # interpolation of prestressing force
                    DatP2 = dot(P[1:3,:],GeomL[j][1])   # interpolation of prestressing geometry
                    DatP3 = dot(P[2,:],uvec)            # interpolation of inclination from deformation
                    if not self.PreSDict[pName].BondLess:
                        epsP = epsI[0] - DatP2[0]*epsI[1] # strain of beam fiber in height of prestressing
                        if self.current==0: 
                            self.PreSDict[pName].PrStrain[j,k]=epsP
                        else:
                            Sig, Dsig, dataL = self.PreSDict[pName].Reinf.Sig(None, 1,0.,j,0,self.PreSDict[pName], [0.,0], [epsP,0],0.,0.,None)#CalcType, Dt, i, ipL, Elem, [dep, 0], [eps, 0], dtp, tmp
                            pF = ( epsP - self.PreSDict[pName].PrStrain[j,k] + self.PreSDict[pName].StateVarN[0,2] )/self.PreSDict[pName].StateVarN[0,2]
                    sigP = pF*Val*DatP1*cos(DatP2[1]) * array([1,-DatP2[0]])# internal forces due to prestressing
                    pv = pv + f*dot( B.T, sigP)         # element nodal forces due to prestressing
                    LP = LP + f*sqrt((1+epsI[0])**2+(DatP2[1]+DatP3)**2)# arc length of deformed prestressing
                if Elem.Rot: pv = dot(Elem.Trans.transpose(),pv)# transform local to global forces (Script Eq. (3.68))
                ndof = 0
                for k in xrange(Elem.nNod):             # assemble
                    for l in xrange(Elem.DofN[k]):
                        ii = Elem.DofI[k,l]
                        VecP[ii] = VecP[ii] - pv[ndof+l] 
                    ndof = ndof + Elem.DofN[k]
            self.PreSDict[pName].PreLength = LP         # current tendon length        
            if self.current==0: self.PreSDict[pName].PreLenZ=LP  # tendon length upon application of prestressing in 1st step
        return 0

    def NodalTemp(self, N, Time, NodeList, VecT):       # nodal temperatures
        for i in xrange(N): VecT[i] = 0
        for i in xrange(len(self.TempList)):            # loop over all temperatures of step
# uhc            nI = self.TempList[i].Index                 # node index
            nI = FindIndexByLabel( NodeList, self.TempList[i].NodeLabel) # node index of temperature
            k = NodeList[nI].GlobDofStart               # global index
            tL= self.TempList[i].Val                    # list with temperature values
            tA= self.AmpVal(Time, self.TempList[i].Amplitude)# amplitude value
            for j in xrange(len(tL)): VecT[k+j]=VecT[k+j]+tA*tL[j]# global temperature vector
        return 0