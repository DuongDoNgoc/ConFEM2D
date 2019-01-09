# ConFemMat
# Copyright (C) [2016] [Ulrich Haeussler-Combe]
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License (GNU GPLv3) as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program; if not, see <http://www.gnu.org/licenses
#
import matplotlib.pyplot as plt
from numpy import array, sin, exp, log, cos, dot, pi, double, zeros, sqrt, outer, transpose, tan
from numpy.linalg import eigh, norm, det, eigvalsh
from numpy import fabs
from scipy import integrate
import sys
import logging

from ConFEM2D_Basics import ZeroD, PrinCLT, PrinCLT_, I21Points, I21Weights, StrNum0, SampleWeightRCShell
from ConFemRandom import RandomField_Routines
try:
    import libMatC
    reload(libMatC)
    from libMatC import *
    ConFemMatCFlag = True    
except ImportError:
    ConFemMatCFlag = False

class Material(object):
    Density   = 0
    Symmetric = False
    RType     = None
    Used      = False                               # flag whether this material is actually used (referenced to by elements), may be later set to True
    Update    = False
    Updat2    = False
    Conc      = None                                # used as handle if actual material is a wrapper, e.g. WraRCShell
    def __init__(self, SymmetricVal, RTypeVal, UpdateVal, Updat2Val, StateVarVal, NDataVal):
        self.Symmetric = SymmetricVal               # flag (True/False) for symmetry of material matrices
        self.RType     = RTypeVal                   # string for type of regularization (None: no regularization)
        self.Update    = UpdateVal                  # flag whether state variables have update method
        self.Updat2    = Updat2Val                  # flag whether state variables have update method connected to special conditions
        self.StateVar  = StateVarVal                # number of state variables (None: no state variables)
        self.NData     = NDataVal                   # number of data items for output
        self.Conc      = None                       # used if actual material is a wrapper, e.g. WraRCShell
        self.CrBwN     = 50                         # crack band regularization: number of intervals for scaling factor interpolation
        self.Type      = None                       # 
    def Mass(self, Elem):                           # mass matrix
        if Elem.dim==10:
#            val = self.Density*Elem.Geom[1,1]*Elem.Geom[1,2] # beam mass per unit length from element integration point
            val = self.Density*Elem.Geom[1,3]       # beam mass per unit length from element integration point
            MatS = array([[val,0],[0,val]])         # beam mass matrix
            return MatS
        elif Elem.dim==11:
            val = self.Density*Elem.Geom[1,1]*Elem.Geom[1,2] # beam mass per unit length from element integration point
#            MatS = array([[val,0,0],[0,val,0],[0,0,0]]) # beam mass matrix
            MatS = array([[val,0,0],[0,val,0],[0,0,1.e-3]]) # beam mass matrix
#            MatS = array([[val,0,0],[0,val,0],[0,0,1.e-2]]) # beam mass matrix
            return MatS
        elif Elem.dim==2:
            val = self.Density*Elem.Geom[1,0]       # deep beam mass per unit thickness
            MatS = array([[val,0],[0,val]])         # beam mass matrix
            return MatS
        elif Elem.dim==1:
            val = self.Density*Elem.Geom[1,0]       # 
            MatS = array([[val,0],[0,val]])
            return MatS
        else: raise NameError("ConFemMaterial::Material.Mass: mass not yet defined for this element type")
    def ViscExten3D(self, Dt, eta, Dps, Elem, ipI, sI):
        VepsOld = array([Elem.StateVar[ipI,sI],Elem.StateVar[ipI,sI+1],Elem.StateVar[ipI,sI+2],Elem.StateVar[ipI,sI+3],Elem.StateVar[ipI,sI+4],Elem.StateVar[ipI,sI+5]]) # strain rate of previous time step 
        if norm(VepsOld)<ZeroD or dot(Dps,VepsOld)<0.: 
            if Dt>ZeroD: VepsOld = Dps/Dt
            else:        VepsOld = zeros((6), dtype=float)
        if Dt>ZeroD and norm(Dps)>ZeroD: Veps = 2.*Dps/Dt - VepsOld             # actual strain rate
        else:                            Veps = VepsOld
        if Dt>ZeroD: zz = 2.*self.eta/Dt
        else:        zz = 0.
        Elem.StateVarN[ipI,sI]   = Veps[0]
        Elem.StateVarN[ipI,sI+1] = Veps[1]
        Elem.StateVarN[ipI,sI+2] = Veps[2]
        Elem.StateVarN[ipI,sI+3] = Veps[3]
        Elem.StateVarN[ipI,sI+4] = Veps[4]
        Elem.StateVarN[ipI,sI+5] = Veps[5]
        return zz, Veps
    def SpeCrEnergy(self, lam, eps_ct):                 # crack energy for uniaxial tension for unit volume (specific crack energy)
        xP, yP = [], []
        tol = 0.001*self.fct
        de = 0.01e-3
        ee = eps_ct                                     # start of integration: critical strain tension
        x = 1.
        while x>tol:                                    # to determine a suitable end for integration of specific crack enery
            x = self.UniaxTensionScaled( lam, ee)
            ee += de
            xP += [ee]
            yP += [x]
        epsEnd = ee
        if False: #self.Type == "MicroPl": 
            plt.plot(xP,yP)
            plt.grid()
        nI = 100                                        # number of integration samples
        sia = zeros((nI+1),dtype=double)
        ee = eps_ct
        de = (epsEnd-ee)/nI
        for i in xrange(nI+1):                          # generate samples
            sia[i] = self.UniaxTensionScaled( lam, ee)
            ee += de
        return integrate.trapz( sia, x=None, dx=de)     # integrate samples
    def SpeCrEnergyScaled(self, la1, la2, n, eps_ct):   # scaled uniaxial tension: samples of scaling factor depending on specific crack energy 
        qq = pow(la2/la1,1./n)                          # factor of geometric row
        CrX, CrY = zeros((n+1),dtype=float), zeros((n+1),dtype=float)
        for i in xrange(n+1):
            arg = la1*qq**i
            CrX[n-i] = self.SpeCrEnergy( arg, eps_ct )/self.SpecCrEn
            CrY[n-i] = arg 
#        plt.plot(CrX,CrY) 
        return CrX, CrY

class Template(Material):                              # elastoplastic Mises
    def __init__(self, PropMat):
#    def __init__(self,         SymmetricVal, RTypeVal, UpdateVal, Updat2Val, StateVarVal, NDataVal):
        Material.__init__(self, True,         None,     True,      False,     1,           8)
#        self.Symmetric = True                       # flag for symmetry of material matrices
#        self.Update = True                          # has specific update procedure for update of state variables
#        self.Updat2 = False                         # no 2 stage update procedure
#        self.StateVar = 1                           # number of state variables per integration point
#        self.NData = 8                              # number of data items

        self.Emod = PropMat[0]                      # Young's modulus
        self.nu = PropMat[1]                        # Poissons's ratio
        self.sigY = PropMat[2]                      # uniaxial yield stress
        self.sigU = PropMat[3]                      # strength
        self.epsU = PropMat[4]                      # limit strain
        self.alphaT = PropMat[5]                    # thermal expansion coefficient
        self.Density = PropMat[6]                   # specific mass
        self.epsY = self.sigY/self.Emod             # uniaxial yield strain  ????
        self.Etan = (self.sigU-self.sigY)/(self.epsU-self.epsY)# tangential / hardening modulus
        self.H = self.Etan/(1-self.Etan/self.Emod)  # hardening modulus
        
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        if CalcType == 0: return [], [], []
        sigy = max(self.sigY,Elem.StateVar[ipI][0]) # current state parameter - current uniaxial yield stress
        Elem.StateVarN[ipI][0] = 0                  # current values of state variables have to initialized again
        if Elem.dim==2 or Elem.dim==3:           # plane stress/strain biaxial
            nu = self.nu                            # Poisson's ratio
            mu = self.Emod/(2.*(1.+nu))             # shear modulus
            C0 = self.Emod*(1-nu)/((1+nu)*(1-2*nu))*array([[1,nu/(1-nu),nu/(1-nu),0,0,0],
                                                           [nu/(1-nu),1,nu/(1-nu),0,0,0],
                                                           [nu/(1-nu),nu/(1-nu),1,0,0,0],
                                                           [0,0,0,(1-2*nu)/(2*(1-nu)),0,0],
                                                           [0,0,0,0,(1-2*nu)/(2*(1-nu)),0],
                                                           [0,0,0,0,0,(1-2*nu)/(2*(1-nu))]]) # triaxial isotropic elasticity 
            if Elem.dim==2:                         # plate plane stress / plane strain
                Sig = array( [Elem.DataP[ipI,3],Elem.DataP[ipI,4],Elem.DataP[ipI,6],0.,0.,Elem.DataP[ipI,5]] ) # stress of previous increment
                dEps = array([Dps[0],Dps[1],0.,0.,0.,Dps[2]]) # total strain increment
            elif Elem.dim==3:
                Sig = array([Elem.DataP[ipI,0],Elem.DataP[ipI,1],Elem.DataP[ipI,2],Elem.DataP[ipI,3],Elem.DataP[ipI,4],Elem.DataP[ipI,5]] ) # stress of previous increment
                dEps = array([Dps[0],Dps[1],Dps[2],Dps[3],Dps[4],Dps[5]]) # total strain increment
            Fn = sqrt(3.*(((Sig[0]-Sig[1])**2+(Sig[0]-Sig[2])**2+(Sig[1]-Sig[2])**2)/6.+Sig[3]**2+Sig[4]**2+Sig[5]**2))-sigy # distance to yield surface of previous step
            Eflag = False
            dEpp = zeros((6),dtype=float)           # initial value plastic strain increment
            dSig = zeros((6),dtype=float)           # initial value plastic strain increment
            dLam = 0.                               # initial value plastic multiplier increment
            dsiy = 0.                               # initial value current yield stress increment
            ni = 20                                 # iteration limit
            for i in xrange(ni):
                if Elem.PlSt: dEps[2]=-( C0[2,0]*(dEps[0]-dEpp[0])
                                        +C0[2,1]*(dEps[1]-dEpp[1])
                                        +C0[2,3]*(dEps[3]-dEpp[3])
                                        +C0[2,4]*(dEps[4]-dEpp[4])
                                        +C0[2,5]*(dEps[5]-dEpp[5]))/C0[2,2]+dEpp[2] # lateral strain for plane stress
                dSig = dot(C0,dEps-dEpp)
                SigN = Sig + dSig
                J2 = ((SigN[0]-SigN[1])**2+(SigN[0]-SigN[2])**2+(SigN[1]-SigN[2])**2)/6.+SigN[3]**2+SigN[4]**2+SigN[5]**2 # 2nd stress deviator invariant
                if sqrt(3.*J2)-sigy<1.e-9:          # elastic loading, unloading or reloading
                    Eflag = True
                    break
                sm = (SigN[0]+SigN[1]+SigN[2])/3.   # 1st stress invariant / mean stress predictor stress
                rr = sqrt(3./(4.*J2))*array([SigN[0]-sm,SigN[1]-sm,SigN[2]-sm,SigN[3],SigN[4],SigN[5]]) # yield gradient predictor stress
                dL = (Fn + dot(rr,dSig)+rr[3]*dSig[3]+rr[4]*dSig[4]+rr[5]*dSig[5] - self.H*dLam)/(3.*mu+self.H) # with Voigt notation correction
                if dL<1.e-9: break
                dLam = dLam + dL                    # update plastic multiplier
                dEpp = dLam*array([rr[0],rr[1],rr[2],2.*rr[3],2.*rr[4],2.*rr[5]]) # plastic strain incremen with Voigt notation correction
            if i>=ni-1:  
                print elI, ipI, i, ni, Sig, dEps, sigy, dLam, dEpp, dSig  
                raise NameError ("ConFemMaterials::Mises.Sig: no convergence")
            Sig = SigN
            if Eflag:                               # elastic loading or unloading / reloading
                if Elem.dim==2:                     # plane stress / strain
                    if Elem.PlSt: MatM = self.Emod/(1-nu**2)*array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]) # plane stress
                    else:         MatM = self.Emod*(1-nu)/((1+nu)*(1-2*nu))*array([[1,nu/(1-nu),0],[nu/(1-nu),1,0],[0,0,(1-2*nu)/(2*(1-nu))]]) # plane strain
                else: MatM = C0
            else:                                   # plastic loading
                Elem.StateVarN[ipI][0]= sqrt(3.*J2) # new equivalent yield limit 
                A = 1./(self.H + 3.*mu)             # --> 2.*mu*dot(rr,rr) --> dot(xx,rr) --> dot(C0,rr)
                CC = C0 -  A*4.*mu**2*outer(rr,rr)  # --> A*outer(xx,xx)            # tangential material stiffness
                cD = 1./CC[2,2]
                if Elem.dim==2:
                    if Elem.PlSt: MatM=array([[CC[0,0]-CC[0,2]*CC[2,0]*cD,CC[0,1]-CC[0,2]*CC[2,1]*cD,CC[0,5]-CC[0,2]*CC[2,5]*cD],
                                              [CC[1,0]-CC[1,2]*CC[2,0]*cD,CC[1,1]-CC[1,2]*CC[2,1]*cD,CC[1,5]-CC[1,2]*CC[2,5]*cD], 
                                              [CC[5,0]-CC[5,2]*CC[2,0]*cD,CC[5,1]-CC[5,2]*CC[2,1]*cD,CC[5,5]-CC[5,2]*CC[2,5]*cD]])
                    else:         MatM=array([[CC[0,0],CC[0,1],CC[0,5]],[CC[1,0],CC[1,1],CC[1,5]],[CC[5,0],CC[5,1],CC[5,5]]])
                elif Elem.dim==3:
                    MatM=array([[CC[0,0],CC[0,1],CC[0,2],CC[0,3],CC[0,4],CC[0,5]],
                                [CC[1,0],CC[1,1],CC[1,2],CC[1,3],CC[1,4],CC[1,5]],
                                [CC[2,0],CC[2,1],CC[2,2],CC[2,3],CC[2,4],CC[2,5]],
                                [CC[3,0],CC[3,1],CC[3,2],CC[3,3],CC[3,4],CC[3,5]],
                                [CC[4,0],CC[4,1],CC[4,2],CC[4,3],CC[4,4],CC[4,5]],
                                [CC[5,0],CC[5,1],CC[5,2],CC[5,3],CC[5,4],CC[5,5]]])
            if Elem.dim==2:
                sig = array([Sig[0],Sig[1],Sig[5]])
                return sig, MatM, [Eps[0], Eps[1], Eps[2], Sig[0], Sig[1], Sig[5], Sig[2]] # data
            else:                                   # 3D, shell
                sig = array([Sig[0],Sig[1],Sig[2],Sig[3],Sig[4],Sig[5]])
                return sig, MatM, [Sig[0],Sig[1],Sig[2],Sig[3],Sig[4],Sig[5], 0.] # data
        else: raise NameError ("ConFemMaterials::Mises.Sig: not implemented for this element type")
    def UpdateStateVar(self, Elem, ff):
        for j in xrange(Elem.StateVar.shape[0]):    # loop over integration points 
            if Elem.StateVarN[j,0]>Elem.StateVar[j,0]: Elem.StateVar[j,0] = Elem.StateVarN[j,0]
        return False

class Elastic(Material):                            # linear elastic
    def __init__(self, PropMat):
#                        (self, SymmetricVal, RTypeVal, UpdateVal, Updat2Val, StateVarVal, NDataVal):
        Material.__init__(self, True, None, False, False, None, 6)
#        self.Symmetric = True                       # flag for symmetry of material matrices
#        self.StateVar = None
#        self.NData = 6                              # number of data items

        self.PropMat = PropMat
        self.alphaT = 1.e-5                         # thermal expansion coefficient ###################
        self.Density = PropMat[2]                   # specific mass
        self.Dam = False                            # flag for damage
    def C3(self, Emod, nu):
        ff = Emod / ( ( 1. + nu ) * ( 1. - 2.*nu ) )
        return ff*array([[1.-nu,nu,nu,0,0,0],[nu,1.-nu,nu,0,0,0],[nu,nu,1.-nu,0,0,0],[0,0,0,(1.-2.*nu)/2.,0,0],[0,0,0,0,(1.-2.*nu)/2.,0],[0,0,0,0,0,(1.-2.*nu)/2.]])
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        if CalcType == 0: return [], [], []
        Emod = self.PropMat[0]
        nu = self.PropMat[1]
        if self.Dam: 
            if   Elem.dim==1: 
                if Elem.RegType==1: 
                    sig, MatM, sigR, MatR = self.StateVarValues( ff, Dt, elI, ipI, Elem, Dps, Eps, EpsR)
                    return sig, MatM, sigR, MatR, [Eps[0], sig[0]]
                else: 
                    sig, MatM = self.StateVarValues( ff, Dt, elI, ipI, Elem, Dps, Eps, EpsR)
                    return sig, MatM, [Eps[0], sig[0]]
            elif Elem.dim==2:
                if Elem.RegType==1:
                    sig, MatM, sigR, MatR = self.StateVarValues( ff, Dt, elI, ipI, Elem, Dps, Eps, EpsR)
                    return sig, MatM, sigR, MatR, [Eps[0], Eps[1], Eps[2], sig[0], sig[1], sig[2]]
                else: 
                    sig, MatM = self.StateVarValues( ff, Dt, elI, ipI, Elem, Dps, Eps, EpsR)
                    return sig, MatM, [Eps[0], Eps[1], Eps[2], sig[0], sig[1], sig[2]]
            elif Elem.dim==3:
                if Elem.RegType==1: 
                    sig, MatM, sigR, MatR = self.StateVarValues( ff, Dt, elI, ipI, Elem, Dps, Eps, EpsR)
#                    return sig, MatM, sigR, MatR, [sig[0], sig[1], sig[2], sig[3], sig[4], sig[5]]
                    return sig, MatM, sigR, MatR, [sig[0], sig[1], sig[2], Eps[0], Eps[1], Eps[2]]
                else: 
                    sig, MatM = self.StateVarValues( ff, Dt, elI, ipI, Elem, Dps, Eps, EpsR)
                    return sig, MatM, [sig[0], sig[1], sig[2], sig[3], sig[4], sig[5]]
            else: raise NameError("ConFemMaterials::Elastic.Dam: not implemented")
        else:
            if Elem.dim==1 or Elem.dim==99:             # uniaxial / spring
                MatM = array([[Emod,0],[0,0]])          # material stiffness uniaxial
                sig = [Elem.DataP[0,1],0] + dot(MatM,Dps) # stress incrementally
                return sig, MatM, [Eps[0], sig[0]]
            elif Elem.dim==2:
                if Elem.PlSt: MatM = Emod/(1-nu**2)*array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]) # plane stress
                else:         MatM = Emod*(1-nu)/((1+nu)*(1-2*nu))*array([[1,nu/(1-nu),0],[nu/(1-nu),1,0],[0,0,(1-2*nu)/(2*(1-nu))]]) # plane strain
                sig = dot(MatM,Eps)                     # stress
                return sig, MatM, [Eps[0], Eps[1], Eps[2], sig[0], sig[1], sig[2]]
            elif Elem.dim==3: 
                MatM = self.C3( Emod, nu)               # triaxial isotropic elasticity
                sig = dot(MatM,Eps)                     # stress
                return sig, MatM, [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]]
            elif Elem.dim==10:                          # bernoulli beam
                hh =  Elem.Geom[1,2]                    # height from element integration point
                AA, SS, JJ = Elem.Geom[1,3], Elem.Geom[1,4], Elem.Geom[1,5] 
                Tr = 0.5*(Temp[0]+Temp[1])              # temperature of reference / middle axis
                Tg = (Temp[0]-Temp[1])/hh               # temperature gradient
                Tps = self.alphaT*array([Tr,Tg])        # temperature strain
                MatM = array([[Emod*AA,-Emod*SS],[-Emod*SS,Emod*JJ]])# beam tangential stiffness
                sig = dot(MatM,Eps-Tps)                 #
                return sig, MatM, [Eps[0],Eps[1],sig[0],sig[1]]
            elif Elem.dim==11:                          # timoshenko beam
                bb =  Elem.Geom[1,1]                    # width from element integration point
                hh =  Elem.Geom[1,2]                    # height from element integration point
                alpha = 0.8                             # factor for shear stiffness
    #            Tr = 0.5*(Temp[0]+Temp[1])              # temperature of reference / middle axis
    #            Tg = (Temp[0]-Temp[1])/hh               # temperature gradient
    #            Tps = self.alphaT*array([Tr,Tg])        # temperature strain
                MatM = array([[Emod*bb*hh,0,0],[0,Emod*bb*hh**3/12.,0],[0,0,bb*hh*alpha*0.5*Emod/(1+nu)]])# beam tangential stiffness
                sig = dot(MatM,Eps)                     #
                return sig, MatM, [Eps[0],Eps[1],Eps[2],sig[0],sig[1],sig[2]]
            elif Elem.dim==20:                          # Kirchhoff slab
                hh =  Elem.Geom[1,1]                    # height
#                KK = Emod*hh**3/(12*(1-nu))             # slab stiffness
                KK = Emod*hh**3/(12.*(1.-nu**2))             # slab stiffness
#                MatM = array([[KK,nu*KK,0],[nu*KK,KK,0],[0,0,(1-nu)*KK]])# slab stiffness. (1-nu)*KK] should be strictly divided by 2 and B_xy doubled, see SB3.FormB.
                MatM = array([[KK,nu*KK,0],[nu*KK,KK,0],[0,0,0.5*(1-nu)*KK]])
                sig = dot(MatM,Eps)                 #
                return sig, MatM, [Eps[0],Eps[1],Eps[2],sig[0],sig[1],sig[2]]
            elif Elem.dim==21:                          # Continuum based shell
                MatM = array([[Emod/(1-nu**2), nu*Emod/(1-nu**2), 0., 0., 0., 0.],
                              [nu*Emod/(1-nu**2), Emod/(1-nu**2), 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., Emod/(2+2*nu), 0., 0.],
                              [0., 0., 0., 0., Emod/(2+2*nu), 0],
                              [0., 0., 0., 0., 0., Emod/(2+2*nu)]]) 
                sig = dot(MatM,Eps)                     #
                return sig, MatM, [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]]
            elif Elem.dim==98:                          # 2D spring with 3 dofs
                MatM = array([[0.,0.,0.],[0.,0.,0.],[0.,0.,Emod]])
                sig = dot(MatM,Eps)                     #
                return sig, MatM, [sig[0],sig[1],sig[2],Eps[0],Eps[1],Eps[2]]
            else: raise NameError ("ConFemMaterials::Elastic.Sig: not implemented for this element type")

class ElasticOrtho(Material):                           # linear orthotropic elastic
    def __init__(self, PropMat):
        Material.__init__(self, True, None, False, False, None, 6)
        self.E1   = PropMat[0]
        self.E2   = PropMat[1]
        self.E3   = PropMat[2]
        self.nu12 = PropMat[3]
        self.nu13 = PropMat[4]
        self.nu23 = PropMat[5]
        self.G1   = PropMat[6]
        self.G2   = PropMat[7]
        self.G3   = PropMat[8]
        self.Density   = PropMat[9]                 # specific density
        if self.E1<ZeroD or self.E2<ZeroD: raise NameError ("ConFemMat::ElasticOrtho.__init__: E1 and E2 should be larger than zero")
        self.nu21 = self.nu12*self.E2/self.E1           # to have symmetry
        self.nu31 = self.nu13*self.E3/self.E1
        self.nu32 = self.nu23*self.E3/self.E2
        self.ff = -1./(-1+self.nu32*self.nu23+self.nu21*self.nu12+self.nu21*self.nu32*self.nu13+self.nu31*self.nu12*self.nu23+self.nu31*self.nu13) 
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        if Elem.dim==3:
            MatM = self.ff*array([[(1.-self.nu32*self.nu23)*self.E1,(self.nu12+self.nu32*self.nu13)*self.E2,(self.nu12*self.nu23+self.nu13)*self.E3,0,0,0],
                                  [(self.nu21+self.nu31*self.nu23)*self.E1,(1.-self.nu31*self.nu13)*self.E2,(self.nu23+self.nu21*self.nu13)*self.E3,0,0,0],
                                  [(self.nu21*self.nu32+self.nu31)*self.E1,(self.nu32+self.nu31*self.nu12)*self.E2,(1.-self.nu21*self.nu12)*self.E3,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0],
                                  [0,0,0,0,0,0]])
            MatM[3,3], MatM[4,4], MatM[5,5] = self.G1, self.G2, self.G3
            sig = dot(MatM,Eps)                     # stress
            return sig, MatM, [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]]
#            return sig, MatM, [Eps[0],Eps[1],Eps[2],Eps[3],Eps[4],Eps[5]]
        elif Elem.dim==98:
            MatM = array([[self.E1,0.,0.],[0.,self.E2,0.],[0.,0.,self.G3]])
            sig = dot(MatM,Eps)                     #
            return sig, MatM, [sig[0],sig[1],sig[2],Eps[0],Eps[1],Eps[2]]
        else: raise NameError ("ConFemMaterials::ElasticOrtho.Sig: not implemented for this element type")

class ElasticLT(Material):                          # linear elastic with limited tensile strength
    """
    Material properties input: Emod, nu, fct, Gf, bw, phi, zeta, cCrit, CrackVisco
    Emod = Young's modulus;     nu = Poisson's ratio
    fct = yield strength in case of uniaxial stress
    Gf = fracture energy
    bw = crack bandwidth
    phi = creep number;     zeta = viscosity parameter
    cCrit = type of criterion for tensile failure 0-stress 1-strain
    CrackVisc = crack viscosity
    """
    def __init__(self, PropMat):
#                        (self, SymmetricVal, RTypeVal, UpdateVal, Updat2Val, StateVarVal, NDataVal):
        Material.__init__(self, False,        None,     True,      True,      13,          8)
#        self.Symmetric = False                       # flag for symmetry of material matrices
#        self.Update = True                          # has specific update procedure
#        self.Updat2 = True                          # two stage update procedure
#        self.StateVar = 13                          # number of state variables per integration point
                                                    # [0] state, [1] max. prin stress,
                                                    # [2] largest crack width reached ever, [3] crack traction related to [2], 
                                                    # [4] max prin strain, [5] direction corresponding to 1st crack (not used)
                                                    # [6] state2, [7] largest crack width reached ever 2, [8] crack traction related to [7] 2
                                                    # [9] principal strain 0 of previous step, [10] principal strain 1 of previous step,
                                                    # [11] principal stress 0 prev. step, [12] prin stress 1 prev. step
#        self.NData = 8                              # number of data items
        self.PropMat = PropMat                      # for wrappers, e.g. RCSHELL
        self.Emod = PropMat[0]
        self.nu = PropMat[1]
        self.fct = PropMat[2]                       # uniaxial tensile strength
        self.Gf = PropMat[3]                        # fracture energy
        self.bw = PropMat[4]                        # crack bandwidth
        self.epsct = PropMat[2]/PropMat[0]          # uniaxial strain at tensile strength
        if self.fct>ZeroD: self.wcr = 2*self.Gf/self.fct           # critical crack width, may be reset by ConfemRandom
        else:              self.wcr=0
#            if self.epsct*self.bw/self.wcr > 0.01: raise NameError("ElasticLT.init: incoherent material parameters")
        self.phi = PropMat[5]                       # creep number
        self.zeta = PropMat[6]                      # viscosity parameter
        self.cCrit = PropMat[7]                     # type of criterion for tensile failure 0: stress, 1: strain
        self.CrackVisc = PropMat[8]                 # crack viscosity
        self.Density = PropMat[9]                   # specific mass
        if self.cCrit == 0: self.cVal = 0. #self.fct   
        else:               self.cVal = self.epsct  # may be reset by ConfemRandom
        self.alpha = 0.5                            # parameter for numerical integration with trapezoidal rule
        self.alphaT = 1.e-5                         # thermal expansion coefficient

        self.sigM = 0.                              # determine max tensile stress/strain for attached element set
        self.Elem = None                            # " -> corresponding element index
        self.IPoint = None                          # " -> corresponding integration point index
        self.sigM2 = 0.                             # determine max tensile stress/strain for attached element set
        self.Elem2 = None                           # " -> corresponding element index
        self.IPoint2 = None                         # " -> corresponding integration point index
        self.NoCracks=0                             # Number of cracks
#    @profile
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        self.fct=RandomField_Routines().get_property(Elem.Set,Elem.Label,self.fct)[0]
        self.wcr,self.cVal=RandomField_Routines().more_properties(self.fct,self.Gf,self.cCrit,self.epsct,self.wcr,self.cVal,Elem.Set,Elem.Label)
        depT = array([self.alphaT*dTmp,0.])         # increment of imposed strain uniaxial !!!
        epsT = array([self.alphaT*Temp,0.])         # imposed strain uniaxial !!!
        if CalcType==0:     # check system. Being executed when the new counter is in turn
#            if self.cCrit==0: sip = Elem.StateVarN[ipI,1] # retrieve largest principal stress
            if self.cCrit==0:
                sip = Elem.StateVarN[ipI,1]-self.fct # retrieve largest principal stress distance to tensile strength, regards randomness uhc
            else:
                sip = Elem.StateVarN[ipI,4] # retrieve largest principal strain
            if Elem.StateVarN[ipI,0]==0:            # check for 1st crack
                if sip>self.sigM:                   # determine point with largest principal stress/strain
                    self.sigM = sip
                    self.Elem = elI
                    self.IPoint = ipI
            elif self.cCrit==0:                     # check for 2nd crack
                if Elem.StateVarN[ipI,1]>self.sigM2: # determine point with largest principal stress
#                    self.sigM2 = Elem.StateVarN[ipI,1]
                    self.sigM2 = Elem.StateVarN[ipI,1]-self.fct # uhc
                    self.Elem2 = elI
                    self.IPoint2 = ipI
            return [], [], []
        else:
            if self.Elem==elI and self.IPoint==ipI and Elem.StateVarN[ipI,0]==0 and self.sigM>self.cVal:
                Elem.StateVarN[ipI,0] = 1               # first crack starts
                self.NoCracks = self.NoCracks+1
                logging.getLogger(__name__).debug('Update ElasticLT 1st crack at some Elems, NoCrack')
                print        '\n', 'El', Elem.Label, ipI, 'ElasticLT Element cracks', self.sigM, self.NoCracks
                print >> ff, '\n', 'El', Elem.Label, ipI, 'ElasticLT Element cracks', self.sigM, self.NoCracks
                self.sigM = 0.      # reset self.variables for next iterations crack check
                self.Elem = None
                self.IPoint = None
            if Elem.StateVarN[ipI,0]==4 and Elem.StateVarN[ipI,6]==0 and self.Elem2==elI and self.IPoint2==ipI and self.sigM2>self.cVal:
                Elem.StateVarN[ipI,6] = 1               # second crack starts
                self.NoCracks = self.NoCracks+1
                logging.getLogger(__name__).debug('Update ElasticLT 2nd crack at some Elems, NoCrack' )
                print        '\n', 'El', Elem.Label, ipI, 'ElasticLT Element cracks twice', self.sigM2, self.NoCracks
                print >> ff, '\n', 'El', Elem.Label, ipI, 'ElasticLT Element cracks twice', self.sigM2, self.NoCracks
                self.sigM2 = 0.
                self.Elem2 = None
                self.IPoint2 = None
            if Elem.dim==1:                             # uniaxial
                V = Dt*self.zeta*self.Emod              # auxiliary values for creep and relaxation depending on Dt
                W = self.zeta*(1+self.phi)
                W_hat = 1+self.alpha*Dt * W
                W_I = 1/W_hat
                W_bar = W_I*(1-(1-self.alpha)*Dt*W) - 1
                C_bar = W_I*self.Emod
                V_bar = W_I*V
                if Elem.StateVarN[ipI,0] == 1:          # cracked state 1 - 1st crack counted
                    MatM = array([[0,0],[0,0]])         # material tangential stiffness
                    Eps[0] = 0
                    sig = array([0,0], dtype=double)
                else:                                   # uncracked state
                    cc = C_bar + self.alpha*V_bar
                    epsP = array( [Elem.DataP[ipI,0], 0.] )  # DataP = data of previous step
                    sigP = array( [Elem.DataP[ipI,1], 0.] )
                    MatM = array([[cc,0],[0,0]] )       # material tangential stiffness
                    siV = W_bar*sigP + V_bar*epsP       # viscoelastic part of stress
                    sig = sigP + dot(MatM,(Dps-depT)) + siV # total stress
                    Elem.StateVarN[ipI,1] = sig[0]  # gan vao max prin stress cua Elem
                    Elem.StateVarN[ipI,4] = Eps[0]  # gan vao max prin strain
                return sig, MatM, [ (Eps[0]-epsT[0]), sig[0]] # ! returns stress inducing strain  , Data= [strain, stress]
            elif Elem.dim==2 or Elem.dim==21:
                if not ConFemMatCFlag: #False: #False: #False: #True: # flag for C-version
                    if Elem.dim==2:     # 2D plate element
                        if not Elem.PlSt: raise NameError("ConFemMaterials::ElasticLT.sig: ElasticLT not yet defined for plane strain")
                        Eps_ = Eps
                    elif Elem.dim==21:  # Continuum-based shell element
                        Eps_ = array([Eps[0],Eps[1],Eps[5]])    # only retrieve strain xx,yy,xy
                    nu = self.nu                                # Poisson's ratio
                    Emod = self.Emod
                    ww1, ww2 = 0., 0.                           # initial value crack width
                    State = int(Elem.StateVarN[ipI,0])          # current state of actual time step
                    pep,phe,pep_ = PrinCLT_( Eps_[0], Eps_[1], 0.5*Eps_[2]) # principal strains: largest value, corr. direction, lower value
                    Elem.StateVarN[ipI,9] =  pep                # 1st larger principal strain
                    Elem.StateVarN[ipI,10] = pep_               # 2nd principal strain

                    if State > 0:                               # cracked state
                        TOL = -1.e-15                           # 
                        State2 = int(Elem.StateVarN[ipI,6])     # current state 2 of actual time step
                        if State2 > 0: nu=0.
                        # state of previous time step
                        epsct_ = self.epsct                     # 
                        StateP1 = Elem.StateVar[ipI,0]          # final state of last time step 1st crack
                        StateP2 = Elem.StateVar[ipI,6]          # final state of last time step 2nd crack
                        wP1 = Elem.StateVar[ipI,2]              # largest crack ever reached of last time step first crack
                        wP2 = Elem.StateVar[ipI,7]              # largest crack ever reached of last time step second crack
                        # auxiliary values
                        alfa = Elem.Lch/self.wcr                # auxiliary value
                        xi = self.bw/Elem.Lch
                        if wP1>ZeroD: beta1 = Elem.StateVar[ipI,3]/wP1*Elem.Lch # auxiliary value ; = crack traction/Lch
                        else:         beta1 = Emod
                        if wP2>ZeroD: beta2 = Elem.StateVar[ipI,8]/wP2*Elem.Lch # auxiliary value
                        else:         beta2 = Emod
                        gam   = (1-xi)*(1-nu**2)
                        delt  = (1-xi)*nu
                        etaDt = self.CrackVisc                  # eta / Delta t , Delta t implicitly considered
                        etaDt2= 0
                        eta_  = 1.-etaDt
                        eta_2 = 1.+etaDt2                       # +  ??? 
                        aect = alfa*epsct_
                        if aect>0.9: raise NameError("ConFemMaterials::ElasticLT.Sig: incoherent system parameters, please reduce element size") # restriction for characteristic element length Lc
                        D1 = 1./(1.-aect*gam*eta_)              # auxiliary value / denominator of whole stuff state 1
                        D2_1 = 1./(1.+beta1/Emod*gam*eta_2)
                        D2_2 = 1./(1.+beta2/Emod*gam*eta_2)
                        etaDt4= etaDt
                        D4 = etaDt4/(1.+alfa*epsct_*gam*etaDt4)
                        kap = D4*self.fct*alfa
                        ka_ = 1-kap/Emod*gam
                        # transformation to local system
                        Trans=array([[cos(phe)**2,sin(phe)**2, cos(phe)*sin(phe)],
                                     [sin(phe)**2,cos(phe)**2,-cos(phe)*sin(phe)],
                                     [-2*cos(phe)*sin(phe),2*cos(phe)*sin(phe),cos(phe)**2-sin(phe)**2]])# transformation matrix for strains from global to local
                        epsL = dot(Trans,Eps_)                   # local strain
                        epsLP = array([Elem.StateVar[ipI,9],Elem.StateVar[ipI,10],0.]) # local strain previous step uhc
                        if etaDt>0. or etaDt2>0.:
                            TranT=array([[cos(phe)**2,sin(phe)**2, 2*cos(phe)*sin(phe)],
                                         [sin(phe)**2,cos(phe)**2,-2*cos(phe)*sin(phe)],
                                         [-cos(phe)*sin(phe),cos(phe)*sin(phe),cos(phe)**2-sin(phe)**2]])# transformation matrix for stresses from global to local
                            sigLP = array([Elem.StateVar[ipI,11],Elem.StateVar[ipI,12],0.])  # local stress previous step uhc
                            # auxiliary values depend on local stress of previous step
                            sigEta1_1 = D1 * etaDt* (epsct_*alfa*gam*sigLP[0]-self.fct*alfa*(epsLP[0]+delt*epsLP[1]))
                            sigEta2_1 = D1 * etaDt* (epsct_*alfa*gam*sigLP[1]-self.fct*alfa*(epsLP[1]))
                            sigEta1_2 = D2_1*etaDt2*(  gam/Emod*sigLP[0]-                  (epsLP[0]+delt*epsLP[1]))  #? presumably not yet correct, but no effects with etaDt2=0
                            sigEta2_2 = D2_1*etaDt2*(  gam/Emod*sigLP[1]-                  (epsLP[1]))                #?
                            sigEta1_4 = D4*(epsct_*alfa*gam*sigLP[0]-self.fct*alfa*(epsLP[0]+delt*epsLP[1]))
                            sigEta2_4 = D4*(epsct_*alfa*gam*sigLP[1]-self.fct*alfa*(epsLP[1])) #0.
                        else: sigEta1_1, sigEta1_2, sigEta1_4, sigEta2_1, sigEta2_2, sigEta2_4 = 0., 0., 0., 0., 0., 0.
                        # presumable crack width values depending on actual local strain
                        ww1_1 = Elem.Lch*(D1*(epsL[0]+delt*epsL[1]-epsct_*gam)-gam*sigEta1_1/Emod)# crack width state 1
                        ww2_1 = Elem.Lch*(D1*(epsL[1]             -epsct_*gam)-gam*sigEta2_1/Emod)# crack width state 1
                        ww1_2 = Elem.Lch*(D2_1*(epsL[0]+delt*epsL[1])           -gam*sigEta1_2/Emod)# crack width state 2
                        ww2_2 = Elem.Lch*(D2_2*(epsL[1])                        -gam*sigEta2_2/Emod)# crack width state 2
                        ww1_4 = Elem.Lch*(ka_*epsL[0]+delt*ka_*epsL[1]        -gam*sigEta1_4/Emod) # crack width state 4
                        ww2_4 = Elem.Lch*(ka_*epsL[1]                         -gam*sigEta2_4/Emod) # crack width state 4

                        MatML = zeros((3,3), dtype=float)
                        sigL  = zeros((3), dtype=float)
                        State_ = None
                        State2_ = 0 #None
                        def State4C():
                            MatML[0,0] = kap
                            MatML[0,1] = kap*delt
                            MatML[1,0] = nu*kap
                            MatML[1,1] = Emod+nu*kap*delt
                            sigL[0] = kap*(epsL[0]  +delt*epsL[1])             +   sigEta1_4
                            sigL[1] = nu*kap*epsL[0]+(Emod+nu*kap*delt)*epsL[1]+nu*sigEta1_4
                            Elem.StateVarN[ipI,2] = ww1_4
                            return  4, ww1_4
                        def State1C(i, ww):
                            if i==0:
                                MatML[0,0] = D1*Emod*(  -aect*eta_) 
                                MatML[0,1] = D1*Emod*(  -aect*delt*eta_)
                                MatML[1,0] = D1*Emod*(  -aect*nu*eta_) 
                                MatML[1,1] = D1*Emod*(1.-aect*(1-xi)*eta_)
                                sigL[0]    = D1*self.fct*(1-alfa*(epsL[0]+delt*epsL[1])*eta_)+sigEta1_1
                                sigL[1]    = D1*Emod*(nu*epsct_*1-nu*aect*eta_*epsL[0]+(1-aect*(1-xi)*eta_)*epsL[1])+nu*sigEta1_1
                                Elem.StateVarN[ipI,i+2] = ww
                                Elem.StateVarN[ipI,i+3] = sigL[i]
                            return 1, ww
                        def State2C():
                            MatML[0,0] = D2_1*(eta_2*beta1)
                            MatML[0,1] = D2_1*(eta_2*delt*beta1)
                            MatML[1,0] = D2_1*(eta_2*nu*beta1)
                            MatML[1,1] = D2_1*(Emod+eta_2*beta1*(1-xi))
                            sigL[0] = MatML[0,0]*epsL[0] + MatML[0,1]*epsL[1] +    sigEta1_2
                            sigL[1] = MatML[1,0]*epsL[0] + MatML[1,1]*epsL[1] + nu*sigEta1_2
                            return 2, ww1_2
                        def State3C():
                            xxx = Emod/(1-nu**2)
                            MatML[0,0] = xxx
                            MatML[0,1] = xxx*nu
                            MatML[1,0] = xxx*nu
                            MatML[1,1] = xxx 
                            sigL[0] = MatML[0,0]*epsL[0] + MatML[0,1]*epsL[1]
                            sigL[1] = MatML[1,0]*epsL[0] + MatML[1,1]*epsL[1]
                            return 3, 0.
                        def State4C2():
                            MatML[1,1] = 0.
                            sigL[1] = 0.
                            Elem.StateVarN[ipI,7] = ww2_4
                            return  4, ww2_4
                        def State1C2():
                            MatML[1,1] = D1*Emod*(-aect*eta_)
                            sigL[1]    = D1*self.fct*(1-alfa*epsL[1]*eta_)+sigEta2_1
                            Elem.StateVarN[ipI,7] = ww2_1
                            Elem.StateVarN[ipI,8] = sigL[1]
                            return 1, ww2_1
                        def State2C2():
                            MatML[1,1] = D2_2*(eta_2*beta2)
                            sigL[1] = MatML[1,1]*epsL[1] + nu*sigEta2_2
                            return 2, ww2_2
                        def State3C2():
                            MatML[1,1] = Emod 
                            sigL[1] = MatML[1,1]*epsL[1]
                            return 3, 0.
                        # actual state of cracking depending on previous state and crack width predictors 
                        if StateP1>=4:
                            if ww1_4>0:                                                                # state 4 open crack loading or unloading
                                State_, ww1 = State4C()  # direction 1 decoupled from 2 for the following (nu = 0)
                                if State2>0:
                                    if StateP2>=4:
                                        if   ww2_4>0:                         State2_,ww2 = State4C2()
                                        else:                                 State2_,ww2 = State3C2(); State2_= 5 # crack closure, state 5
                                    else:
                                        if   ww2_1>self.wcr:                  State2_,ww2 = State4C2()  # loading beyond critical crack width
                                        elif epsL[1]>epsLP[1]+TOL and ww2_1>wP2:  State2_,ww2 = State1C2() # state 1 loading
                                        elif ww2_2>0:                         State2_,ww2 = State2C2() # state 2 unloading
                                        elif ww2_2<=0:                        State2_,ww2 = State3C2() # state 3 crack closure
                                        else: raise NameError ("ConFemMaterials::ElasticLT.sig: crack exception 2")
                                else:                                         State2_ = 0
                            else:                                             State_, ww1 = State3C(); State_= 5 # crack closure, state 5
                        else:
                            if ww1_1>self.wcr:                                State_, ww1 = State4C()  # starting state 4 loading beyond critical crack width
                            elif epsL[0]>epsLP[0]+TOL and ww1_1>wP1:          State_, ww1 = State1C(0, ww1_1)# state 1 loading in tensile range
                            elif ww1_2>0:                                     State_, ww1 = State2C()  # state 2 unloading
                            elif ww1_2<=0:                                    State_, ww1 = State3C()  # state 3 crack closure in compressive range
                            else:
                                print elI,ipI,Eps_,epsL,self.wcr,State,wP1,ww1_1,ww1_2,ww1_4
                                if ff<>None: print >> ff, elI,ipI,Eps_,epsL,self.wcr,State,wP1,ww1_1,ww1_2,ww1_4
                                raise NameError ("ConFemMaterials::ElasticLT.sig: crack exception")
                        # finish 
                        Elem.StateVarN[ipI,0] = State_
                        Elem.StateVarN[ipI,6] = State2_
                        MatM  = dot(Trans.transpose(),dot(MatML,Trans))     # transformation of tangent stiffness into global system
                        sig = dot(Trans.transpose(),sigL)       # transformation of local stress into global system
                        if Elem.dim==21:
                            cc = 0.5*Emod/(2.+0.*nu)            # zero poisson's ratio assumed for out of plane shear
                            sig_ = array([sig[0],sig[1],0.,cc*Eps[3],cc*Eps[4],sig[2]])
                            MatM_ = array([[MatM[0,0], MatM[0,1], 0., 0., 0., MatM[0,2]],
                                           [MatM[1,0], MatM[1,1], 0., 0., 0., MatM[1,2]],
                                           [0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., cc, 0., 0.],
                                           [0., 0., 0., 0., cc, 0],
                                           [MatM[2,0], MatM[2,1], 0., 0., 0., MatM[2,2]]])
#                        Elem.StateVarN[ipI,1] = sigL[0]         # larger principal stress uhc
                        Elem.StateVarN[ipI,1] = sigL[1]         # uhc larger principal stress, direction 1 already out of the game
                        Elem.StateVarN[ipI,11] = sigL[0]        # (principal) stress belonging to 1st larger principal strain
                        Elem.StateVarN[ipI,12] = sigL[1]        # (principal) stress belonging to 2nd principal strain
                    else:   # uncracked state
                        if Elem.dim==2:
                            xxx = Emod/(1-nu**2)
                            MatM = array([[xxx,xxx*nu,0],[xxx*nu,xxx,0],[0,0,0.5*xxx*(1-nu)]]) # isotropic linear elastic plane stress
                            sig = dot(MatM,Eps_) # stress
                        elif Elem.dim==21:
                            MatM_ = array([[Emod/(1-nu**2), nu*Emod/(1-nu**2), 0., 0., 0., 0.],
                                           [nu*Emod/(1-nu**2), Emod/(1-nu**2), 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., Emod/(2+2*nu), 0., 0.],
                                           [0., 0., 0., 0., Emod/(2+2*nu), 0],
                                           [0., 0., 0., 0., 0., Emod/(2+2*nu)]]) 
                            sig_ = dot(MatM_,Eps)
                            sig = array([sig_[0],sig_[1],sig_[5]])                     #
                        pig,phi,pig_ = PrinCLT_( sig[0], sig[1], sig[2]) # principal stresses, larger value, corr. direction, lower value
                        Elem.StateVarN[ipI,1] = pig             # larger 1st principal stress
                        Elem.StateVarN[ipI,11] = pig            # "
                        Elem.StateVarN[ipI,12] = pig_           # 2nd prinicipal stress
                    if Elem.dim==2:
                        return sig, MatM, [ Eps_[0], Eps_[1], Eps_[2], sig[0], sig[1], sig[2], ww1, ww2]
                    elif Elem.dim==21: 
                        return sig_, MatM_, [Eps_[0],Eps_[1],Eps_[2],sig[0],sig[1],sig[2],ww1,ww2,sig_[0],sig_[1],sig_[2],sig_[3],sig_[4],sig_[5]]
#                        return sig_, MatM_, [sig_[1],sig_[2],sig_[3],sig_[4],sig_[5],Eps_[0],Eps_[1],Eps_[2],sig[0],sig[1],sig[2],ww1,ww2,sig_[0]]
                # C-Version
                else:
                    sig  = zeros((6),dtype=float)
                    MatM_ = zeros((36),dtype=float)
                    ww   = zeros((9),dtype=float)   # dimensioned not only for ww but also to channel other values for control purposes
                    rc = ElasticLTC1( Elem.dim, Elem.PlSt, Elem.Lch_, Elem.StateVar[ipI], Elem.StateVarN[ipI], Elem.DataP[ipI],\
                                  self.Emod, self.nu, self.epsct, self.wcr, self.bw, self.CrackVisc, self.fct, Eps, sig, MatM_, ww)
                    if rc<>0: raise NameError("ConFemMaterials::ElasticLT:sig:ElasticLTC1 RC "+str(rc))
                    if Elem.dim==2:
                        MatM = array([[MatM_[0],MatM_[1],MatM_[2]],[MatM_[3],MatM_[4],MatM_[5]],[MatM_[6],MatM_[7],MatM_[8]]])
                        return [sig[0], sig[1], sig[2]], MatM, [ Eps[0], Eps[1], Eps[2], sig[0], sig[1], sig[2], ww[0], ww[1]]
                    elif Elem.dim==21:
                        MatM = array([[MatM_[0], MatM_[1], MatM_[2], MatM_[3], MatM_[4], MatM_[5]],\
                                      [MatM_[6], MatM_[7], MatM_[8], MatM_[9], MatM_[10],MatM_[11]],\
                                      [MatM_[12],MatM_[13],MatM_[14],MatM_[15],MatM_[16],MatM_[17]],\
                                      [MatM_[18],MatM_[19],MatM_[20],MatM_[21],MatM_[22],MatM_[23]],\
                                      [MatM_[24],MatM_[25],MatM_[26],MatM_[27],MatM_[28],MatM_[29]],\
                                      [MatM_[30],MatM_[31],MatM_[32],MatM_[33],MatM_[34],MatM_[35]]])
#                        return sig, MatM, [Eps[0],Eps[1],Eps[5],sig[0],sig[1],sig[5],ww[0],ww[1],sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]]
                        return sig, MatM, [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5],Eps[0],Eps[1],Eps[5],sig[0],sig[1],sig[5],ww[0],ww[1]]
    def UpdateStateVar(self, Elem, ff):     # update integration point's state variables 0,1,2,3,4,5,6,7,8
        if Elem.Type=='SH4': 
            if   Elem.nInt==2 :nn = min(Elem.StateVar.shape[0],16) # loop over integration and integration sub points excluding reinforcement with SH4 elements
            elif Elem.nInt==5 :nn = min(Elem.StateVar.shape[0],20)
        else:                  nn = Elem.StateVar.shape[0]    # loop over integration points
        SFlag = False
        for j in xrange(nn):
            if (Elem.StateVar[j][0]!=Elem.StateVarN[j][0]): 
                print                     'El', Elem.Label, j, 'ElasticLT state change', Elem.StateVar[j][0],'->',Elem.StateVarN[j][0]
                if ff<>None: print >> ff, 'El', Elem.Label, j, 'ElasticLT state change', Elem.StateVar[j][0],'->',Elem.StateVarN[j][0]
                SFlag = True
            if (Elem.StateVar[j][6]!=Elem.StateVarN[j][6]): 
                print 'El', Elem.Label, j, 'ElasticLT state 2 change', Elem.StateVar[j][6],'->',Elem.StateVarN[j][6]
                if ff<>None: print >> ff, 'El', Elem.Label, j, 'ElasticLT state 2 change', Elem.StateVar[j][6],'->',Elem.StateVarN[j][6]
                SFlag = True
            for k in (0,1,4,5,6): Elem.StateVar[j,k] = Elem.StateVarN[j,k]
            if Elem.StateVarN[j,2]>Elem.StateVar[j,2]: 
                Elem.StateVar[j,2] = Elem.StateVarN[j,2]    # maximum crack width ever reached
                Elem.StateVar[j,3] = Elem.StateVarN[j,3]    # related crack traction
            if Elem.StateVarN[j,7]>Elem.StateVar[j,7]: 
                Elem.StateVar[j,7] = Elem.StateVarN[j,7]    # principal strain direction 
                Elem.StateVar[j,8] = Elem.StateVarN[j,8]    # principal strain direction
        return SFlag
    def UpdateStat2Var(self, Elem, ff, SFlag, LoFl):
        if Elem.Type=='SH4': 
            if   Elem.nInt==2 :nn = min(Elem.StateVar.shape[0],16) # loop over integration and integration sub points excluding reinforcement with SH4 elements
            elif Elem.nInt==5 :nn = min(Elem.StateVar.shape[0],20)
        else:                  nn = Elem.StateVar.shape[0]  # loop over integration points
        if LoFl or SFlag:                                   # LoFl: flag for loading increment, SFlag
            for j in xrange(nn):
                for k in (9,11):  Elem.StateVar[j,k] = Elem.StateVarN[j,k]   # principal strain, stress of 0 direction
                for k in (10,12): Elem.StateVar[j,k] = Elem.StateVarN[j,k]   # principal strain, stress of 1 direction
        
class IsoDamage(Material):                                  # isotropic damage
    """
    Material parameters input : E0, nu, fC, fCt, LimTY, alpha_biax, r_confined, alpha_confined, RegTY, R, neta_artificial
    E0 = initial Young's modulus;   nu = Poisson's ratio
    fC, fCt = uniaxial compressive/tensile strength
    LimTY = type of triaxial strength surface. Default choose = 1
    alpha_biax = biaxial compressive strength related to uniaxial compressive strength. Recommend = 1.2
    r_confined = circumferential stress related to longitudinal stress (compression) of confined cylinder specimen
    alpha_confined = longitudinal strength related uniaxial compressive strength cylinder specimen
    RegTY = regularization type. 0- no regulation, 1- Gradient damage, might not work with all element types
                                 2- Crack band - might not work with all element types
    R = characteristic length for RegTY= 1; or crack energy for RegTY=2
    neta_artificial = artificial viscosity - this might sometimes be necessary for convergence (trial and error!)

    Some care has to be taken in choosing the relation of fc-E0; fc-fct; r_confined-alpha_confined and the value of alpha_biax. These should be referenced in MC2010.
    """
    def __init__(self, PropMat):
#                        (self, SymmetricVal, RTypeVal, UpdateVal, Updat2Val, StateVarVal, NDataVal):
        Material.__init__(self, False, PropMat[8], True, False, 9, 8)
#        self.Symmetric = False                             # flag for symmetry of material matrices
#        self.RType = PropMat[4]                            # type of regularization 0: without, 1: gradient 2: crack band
#        self.Update = True                                 # has specific update procedure for update of state variables
#        self.StateVar = 9                                  # number of state variables [0] damage [1] equivalent damage strain, [2-7]strain rate of previous time step (voigt?), [8] crack energy
#        self.NDataVal = 8
        self.PropMat= PropMat                               # for wrappers, e.g. RCSHELL
        self.Emod= PropMat[0]
        self.nu  = PropMat[1]
        self.LiTy= PropMat[4]                               # type of limit function
        self.EvalPar()                                      # uses PropMat[2,3,5,6,7]
        a, b = self.cc0*(1.+self.nu)**2/3., self.cc1*(1.+self.nu)/sqrt(3.) + self.cc2 + self.cc3*(1.-2.*self.nu) #auxiliary values
        self.alpha = 0.5*b + sqrt(0.25*b**2+a)              # scaling factor for tension
        self.RegPar = PropMat[9]                            # regularization parameter;  char. length for gradient regularization RegType 1
                                                            #                            crack energy for crack band approach RegType 2
        self.eta = PropMat[10]                              # artificial viscosity
        self.svrTol = 0.01                                  # minimum reference value for viscous stresses - for control computations only
        self.Density = PropMat[11]                          # specific mass
        self.kapStrength = 0.5*(self.edt+sqrt(self.edt**2+2.*self.ed**2)) # equivalent strain for uniaxial compressive strength
        self.dDestr   = 1.-1.e-6                            # maximum destruction. Damage should not be zero to avoid numerical singularity
        self.kapUlt   = exp(log(-log(1.-self.dDestr))/self.gd)*self.ed+self.edt
        self.fc       = self.Emod*exp(-pow(( self.kapStrength - self.edt)/self.ed ,self.gd)) * self.kapStrength
        self.fct      = self.fc/self.alpha
        self.eps_ct   = self.kapStrength/self.alpha         # strain of uniaxial tensile strength
        self.gam2     = 350.                                # parameter for scaling of equivalent damage strain
        self.SpecCrEn = self.SpeCrEnergy( 1.0, self.eps_ct) # specific crack energy unscaled
        if self.RType==2: self.bw = self.RegPar/self.SpecCrEn 
        else:             self.bw = 0.
        self.CrX, self.CrY = self.SpeCrEnergyScaled( 0.01, 100., self.CrBwN, self.eps_ct) # arrays to determine scaling factor for given characteristic element length
        print 'IsoDamage', self.SpecCrEn, self.fc, self.fct
        nu = self.nu
        ff = self.Emod / ( ( 1. + nu ) * ( 1. - 2.*nu ) )
        self.C3_ = ff*array([[1.-nu,nu,nu,0,0,0],[nu,1.-nu,nu,0,0,0],[nu,nu,1.-nu,0,0,0],[0,0,0,(1.-2.*nu)/2.,0,0],[0,0,0,0,(1.-2.*nu)/2.,0],[0,0,0,0,0,(1.-2.*nu)/2.]]) # linear elastic stiffness matrix
    def EvalPar(self):  # use ProMat [2,3,5,6,7] to compute constant material coef.s
        ll   = exp(-1./2.)
        fC = self.PropMat[2]
        fT = self.PropMat[3]
        epsC = self.PropMat[2]/self.Emod/ll 
        self.gd = 2.0
        self.edt= epsC*(2.*log(fC/self.Emod/epsC)+1.)
        self.ed = 2.*sqrt(-log(fC/self.Emod/epsC))*epsC
        alpha = self.PropMat[3]/self.PropMat[2]             # fT/fC 
        beta = self.PropMat[5]                              #    1.2 
        ga = self.PropMat[6]                                #    0.2 
        a3 = self.PropMat[7]                                #    2.0
        A = -3./2.*(alpha*ga*beta-beta*a3*alpha+beta*alpha-ga*alpha+ga*beta)/alpha/(-ga+2.*a3*ga-a3**2-ga**2+a3-ga*alpha+ga*beta)/beta
        B = -1./3.*(-ga*beta**2*alpha+2*alpha*beta**2-2*a3*beta**2*alpha+3*ga*beta**2+4*ga*alpha**2*beta+3*ga**2*beta*alpha+2*a3*ga*beta-ga**2*beta-6*a3*ga*beta*alpha-3*beta*alpha+3*a3**2*beta*alpha+beta*a3-4*ga*beta-a3**2*beta+alpha**2*beta-alpha**2*beta*a3+4*a3*ga*alpha+ga*alpha-2*ga**2*alpha-2*a3**2*alpha-3*ga*alpha**2+2*a3*alpha)/alpha/beta*sqrt(6.)/(ga*alpha-2*a3*ga+ga+ga**2-a3+a3**2-ga*beta) 
        C =  1./2.*(-ga*beta**2-alpha*beta**2+4*a3*ga*beta*alpha+2*beta*alpha-2*a3*ga*alpha-2*a3**2*beta*alpha+a3*beta**2*alpha+ga*beta**2*alpha-2*ga**2*beta*alpha-2*ga*alpha**2*beta-ga*alpha+a3**2*alpha+ga**2*alpha-a3*alpha+ga*alpha**2+2*ga*beta)/alpha/beta*sqrt(6.)/(-ga+2*a3*ga-a3**2-ga**2+a3-ga*alpha+ga*beta)
        D = -1./3.*(-a3*beta**2*alpha+alpha*beta**2+ga*beta**2*alpha+a3**2*beta+alpha**2*beta*a3-beta*a3-alpha**2*beta+ga*beta-2*a3*ga*beta-ga*alpha**2*beta+ga**2*beta+2*a3*ga*alpha+a3*alpha-ga*alpha-a3**2*alpha-ga**2*alpha)*sqrt(3.)/alpha/(-ga+2*a3*ga-a3**2-ga**2+a3-ga*alpha+ga*beta)/beta
        nu = self.nu
        # Hsieh-Ting-Chen's 4 coef. damage function
        self.cc0 = 2.*A/(1.+nu)**2
        self.cc1 = C*sqrt(2.)/(1+nu)
        self.cc2 = B*sqrt(1.5)/(1+nu)
        self.cc3 = D/sqrt(3.)/(1-2*nu)-B/sqrt(6.)/(1+nu)
    def UniaxTensionScaled(self, gam1, eps):
        x = self.alpha*eps-self.kapStrength                 # tensile strain delta after tensile strength strain
        kap_ =gam1*x + (1.-gam1)/self.gam2 * (1.-exp(-self.gam2*x)) + self.kapStrength
        return self.Emod*exp(-pow(( kap_ - self.edt)/self.ed ,self.gd)) * eps
    def EquivStrain1(self, I1, J2, J2S, Eps, EpsD, ipI):    # evaluate variables derived from Hsieh-Ting-Chen's damage function
        la,v = eigh(Eps)                                    # principal values and eigenvectors of strain tensor
        ila = 0                                             # largest principal value
        if la[1]>la[0]: ila=1                       
        if la[2]>la[1] and la[2]>la[0]: ila=2
        laMax = la[ila]
        xxx = 0.25*pow((self.cc1*J2S+self.cc2*laMax+self.cc3*I1),2.0) + self.cc0*J2
        if xxx<0.: raise NameError("ConFemMaterials::IsoDamage.EquivStrain1: inconsistent state")
        kap_ = 0.5*(self.cc1*J2S+self.cc2*laMax+self.cc3*I1)+(sqrt(xxx)) # equivalent damage strain
        if J2S>ZeroD:
            eins = array([1.,1.,1.,0.,0.,0.])               # auxiliary vector
            nd_ = array([v[0,ila]*v[0,ila],v[1,ila]*v[1,ila],v[2,ila]*v[2,ila],v[1,ila]*v[2,ila],v[0,ila]*v[2,ila],v[0,ila]*v[1,ila]]) # corresponding to Voigt notation
            nd  = (self.cc0+self.cc1*kap_/(2.0*J2S))*EpsD + self.cc2*kap_*nd_ +self.cc3*kap_*eins # gradient of damage function dF/depsilon
            Hd = -( self.cc1*J2S + self.cc2*laMax + self.cc3*I1 -2*kap_ ) # H_d: dF/dkappa local
        else:
            nd = zeros((6))
            nd_ = zeros((6))
            Hd = 1
        return kap_, nd, Hd, laMax
    def EquivStrain2(self, I1, J2):     # not used yet
        kap_ = self.k0 * I1 + sqrt( (self.k1 * I1)**2 + self.k2*J2 )
        Hd = 1.                             #
        xx = sqrt((self.k1*I1)**2+self.k2*J2)
        if xx>ZeroD: nd = (self.k0 + self.k1**2*I1/xx ) * eins + 0.5*self.k2/xx*EpsD #
        else:        nd = zeros((6),dtype=float)
        return kap_, nd, Hd
        
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps_, Eps_, dTmp, Temp, EpsR):
        if CalcType == 0: return [], [], []
        Emod = self.Emod
        nu = self.nu

        if not ConFemMatCFlag: # False: # flag for not C-version
            D = Elem.StateVar[ipI,0]                        # D of last converged load / time increment
            kapOld = Elem.StateVar[ipI,1]                   # kappa of last converged load / time increment
            if Elem.dim==1:
                Eps = array([[Eps_[0],0,0],[0,-self.nu*Eps_[0],0],[0,0,-self.nu*Eps_[0]]]) # tensor notation
                Eps__ = array([Eps[0,0],Eps[1,1],Eps[2,2],0,0,0]) # voigt notation
                Dps__ = array([Dps_[0],-self.nu*Dps_[0],-self.nu*Dps_[0],0,0,0]) # voigt notation
            elif Elem.dim==2:
                if Elem.PlSt:
                    alpha = self.nu/(1-self.nu)
                    Eps = array([[Eps_[0],0.5*Eps_[2],0],[0.5*Eps_[2],Eps_[1],0],[0,0,-alpha*(Eps_[0]+Eps_[1])]])# plane stress --> strain tensor 
                    Eps__ = array([Eps_[0],Eps_[1],-self.nu*(Eps_[0]+Eps_[1])/(1-self.nu),0.,0.,Eps_[2]]) # plane stress --> strain Voigt notation
                    Dps__ = array([Dps_[0],Dps_[1],-self.nu*(Dps_[0]+Dps_[1])/(1-self.nu),0.,0.,Dps_[2]]) # plane stress --> strain Voigt notation
                else:         
                    Eps = array([[Eps_[0],0.5*Eps_[2],0],[0.5*Eps_[2],Eps_[1],0],[0,0,0]]) # plane strain --> strain tensor
                    Eps__ = array([Eps_[0],Eps_[1],0.,0.,0.,Eps_[2]]) # plane strain --> strain Voigt notation
                    Dps__ = array([Dps_[0],Dps_[1],0.,0.,0.,Dps_[2]]) # plane strain --> strain Voigt notation
            elif Elem.dim==3: 
                Eps = array([[Eps_[0],0.5*Eps_[5],0.5*Eps_[4]],[0.5*Eps_[5],Eps_[1],0.5*Eps_[3]],[0.5*Eps_[4],0.5*Eps_[3],Eps_[2]]]) # triaxial strain tensor
                Eps__ = array([Eps_[0],Eps_[1],Eps_[2],Eps_[3],Eps_[4],Eps_[5]])                            # triaxial strain Voigt notation
                Dps__ = array([Dps_[0],Dps_[1],Dps_[2],Dps_[3],Dps_[4],Dps_[5]])                            # triaxial strain Voigt notation
            elif Elem.dim==21:                          # continuum bases shell
                xxx = -self.nu/(1-self.nu)*(Eps_[0]+Eps_[1])
                Eps = array([[Eps_[0],0.5*Eps_[5],0.5*Eps_[4]],[0.5*Eps_[5],Eps_[1],0.5*Eps_[3]],[0.5*Eps_[4],0.5*Eps_[3],xxx]]) # triaxial strain tensor
                Eps__ = array([Eps_[0],Eps_[1],xxx,Eps_[3],Eps_[4],Eps_[5]])                                # triaxial strain Voigt notation
                xxx = -self.nu/(1-self.nu)*(Dps_[0]+Dps_[1])
                Dps__ = array([Dps_[0],Dps_[1],xxx,Dps_[3],Dps_[4],Dps_[5]])                            # triaxial strain Voigt notation
            else: raise NameError("ConFemMaterials::IsoDamage.Sig: element type not implemented for this material")

            I1=Eps[0,0]+Eps[1,1]+Eps[2,2]                   # 1st invariant strain tensor
            pp=I1/3.                                        # volumetric strain
            EpsD = array([Eps[0,0]-pp,Eps[1,1]-pp,Eps[2,2]-pp,Eps[1,2],Eps[0,2],Eps[0,1]]) # deviatoric strains tensor components
            J2 =0.5*(EpsD[0]*EpsD[0]+EpsD[1]*EpsD[1]+EpsD[2]*EpsD[2])+EpsD[3]*EpsD[3]+EpsD[4]*EpsD[4]+EpsD[5]*EpsD[5]# 2nd invariant of deviator
            J2S=sqrt(J2)
            J3=EpsD[0]*EpsD[1]*EpsD[2]-EpsD[0]*EpsD[4]**2-EpsD[3]**2*EpsD[2]+2*EpsD[3]*EpsD[5]*EpsD[4]-EpsD[5]**2*EpsD[1] # 3rd invariant of strain deviator
            if J2S>ZeroD: xi = 0.5 * (StrNum0*J3/sqrt(J2**3) + 1.)    # tension indicator
            else:         xi = -1.
            if self.LiTy==1: kap_, nd, Hd, laMax = self.EquivStrain1( I1, J2, J2S, Eps, EpsD, ipI)   # compute kap_=equiv damage strain, nd=dF/de (vector), Hd= dF/dkappa(scarlar), eigenvalue related to e1
            else: raise NameError("ConFemMaterials::IsoDamage.Sig: unknown type of limit function")
            
            if self.RType==1:
                if Elem.dim==  1: kap = EpsR[1]
                elif Elem.dim==2: kap = EpsR[3]
                elif Elem.dim==3: kap = EpsR[6]
                dkk = 1.    # will be used later for regularization
            elif self.RType==2:
                if kap_> self.kapStrength:
                    beta = Elem.CrBwS                       # element specific scaling factor
                    kap = beta*(kap_-self.kapStrength) + (1.-beta)/self.gam2 * (1-exp(-self.gam2*(kap_-self.kapStrength))) + self.kapStrength # scaled damage strain
                    dkk = beta                         + (1.-beta)           *    exp(-self.gam2*(kap_-self.kapStrength)) # scaling factor for tangential material stiffness
                else: 
                    kap = kap_
                    dkk = 1. 
            else:                 
                kap = kap_
                dkk = 1.
            if kap>self.kapUlt: kap=self.kapUlt             # Damage should not be zero to avoid numerical singularity. This constrains D to dDestr
            #
            sig0  = dot(self.C3_,Eps__)  # Elastic predictor stress
            #
            if kap>self.edt and kap>=kapOld and J2S>ZeroD:  # case loading with nonzero strain deviator
                D = 1.0 - exp(-pow(( kap-self.edt)/self.ed ,self.gd))  # scalar damage
                hdI = pow(( kap-self.edt)/self.ed ,self.gd)*self.gd/(kap-self.edt)*exp(-pow((kap-self.edt)/self.ed,self.gd)) * dkk # dD/dkappa
                sig0  = dot(self.C3_,Eps__)
                if self.RType==1: CD = zeros((6,6))         # hdI/Hd * outer(sig0,nd) #  # tangential material stiffness loading (voigt for Eps__ should be correct as C3_*Eps__ is a stress)
                else:             CD = hdI/Hd * outer(sig0,nd)
            else: 
                CD = zeros((6,6),dtype=float)               # case of unloading or zero strain deviator
                nd = zeros((6))
                Hd = 1
                hdI = 0
            Elem.StateVarN[ipI,0] = D                       # store damage of actual iteration
            Elem.StateVarN[ipI,1] = kap                     # store equivalent damage strain of actual iteration
            if self.eta>0.:                                 # viscous regularization
                zz, Veps = self.ViscExten3D( Dt, self.eta, Dps__, Elem, ipI, 2)
            else:
                zz = 0.
                Veps = zeros((6),dtype=float)
            CC = (1-D)*self.C3_ - CD                        # triaxial tangential material stiffness
            for k in xrange(6): CC[k,k] = CC[k,k] + zz      # viscous extension for tangential material stiffness
            sigV = self.eta*Veps
            sig = (1-D)*dot(self.C3_,Eps__)
            svs = 0.
            for i in xrange(6):                             # to record size of viscous stress 
                if abs(sig[i])>self.svrTol: xxx = sigV[i]/sig[i]
                else:                       xxx = 0.
                if abs(xxx)>abs(svs): svs = xxx         
            sig = sig  + sigV                               # triaxial stress with viscous extension
            if kap>self.kapStrength and kap>kapOld: Elem.StateVarN[ipI,8] = dot(sig,Dps__) # Crack energy increment for step
            else:                                   Elem.StateVarN[ipI,8] = 0.
            # Returning
            if Elem.dim==1:
                if self.RType==1:
                    ccc = 0.5*self.RegPar**2
                    fact = 0.5*self.Emod
                    nd[0]=nd[0]-self.nu*nd[1]-self.nu*nd[2]
                    CR = array([[0,-self.Emod*Eps_[0]*hdI],[-fact*nd[0]/Hd,fact]])
                    return [sig[0],ccc*Eps_[1]*fact], array([[CC[0,0],0],[0,ccc*fact]]), [0,(EpsR[1]-kap_)*fact], CR, [Eps_[0], sig[0]]
                else:
                    return [sig[0],0], array([[CC[0,0],0],[0,0]]), [Eps_[0], sig[0]]
            elif Elem.dim==2:
                if self.RType==1:
                    ccc = 0.5*self.RegPar**2
                    fact = 0.5*self.Emod
                    if Elem.PlSt:                           # plane stress
                        CC_= array([[CC[0,0]-CC[0,2]*CC[2,0]/CC[2,2],CC[0,1]-CC[0,2]*CC[2,1]/CC[2,2],CC[0,5]-CC[0,2]*CC[2,5]/CC[2,2], 0, 0],
                                    [CC[1,0]-CC[1,2]*CC[2,0]/CC[2,2],CC[1,1]-CC[1,2]*CC[2,1]/CC[2,2],CC[1,5]-CC[1,2]*CC[2,5]/CC[2,2], 0, 0], 
                                    [CC[5,0]-CC[5,2]*CC[2,0]/CC[2,2],CC[5,1]-CC[5,2]*CC[2,1]/CC[2,2],CC[5,5]-CC[5,2]*CC[2,5]/CC[2,2], 0, 0],
                                    [                              0,                              0,                              0, ccc*fact,0],
                                    [                              0,                              0,                              0, 0, ccc*fact]])
                        CR = array([[                           0,                           0,             0, -sig0[0]*hdI],
                                    [                           0,                           0,             0, -sig0[1]*hdI],
                                    [                           0,                           0,             0, -sig0[5]*hdI],
                                    [-fact*(nd[0]-alpha*nd[2])/Hd,-fact*(nd[1]-alpha*nd[2])/Hd,-fact*nd[5]/Hd, fact]])
                    else:                                   # plane strain
                        CC_= array([[CC[0,0],CC[0,1],CC[0,5], 0, 0],
                                    [CC[1,0],CC[1,1],CC[1,5], 0, 0], 
                                    [CC[5,0],CC[5,1],CC[5,5], 0, 0],
                                    [      0,      0,      0, ccc*fact, 0],
                                    [      0,      0,      0, 0, ccc*fact]])
                        CR = array([[             0,             0,             0, -sig0[0]*hdI],
                                    [             0,             0,             0, -sig0[1]*hdI],
                                    [             0,             0,             0, -sig0[5]*hdI],
                                    [-fact*nd[0]/Hd,-fact*nd[1]/Hd,-fact*nd[5]/Hd, fact]])
                    return [sig[0],sig[1],sig[5],ccc*Eps_[3]*fact,ccc*Eps_[4]*fact], CC_, [0,0,0,(EpsR[3]-kap_)*fact], CR, [Eps_[0], Eps_[1], Eps_[2], sig[0], sig[1], sig[2]]                            
                else:
                    if Elem.PlSt: CC_= array([[CC[0,0]-CC[0,2]*CC[2,0]/CC[2,2],CC[0,1]-CC[0,2]*CC[2,1]/CC[2,2],CC[0,5]-CC[0,2]*CC[2,5]/CC[2,2]],
                                              [CC[1,0]-CC[1,2]*CC[2,0]/CC[2,2],CC[1,1]-CC[1,2]*CC[2,1]/CC[2,2],CC[1,5]-CC[1,2]*CC[2,5]/CC[2,2]], 
                                              [CC[5,0]-CC[5,2]*CC[2,0]/CC[2,2],CC[5,1]-CC[5,2]*CC[2,1]/CC[2,2],CC[5,5]-CC[5,2]*CC[2,5]/CC[2,2]]])
                    else: 
                        CC_= array([[CC[0,0],CC[0,1],CC[0,5]],[CC[1,0],CC[1,1],CC[1,5]],[CC[5,0],CC[5,1],CC[5,5]]])
                    return [sig[0],sig[1],sig[5]], CC_, [Eps__[0], Eps__[1], Eps__[2], Eps__[5], sig[0], sig[1], sig[2], sig[5]]
            elif Elem.dim==3:
                if self.RType==1:
                    ccc = 0.5*self.RegPar**2
                    fact = 0.5*self.Emod
                    CC_= array([[CC[0,0],CC[0,1],CC[0,2],CC[0,3],CC[0,4],CC[0,5],0,0,0],
                                [CC[1,0],CC[1,1],CC[1,2],CC[1,3],CC[1,4],CC[1,5],0,0,0],
                                [CC[2,0],CC[2,1],CC[2,2],CC[2,3],CC[2,4],CC[2,5],0,0,0],
                                [CC[3,0],CC[3,1],CC[3,2],CC[3,3],CC[3,4],CC[3,5],0,0,0],
                                [CC[4,0],CC[4,1],CC[4,2],CC[4,3],CC[4,4],CC[4,5],0,0,0],
                                [CC[5,0],CC[5,1],CC[5,2],CC[5,3],CC[5,4],CC[5,5],0,0,0],
                                [0,      0,      0,      0,      0,      0,      ccc*fact,0,0],
                                [0,      0,      0,      0,      0,      0,      0,ccc*fact,0],
                                [0,      0,      0,      0,      0,      0,      0,0,ccc*fact]])
                    CR = array([[ 0,             0,             0,             0,             0,             0,            -sig0[0]*hdI],
                                [ 0,             0,             0,             0,             0,             0,            -sig0[1]*hdI],
                                [ 0,             0,             0,             0,             0,             0,            -sig0[2]*hdI],
                                [ 0,             0,             0,             0,             0,             0,            -sig0[3]*hdI],
                                [ 0,             0,             0,             0,             0,             0,            -sig0[4]*hdI],
                                [ 0,             0,             0,             0,             0,             0,            -sig0[5]*hdI],
                                [-fact*nd[0]/Hd,-fact*nd[1]/Hd,-fact*nd[2]/Hd,-fact*nd[3]/Hd,-fact*nd[4]/Hd,-fact*nd[5]/Hd, fact]])
                    return [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5],ccc*Eps_[6]*fact,ccc*Eps_[7]*fact,ccc*Eps_[8]*fact], CC_, [0,0,0,0,0,0,(EpsR[6]-kap_)*fact], CR, [sig[0],sig[1],sig[2],Eps[0],Eps[1],Eps[2]]
                else:
                    return sig, CC, [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]]
            elif Elem.dim==21:                              # Continuum based shell
                CC_ = array([[CC[0,0],CC[0,1],CC[0,2],CC[0,3],CC[0,4],CC[0,5]],
                             [CC[1,0],CC[1,1],CC[1,2],CC[1,3],CC[1,4],CC[1,5]],
                             [0., 0., 0., 0., 0., 0.],
                             [CC[3,0],CC[3,1],CC[3,2],CC[3,3],CC[3,4],CC[3,5]],
                             [CC[4,0],CC[4,1],CC[4,2],CC[4,3],CC[4,4],CC[4,5]],
                             [CC[5,0],CC[5,1],CC[5,2],CC[5,3],CC[5,4],CC[5,5]]])
                if Elem.ShellRCFlag: return sig, CC_, [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5], Eps__[0],Eps__[1],Eps__[5], sig[0],sig[1],sig[5], D, svs]
#                if Elem.ShellRCFlag: return sig, CC_, [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5], Eps__[0],Eps__[1],Eps__[5], sig[0],sig[1],sig[5], D,kap_]
                else:                return sig, CC_, [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]]
            else: raise NameError ("ConFemMaterials::Isodam.Sig: not implemented for this element type")
        # C-Version
        else:
            DataOut = zeros((1),dtype=float)
            if Elem.dim==1:
                sig  = zeros((2),dtype=float)
                sigR = zeros((2),dtype=float)
                MatM_= zeros((4),dtype=float)
                CR   = zeros((4),dtype=float)
                PlSt_ = False
            elif Elem.dim==2:
                sig  = zeros((5),dtype=float)
                sigR = zeros((4),dtype=float)
                MatM_= zeros((25),dtype=float)
                CR   = zeros((16),dtype=float)
                PlSt_ = Elem.PlSt
            elif Elem.dim==3:
                sig  = zeros((9),dtype=float)  
                sigR = zeros((7),dtype=float)
                MatM_= zeros((81),dtype=float)
                CR   = zeros((49),dtype=float)
                PlSt_ = False
            elif Elem.dim==21:
                sig  = zeros((6),dtype=float)  
                sigR = zeros((1),dtype=float)
                MatM_= zeros((36),dtype=float)
                CR   = zeros((1),dtype=float)
                PlSt_ = False
            if len(EpsR)==0: EpsR = zeros((1),dtype=float)
            rc = IsoDamC1( Elem.dim, PlSt_, Elem.Lch_, Elem.StateVar[ipI], Elem.StateVarN[ipI],\
                           Eps_, sig, MatM_, self.LiTy, self.cc0, self.cc1, self.cc2, self.cc3, self.RType, EpsR, self.kapStrength,\
                           Elem.CrBwS, self.gam2, self.kapUlt, self.edt, self.ed, self.gd, nu, Emod, Dps_, self.eta, self.RegPar, sigR, CR, Dt, self.svrTol, DataOut)
            D_   = Elem.StateVar[ipI,0]
            kap_ = Elem.StateVar[ipI,1]
###############################                
#            if Elem.Label==50 and ipI==0:
#                print 'ZZZ', Elem.Label, Eps_, sig, '__', sigR #  array([ [MatM_[0],MatM_[1],MatM_[2]],[MatM_[3],MatM_[4],MatM_[5]],[MatM_[6],MatM_[7],MatM_[8]]])
###############################                
            if rc>110:
                raise NameError("ConFemMaterials::IsoDamage:sig:ElasticLTC1 RC "+str(rc)) 
            if Elem.dim==1:
                if self.RType==1:
                    return [sig[0],sig[1]], array([[MatM_[0],MatM_[1]],[MatM_[2],MatM_[3]]]), [sigR[0],sigR[1]], array([[CR[0],CR[1]],[CR[2],CR[3]]]), [Eps_[0], sig[0]]
                else:
                    return [sig[0],0],      array([[MatM_[0],MatM_[1]],[MatM_[2],MatM_[3]]]),                                                          [Eps_[0], sig[0]]
            elif Elem.dim==2:
                if self.RType==1:
                    pass
                else:
                    return ([sig[0],sig[1],sig[2]]), array([ [MatM_[0],MatM_[1],MatM_[2]],[MatM_[3],MatM_[4],MatM_[5]],[MatM_[6],MatM_[7],MatM_[8]]]), [Eps_[0],Eps_[1],0.,Eps_[2],sig[0],sig[1],0.,sig[2]]
            elif Elem.dim==3:
                if self.RType==1:
                    pass
                else:
                    return [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]],\
                    array([[MatM_[0], MatM_[1], MatM_[2], MatM_[3], MatM_[4], MatM_[5] ],
                           [MatM_[6], MatM_[7], MatM_[8], MatM_[9], MatM_[10],MatM_[11]],
                           [MatM_[12],MatM_[13],MatM_[14],MatM_[15],MatM_[16],MatM_[17]],
                           [MatM_[18],MatM_[19],MatM_[20],MatM_[21],MatM_[22],MatM_[23]],
                           [MatM_[24],MatM_[25],MatM_[26],MatM_[27],MatM_[28],MatM_[29]],
                           [MatM_[30],MatM_[31],MatM_[32],MatM_[33],MatM_[34],MatM_[35]]]),\
                           [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]]
            elif Elem.dim==21:
                if Elem.ShellRCFlag:
                    return [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]],\
                    array([[MatM_[0], MatM_[1], MatM_[2], MatM_[3], MatM_[4], MatM_[5] ],
                           [MatM_[6], MatM_[7], MatM_[8], MatM_[9], MatM_[10],MatM_[11]],
                           [MatM_[12],MatM_[13],MatM_[14],MatM_[15],MatM_[16],MatM_[17]],
                           [MatM_[18],MatM_[19],MatM_[20],MatM_[21],MatM_[22],MatM_[23]],
                           [MatM_[24],MatM_[25],MatM_[26],MatM_[27],MatM_[28],MatM_[29]],
                           [MatM_[30],MatM_[31],MatM_[32],MatM_[33],MatM_[34],MatM_[35]]]),\
                           [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5], Eps_[0],Eps_[1],Eps_[5], sig[0],sig[1],sig[5], D_, DataOut[0]]
                else:
                    return [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]],\
                    array([[MatM_[0], MatM_[1], MatM_[2], MatM_[3], MatM_[4], MatM_[5] ],
                           [MatM_[6], MatM_[7], MatM_[8], MatM_[9], MatM_[10],MatM_[11]],
                           [MatM_[12],MatM_[13],MatM_[14],MatM_[15],MatM_[16],MatM_[17]],
                           [MatM_[18],MatM_[19],MatM_[20],MatM_[21],MatM_[22],MatM_[23]],
                           [MatM_[24],MatM_[25],MatM_[26],MatM_[27],MatM_[28],MatM_[29]],
                           [MatM_[30],MatM_[31],MatM_[32],MatM_[33],MatM_[34],MatM_[35]]]),\
                           [sig[0],sig[1],sig[2],sig[3],sig[4],sig[5]]
    def UpdateStateVar(self, Elem, ff):
        for j in xrange(Elem.StateVar.shape[0]):
            if Elem.StateVarN[j,1]>Elem.StateVar[j,1]:          # damage measures
                Elem.StateVar[j,0] = Elem.StateVarN[j,0]
                Elem.StateVar[j,1] = Elem.StateVarN[j,1]
            Elem.StateVar[j,8] = Elem.StateVar[j,8] + Elem.StateVarN[j,8]       # "crack" energy
            Elem.StateVar[j,2] = Elem.StateVarN[j,2]            # strain velocities - used in ViscExten3D
            Elem.StateVar[j,3] = Elem.StateVarN[j,3]
            Elem.StateVar[j,4] = Elem.StateVarN[j,4]
            Elem.StateVar[j,5] = Elem.StateVarN[j,5]
            Elem.StateVar[j,6] = Elem.StateVarN[j,6]
            Elem.StateVar[j,7] = Elem.StateVarN[j,7]
        return False
        
class MicroPlane(Material):                                     # microplane damage
    def __init__(self, PropMat):
#                        (self, SymmetricVal, RTypeVal, UpdateVal, Updat2Val, StateVarVal, NDataVal):
        Material.__init__(self, False, 2, False, False, 1, 6)
#        self.Symmetric = False                     # flag for symmetry of material matrices
#        self.RType = PropMat[4]                    # type of regularization 0: without, 1: gradient 2: crack band
#        self.Update = True                         # has specific update procedure for update of state variables
#        self.StateVar = 9                          # number of state variables [0] damage [1] equivalent damage strain, [2-7]strain rate of previous time step (voigt?), [8] crack energy
#        self.NDataVal = 8
        self.PropMat= PropMat                                   # for wrappers, e.g. RCSHELL
#        self.Update = False # True                               # has no specific update procedure for update of state variables
#        self.Updat2 = False                                     # no 2 stage update procedure
#        self.NData = 6                                          # number of data items
#        self.Symmetric = False                                  # flag for symmetry of material matrices
        self.Type = "MicroPl"

        self.EE = PropMat[0]                                    # macroscopic Young's modulus
        self.nu = PropMat[1]                                    # macroscopic Poissons's ratio
        self.type = PropMat[2]                                  # type of damage function
        #
        self.nInt = len(I21Points)
        self.nState = 1                                         # number of state variables per integration direction of unit sphere
        self.iS = self.nState*self.nInt+5                       # entry index for strain rate for, e.g. viscous extension
        self.StateVar = self.nState*self.nInt+5+6               # number of state variables per integration point of element; 1st extra for dd_iso, 2nd for I1, 3rd for J2, 
                                                                # 4th currently not used ()for previous eps_33, 5th for eps_33 in case of plane state, 6-11 for strain increment
        #
        ff = self.EE / ( ( 1. + self.nu ) * ( 1. - 2.*self.nu ) )
        self.EMat = ff*array([[1.-self.nu,self.nu,self.nu,0,0,0],
                              [self.nu,1.-self.nu,self.nu,0,0,0],
                              [self.nu,self.nu,1.-self.nu,0,0,0],
                              [0,0,0,(1.-2.*self.nu)/2.,0,0],
                              [0,0,0,0,(1.-2.*self.nu)/2.,0],
                              [0,0,0,0,0,(1.-2.*self.nu)/2.]])  # 3D isotropic linear elasticity
        KK = self.EE/(3.*(1.-2.*self.nu))                       # macroscopic bulk modulus
        GG = self.EE/(2.*(1.+self.nu))                          # macroscopic shear modulus
        # V-D split
        self.E_V= 3.*KK                                         # moduli of microplane elasticity V-D split
        self.E_D= 2.*GG
        self.E_DM = array([[self.E_D,0,0],[0,self.E_D,0],[0,0,self.E_D]])
        self.PlStressL = 0.01                                   # limit for plane stress iteration (dim = 2 (plane stress ), 21)
        #
        if self.type == 1:                                      # V-D split Vree / Leukart damage function 
            self.alphaV = PropMat[3] #0.9 #0.96
            self.betaV  = PropMat[4] #3000. #300.
            self.kap0V  = PropMat[5] #0.0001 #0.0005
        elif self.type == 2:                                    # V-D split Vree / uhc damage function
            self.fct= PropMat[3]                                # tensile strength (PropMat[4,5] not used
            self.eps_ct = 1.648721271*self.fct/self.EE          # strain for uniaxial tensile strength to make e_0=0
            self.gd = 2.
            self.e0 = 0.                                        # with eps_ct as above
            self.ed = 2.331643981*PropMat[3]/self.EE            # with eps_ct as above
            self.gam2 = 12.*350.                                # parameter for scaling of equivalent damage strain regularization
            self.SpecCrEn = self.SpeCrEnergy( 1.0, self.eps_ct) # specific crack energy unscaled
            print 'microplane', self.SpecCrEn, self.fct
            self.RType  = PropMat[7]                            # indicator for regularization
            self.RegPar = PropMat[8]                            # crack energy for regularization (RType=2)
            if self.RType==2: self.bw = self.RegPar/self.SpecCrEn 
            else:             self.bw = 0.
            self.CrX, self.CrY = self.SpeCrEnergyScaled( 0.01, 100., self.CrBwN, self.eps_ct) # arrays to determine scaling factor for given characteristic element length
            self.dDestr   = 1.-1.e-3                            # for maximum damage allowed, subtractor should not be small in order to avoid problems in dividing by MMatS[2,2], see below  
            self.kapUlt   = exp(log(-log(1.-self.dDestr))/self.gd)*self.ed+self.e0
        #
        self.kV = PropMat[4]                                    # (measure for) ratio of uniaxial compressive strength to tensile strength
        self.kV0 = (self.kV-1.)/(2.*self.kV*(1.-2.*self.nu))
        self.kV1 = self.kV0
        self.kV2 = 3./(self.kV*(1.+self.nu)**2)
        self.eta = PropMat[9]                                   # for viscous regularization
        self.Density = PropMat[10]
    def UniaxTensionScaled(self, gam1, eps):
        xx = eps - self.eps_ct
        kap_ =gam1*xx + (1.-gam1)/(self.gam2) * (1.-exp(-self.gam2*xx)) + self.eps_ct
        return self.EE * exp(-pow(( kap_-self.e0)/self.ed ,self.gd)) * eps
    def C3(self, Emod, nu):
        ff = Emod / ( ( 1. + nu ) * ( 1. - 2.*nu ) )
        return ff*array([[1.-nu,nu,nu,0,0,0],[nu,1.-nu,nu,0,0,0],[nu,nu,1.-nu,0,0,0],[0,0,0,(1.-2.*nu)/2.,0,0],[0,0,0,0,(1.-2.*nu)/2.,0],[0,0,0,0,0,(1.-2.*nu)/2.]])
    def DamFunc1(self, kapOld, eta):
        kap = max(self.kap0V,kapOld)
        if eta>kap: kap=eta
        dd = 1.-self.kap0V/kap*(1.+self.alphaV*( exp( self.betaV*(self.kap0V-kap) )-1. ))
        if dd>ZeroD: Dp = self.kap0V/kap**2*(1+self.alphaV*(exp(self.betaV*(self.kap0V-kap))-1.))+self.kap0V/kap*self.alphaV*self.betaV*exp(self.betaV*(self.kap0V-kap))
        else:        Dp = 0.
        return kap, dd, Dp
#    def DamFunc2(self, kapOld, eta):
#        kap = max(self.e0,kapOld)
#        if eta>kap: kap=eta
#        if kap<=self.e0: dd = 0.
#        else:            dd = 1.-exp(-((kap-self.e0)/self.ed)**self.gd)
#        if dd>ZeroD:     Dp = (1.-dd) * self.gd/self.ed**self.gd * (kap-self.e0)**(self.gd-1.) 
#        else:            Dp = 0.
#        return kap, dd, Dp
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        def DevStiffness():
            DVD = zeros((6,6),dtype=float)
            DVD[0,0] = m00*n02*tbt2+2.*m01*nn[0]*tbt*nn[1]*obt+2.*m02*nn[0]*obt*nn[2]*tbt+m11*n12*obt2+2.*m12*nn[1]*obt2*nn[2]+m22*n22*obt2
            DVD[0,1] = m00*n02*tbt*obt+m01*nn[0]*tbt2*nn[1]+m02*nn[0]*obt*nn[2]*tbt+m01*nn[0]*obt2*nn[1]+m11*n12*obt*tbt+m12*nn[1]*obt2*nn[2]+m02*nn[0]*obt2*nn[2]+m12*nn[1]*obt*nn[2]*tbt+m22*n22*obt2
            DVD[0,2] = m00*n02*tbt*obt+m01*nn[0]*tbt*nn[1]*obt+m02*nn[0]*tbt2*nn[2]+m01*nn[0]*obt2*nn[1]+m11*n12*obt2+m12*nn[1]*obt*nn[2]*tbt+m02*nn[0]*obt2*nn[2]+m12*nn[1]*obt2*nn[2]+m22*n22*obt*tbt 
            DVD[0,3] = -0.50*m01*nn[0]*tbt*nn[2]-0.50*m02*nn[0]*tbt*nn[1]-0.50*m11*nn[2]*nn[1]*obt-0.50*m12*n12*obt-0.50*m12*n22*obt-0.50*m22*nn[2]*obt*nn[1] 
            DVD[0,4] = -0.50*m00*nn[0]*tbt*nn[2]-0.50*m02*n02*tbt-0.50*m01*nn[2]*nn[1]*obt-0.50*m12*nn[0]*obt*nn[1]-0.50*m02*n22*obt-0.50*m22*nn[0]*obt*nn[2]
            DVD[0,5] = -0.50*m00*nn[0]*tbt*nn[1]-0.50*m01*n12*obt-0.50*m02*nn[2]*nn[1]*obt-0.50*m01*n02*tbt-0.50*m11*nn[0]*obt*nn[1]-0.50*m12*nn[0]*obt*nn[2]
            DVD[1,1] = m00*n02*obt2+2.*m01*nn[0]*tbt*nn[1]*obt+2.*m02*nn[0]*obt2*nn[2]+m11*n12*tbt2+2.*m12*nn[1]*obt*nn[2]*tbt+m22*n22*obt2
            DVD[1,2] = m00*n02*obt2+m01*nn[0]*obt2*nn[1]+m02*nn[0]*obt*nn[2]*tbt+m01*nn[0]*tbt*nn[1]*obt+m11*n12*obt*tbt+m12*nn[1]*tbt2*nn[2]+m02*nn[0]*obt2*nn[2]+m12*nn[1]*obt2*nn[2]+m22*n22*obt*tbt
            DVD[1,3] = -0.50*m01*nn[0]*obt*nn[2]-0.50*m02*nn[0]*obt*nn[1]-0.50*m11*nn[1]*tbt*nn[2]-0.50*m12*n12*tbt-0.50*m12*n22*obt-0.50*m22*nn[2]*obt*nn[1]
            DVD[1,4] = -0.50*m00*nn[0]*obt*nn[2]-0.50*m02*n02*obt-0.50*m01*nn[2]*nn[1]*tbt-0.50*m12*nn[1]*tbt*nn[0]-0.50*m02*n22*obt-0.50*m22*nn[0]*obt*nn[2]
            DVD[1,5] = -0.50*m00*nn[0]*obt*nn[1]-0.50*m01*n12*tbt-0.50*m02*nn[2]*nn[1]*obt-0.50*m01*n02*obt-0.50*m11*nn[1]*tbt*nn[0]-0.50*m12*nn[0]*obt*nn[2]
            DVD[2,2] = m00*n02*obt2+2.*m01*nn[0]*obt2*nn[1]+2.*m02*nn[0]*obt*nn[2]*tbt+m11*n12*obt2+2.*m12*nn[1]*obt*nn[2]*tbt+m22*n22*tbt2
            DVD[2,3] = -0.50*m01*nn[0]*obt*nn[2]-0.50*m02*nn[0]*obt*nn[1]-0.50*m11*nn[2]*nn[1]*obt-0.50*m12*n12*obt-0.50*m12*n22*tbt-0.50*m22*nn[2]*tbt*nn[1]
            DVD[2,4] = -0.50*m00*nn[0]*obt*nn[2]-0.50*m02*n02*obt-0.50*m01*nn[2]*nn[1]*obt-0.50*m12*nn[0]*obt*nn[1]-0.50*m02*n22*tbt-0.50*m22*nn[0]*nn[2]*tbt
            DVD[2,5] = -0.50*m00*nn[0]*obt*nn[1]-0.50*m01*n12*obt-0.50*m02*nn[1]*nn[2]*tbt-0.50*m01*n02*obt-0.50*m11*nn[0]*obt*nn[1]-0.50*m12*nn[0]*nn[2]*tbt
            DVD[3,3] = 0.25*m11*n22+0.50*m12*nn[1]*nn[2]+0.25*m22*n12
            DVD[3,4] = 0.25*m01*n22+0.25*m12*nn[0]*nn[2]+0.25*m02*nn[1]*nn[2]+0.25*m22*nn[0]*nn[1]
            DVD[3,5] = 0.25*m01*nn[1]*nn[2]+0.25*m02*n12+0.25*m11*nn[0]*nn[2]+0.25*m12*nn[0]*nn[1]
            DVD[4,4] = 0.25*m00*n22+0.50*m02*nn[0]*nn[2]+0.25*m22*n02
            DVD[4,5] = 0.25*m00*nn[1]*nn[2]+0.25*m02*nn[0]*nn[1]+0.25*m01*nn[0]*nn[2]+0.25*m12*n02
            DVD[5,5] = 0.25*m00*n12+0.50*m01*nn[0]*nn[1]+0.25*m11*n02
            DVD[1,0] = DVD[0,1]
            DVD[2,0] = DVD[0,2]
            DVD[2,1] = DVD[1,2]
            DVD[3,0] = DVD[0,3]
            DVD[3,1] = DVD[1,3]
            DVD[3,2] = DVD[2,3]
            DVD[4,0] = DVD[0,4]
            DVD[4,1] = DVD[1,4]
            DVD[4,2] = DVD[2,4]
            DVD[4,3] = DVD[3,4]
            DVD[5,0] = DVD[0,5]
            DVD[5,1] = DVD[1,5]
            DVD[5,2] = DVD[2,5]
            DVD[5,3] = DVD[3,5]
            DVD[5,4] = DVD[4,5]
            return DVD
        if CalcType == 0: return [], [], []
        obt, obs, obn, tbt = 1./3., 1./6., 1./9., -2./3.
        obt2 = obt**2
        tbt2 = tbt**2
        VV   = array([[obt,0,0],[0,obt,0],[0,0,obt]])                                       # projection tensor for volumetric part
        ep2  = Elem.StateVarN[ipI,self.nState*self.nInt+4]                                  # value of last iteration taken, not from last converged step
        if Elem.dim==2:
            if Elem.PlSt: 
                epsT = array([ [Eps[0],0.5*Eps[2],0.] , [0.5*Eps[2],Eps[1],0.] , [0.,0.,ep2] ])
                Dps__ = array([ Dps[0], Dps[1], 0., 0., 0., Dps[2]])                        # plane stress --> strain Voigt notation --  used for viscous regularization only with diagonal stiffness
            else:         
                epsT = array([ [Eps[0],0.5*Eps[2],0.] , [0.5*Eps[2],Eps[1],0.] , [0.,0.,0.] ])        #
                Dps__ = array( [Dps[0], Dps[1], 0., 0., 0., Dps[2] ])                       # plane strain --> strain Voigt notation
        elif Elem.dim==3: 
            epsT  = array([ [Eps[0],0.5*Eps[5],0.5*Eps[4]] , [0.5*Eps[5],Eps[1],0.5*Eps[3]] , [0.5*Eps[4],0.5*Eps[3],Eps[2] ]]) # strain tensor arrangement
            Dps__ = array([ Dps[0], Dps[1], Dps[2], Dps[3], Dps[4], Dps[5]])                # triaxial strain increment Voigt notation
        elif Elem.dim==21:                                                                  # continuum based shell
            epsT  = array([ [Eps[0],0.5*Eps[5],0.5*Eps[4]] , [0.5*Eps[5],Eps[1],0.5*Eps[3]] , [0.5*Eps[4],0.5*Eps[3],ep2] ]) # strain tensor arrangement
            Dps__ = array([ Dps[0], Dps[1], 0.,     Dps[3], Dps[4], Dps[5]])                # triaxial strain increment for continuum bases shell Voigt notation
        else: raise NameError("ConFemMaterials::Microplane.Sig: not implemented")
        VVV = zeros((6,6),dtype=float)
        VVV[0,0] = obn
        VVV[0,1] = obn
        VVV[0,2] = obn
        VVV[1,1] = obn
        VVV[1,2] = obn
        VVV[2,2] = obn
        VVV[1,0], VVV[2,0], VVV[2,1] = VVV[0,1], VVV[0,2], VVV[1,2]
        epsVol = VV[0,0]*epsT[0,0]+VV[1,1]*epsT[1,1]+VV[2,2]*epsT[2,2]
        ns = self.nState
        PlStFlag = False
        if (Elem.dim==2 and Elem.PlSt) or Elem.dim==21: PlStFlag = True
        #
        for ii in xrange(10):                                                               # for eventual iteration of plane stress
            dd_iso, I1, J2 = 0., 0., 0.
            MMatS, MMatT, DV1 = zeros((6,6),dtype=float), zeros((6,6),dtype=float), zeros((6,6),dtype=float)
            sig = zeros((3,3),dtype=float)
            for i in xrange(self.nInt):
                kapOld = Elem.StateVar[ipI,ns*i]
                nn = array([I21Points[i,0],I21Points[i,1],I21Points[i,2]])
                # V-D-Split projection tensors
                D0 =     array([[nn[0]    -nn[0]*obt, 0.5*nn[1], 0.5*nn[2]],
                                [0.5*nn[1]         ,-nn[0]*obt , 0.       ],
                                [0.5*nn[2]         , 0.       , -nn[0]*obt]])
                D1 =     array([[-nn[1]*obt         , 0.5*nn[0], 0.       ],
                                [0.5*nn[0], nn[1]-nn[1]*obt    , 0.5*nn[2]],
                                [0.       , 0.5*nn[2]         , -nn[1]*obt]])
                D2 =     array([[-nn[2]*obt        ,       0.  , 0.5*nn[0]],
                                [0.       ,      -nn[2]*obt    , 0.5*nn[1]],
                                [0.5*nn[0], 0.5*nn[1],     nn[2]-nn[2]*obt]])
                # vector deviator strains
                epsDD = array([D0[0,0]*epsT[0,0]+D0[0,1]*epsT[0,1]+D0[0,2]*epsT[0,2]+\
                               D0[1,0]*epsT[1,0]+D0[1,1]*epsT[1,1]+D0[1,2]*epsT[1,2]+\
                               D0[2,0]*epsT[2,0]+D0[2,1]*epsT[2,1]+D0[2,2]*epsT[2,2],\
                               D1[0,0]*epsT[0,0]+D1[0,1]*epsT[0,1]+D1[0,2]*epsT[0,2]+\
                               D1[1,0]*epsT[1,0]+D1[1,1]*epsT[1,1]+D1[1,2]*epsT[1,2]+\
                               D1[2,0]*epsT[2,0]+D1[2,1]*epsT[2,1]+D1[2,2]*epsT[2,2],\
                               D2[0,0]*epsT[0,0]+D2[0,1]*epsT[0,1]+D2[0,2]*epsT[0,2]+\
                               D2[1,0]*epsT[1,0]+D2[1,1]*epsT[1,1]+D2[1,2]*epsT[1,2]+\
                               D2[2,0]*epsT[2,0]+D2[2,1]*epsT[2,1]+D2[2,2]*epsT[2,2]])
                # microplane strain invariants, state variable
                I1mp = 3.*epsVol
                J2mp = 3./2.*dot(epsDD,epsDD)
                eta_ = self.kV0*I1mp + sqrt( (self.kV1*I1mp)**2 + self.kV2*J2mp )
                # damage functions
                if   self.type==1: 
                    kap, dd, Dp = self.DamFunc1( kapOld, eta_ )
                elif self.type==2: 
                    if self.RType==2:                                                           # crack band regularization 
                        if eta_>self.eps_ct:                                                    # limit strain exceeded
                            beta= Elem.CrBwS
                            xx  = eta_ - self.eps_ct
                            eta = beta*xx + (1.-beta)/(self.gam2) * (1.-exp(-self.gam2*xx)) + self.eps_ct
                            dkk = beta +    (1.-beta)             *     exp(-self.gam2*xx)
                        else:                                                                   # below limit strain
                            eta = eta_
                            dkk = 1.
                    else:                                                                       # no regularization
                        eta = eta_
                        dkk = 1.
                    if eta>self.kapUlt: eta = self.kapUlt                                       # to have some residual stiffness
                    # damage function
                    kap = max( self.e0, kapOld, eta)
                    if kap<=self.e0: 
                        dd = 0.
                        Dp = 0.
                    else:            
                        dd = 1.-exp(-((kap-self.e0)/self.ed)**self.gd)
                        Dp = (1.-dd) * self.gd/self.ed**self.gd * (kap-self.e0)**(self.gd-1.) 
                    Dp = Dp*dkk                                                                 # regularization scaling of derivative dD/dkap tangential stiffness
                # stresses
                sigVV = (1.-dd)*    self.E_V*epsVol                                             # volumetric stress (scalar)
                sigDD = (1.-dd)*dot(self.E_DM,epsDD)                                            # deviatoric stress (vector)
                sig   = sig + 6.*I21Weights[i]*(sigVV*VV+sigDD[0]*D0+sigDD[1]*D1+sigDD[2]*D2)   # stress tensor integration # factor 6. calibrates the weighting factors
                # secant material stiffness
                n02 = nn[0]**2
                n12 = nn[1]**2
                n22 = nn[2]**2
                s00, s01, s02 = 1., 1., 1. 
                m00, m11, m22 = 1., 1., 1. 
                m01, m02, m12 = 0., 0., 0.
                DDD = DevStiffness()
                MMatS = MMatS + 6.*I21Weights[i]*( (1.-dd)*self.E_V*VVV + (1.-dd)*self.E_D*DDD ) # presumably symmetric, i.e. only upper right part builded
                # corrector for tangential material stiffness
                if kap>kapOld and dd>ZeroD:
                    s00, s01, s02 = epsDD[0], epsDD[1], epsDD[2]
                    m00, m11, m22 = s00**2,   s01**2,   s02**2
                    m01, m02, m12 = s00*s01,  s00*s02,  s01*s02
                    DDD = DevStiffness()
                    DV1[0,0] = -obt*s00*nn[0]*tbt-obt2*s01*nn[1]-obt2*s02*nn[2] 
                    DV1[0,1] = DV1[0,0] 
                    DV1[0,2] = DV1[0,0]
                    DV1[1,0] = -obt2*s00*nn[0]-obt*s01*nn[1]*tbt-obt2*s02*nn[2]
                    DV1[1,1] = DV1[1,0]
                    DV1[1,2] = DV1[1,0]
                    DV1[2,0] = -obt2*s00*nn[0]-obt2*s01*nn[1]-obt*s02*nn[2]*tbt
                    DV1[2,1] = DV1[2,0]
                    DV1[2,2] = DV1[2,0]
                    DV1[3,0] = obs*s01*nn[2]+obs*s02*nn[1]
                    DV1[3,1] = DV1[3,0]
                    DV1[3,2] = DV1[3,0]
                    DV1[4,0] = obs*s00*nn[2]+obs*s02*nn[0]
                    DV1[4,1] = DV1[4,0]
                    DV1[4,2] = DV1[4,0]
                    DV1[5,0] = obs*s00*nn[1]+obs*s01*nn[0]
                    DV1[5,1] = DV1[5,0]
                    DV1[5,2] = DV1[5,0]
                    xx = sqrt( self.kV1**2 * I1mp**2 + self.kV2 * J2mp )
                    thet =  9.*self.kV1**2/xx
                    psi  =  1.5*self.kV2/xx
                    alph = (3.*self.kV0 + thet*epsVol)*self.E_V*epsVol
                    bet  = (3.*self.kV0 + thet*epsVol)*self.E_D
                    gam  = psi*self.E_V*epsVol
                    delt = psi*self.E_D
                    MMatT = MMatT - 6.*I21Weights[i]*Dp*( alph * VVV + bet * DV1 + gam * transpose(DV1) + delt * DDD )
#                    if Elem.Label == 5: 
#                        print 'XXX', Elem.Label, i, dd, delt
#                        for jj_ in xrange(6):
#                            for kk_ in xrange(6): sys.stdout.write('%14.6e'%(DDD[jj_,kk_]))
#                            sys.stdout.write('\n')
                #            
                dd_iso += 2.*I21Weights[i] * dd
                I1 += 2.*I21Weights[i] * I1mp
                J2 += 2.*I21Weights[i] * J2mp
                # microplane state variable updates
                Elem.StateVarN[ipI,ns*i] = kap
                
            ep2_ = -(MMatS[2,0]*Eps[0]+MMatS[2,1]*Eps[1]+MMatS[2,3]*Eps[3]+MMatS[2,4]*Eps[4]+MMatS[2,5]*Eps[5]) / MMatS[2,2] # lateral normal strain in case of zero lateral stress
            if abs(ep2)>ZeroD*1.e-3: xyz = abs((ep2-ep2_)/ep2)
            else:                    xyz = 0. 
            if not PlStFlag or xyz<self.PlStressL: break
#            elif ii>0:                   print >> ff, 'XXX', Elem.Label, ipI, ii, xyz
            if PlStFlag: epsT[2,2] = ep2_
            ep2 = ep2_

#        I1_ = epsT[0,0]+epsT[1,1]+epsT[2,2]         # 1st invariant of strain tensor
#        epsTD = array([[Eps[0]-I1/3.,0.5*Eps[5],  0.5*Eps[4]],[0.5*Eps[5],   Eps[1]-I1/3.,0.5*Eps[3]],[0.5*Eps[4],   0.5*Eps[3],  Eps[2]-I1/3.]])
#        J2_ = 0.5*(epsTD[0,0]**2+epsTD[1,1]**2+epsTD[2,2]**2 + 2.*epsTD[0,1]**2 + 2.*epsTD[0,2]**2 + 2.*epsTD[1,2]**2)  # 2nd Invariant of strain deviator
#        print 'X', (I1-I1_)/I1, (J2-J2_)/J2
        Elem.StateVarN[ipI,self.nState*self.nInt] = dd_iso
        Elem.StateVarN[ipI,self.nState*self.nInt+1] = I1
        Elem.StateVarN[ipI,self.nState*self.nInt+2] = J2
        Elem.StateVarN[ipI,self.nState*self.nInt+4] = ep2_
        MatM = MMatS + MMatT
        #
        if self.eta>0.: 
            zz, Veps = self.ViscExten3D( Dt, self.eta, Dps__, Elem, ipI, self.iS)           # for viscous regularization
            sig[0,0] = sig[0,0] + self.eta*Veps[0]                                          # Voigt notation
            sig[1,1] = sig[1,1] + self.eta*Veps[1]
            sig[2,2] = sig[2,2] + self.eta*Veps[2]
            sig[1,2] = sig[1,2] + self.eta*Veps[3]
            sig[0,2] = sig[0,2] + self.eta*Veps[4]
            sig[0,1] = sig[0,1] + self.eta*Veps[5]
            for i in xrange(6): MatM[i,i] = MatM[i,i] + zz 
        #
        if Elem.dim==2:                                                                     # 2D
            if Elem.PlSt:                                                                   # plane stress
                cD = 1./MatM[2,2]
                MatM_ =array([[MatM[0,0]-MatM[0,2]*MatM[2,0]*cD,MatM[0,1]-MatM[0,2]*MatM[2,1]*cD,MatM[0,5]-MatM[0,2]*MatM[2,5]*cD],
                              [MatM[1,0]-MatM[1,2]*MatM[2,0]*cD,MatM[1,1]-MatM[1,2]*MatM[2,1]*cD,MatM[1,5]-MatM[1,2]*MatM[2,5]*cD], 
                              [MatM[5,0]-MatM[5,2]*MatM[2,0]*cD,MatM[5,1]-MatM[5,2]*MatM[2,1]*cD,MatM[5,5]-MatM[5,2]*MatM[2,5]*cD]])
            else:                                                                           # plane strain
                MatM_ =array([[MatM[0,0],MatM[0,1],MatM[0,5]],[MatM[1,0],MatM[1,1],MatM[1,5]],[MatM[5,0],MatM[5,1],MatM[5,5]]])
            return array([sig[0,0],sig[1,1],sig[0,1]]), MatM_, [Eps[0], Eps[1], Eps[2], sig[0,0], sig[1,1], sig[0,1]]
        elif Elem.dim==3:                                                                   # 3D
            return array([sig[0,0],sig[1,1],sig[2,2],sig[1,2],sig[0,2],sig[0,1]]), MatM, [sig[0,0],sig[1,1],sig[2,2],sig[1,2],sig[0,2],sig[0,1]]
        elif Elem.dim==21:                                                                  # Continuum based shell
            if abs(MatM[2,2])<ZeroD: 
                print 'X\n', MMatS, '\n', MatM 
                raise NameError("ConFemMaterials::Microplane.Sig: tangential Mat[2,2] to small")
            cD = 1./MatM[2,2]
            MatM_ = array([[MatM[0,0]-MatM[0,2]*MatM[2,0]*cD, MatM[0,1]-MatM[0,2]*MatM[2,1]*cD, 0., MatM[0,3]-MatM[0,2]*MatM[2,3]*cD, MatM[0,4]-MatM[0,2]*MatM[2,4]*cD, MatM[0,5]-MatM[0,2]*MatM[2,5]*cD],
                           [MatM[1,0]-MatM[1,2]*MatM[2,0]*cD, MatM[1,1]-MatM[1,2]*MatM[2,1]*cD, 0., MatM[1,3]-MatM[1,2]*MatM[2,3]*cD, MatM[1,4]-MatM[1,2]*MatM[2,4]*cD, MatM[1,5]-MatM[1,2]*MatM[2,5]*cD],
                           [0.,                               0.,                               0., 0.,                               0.,                               0.],
                           [MatM[3,0]-MatM[3,2]*MatM[2,0]*cD, MatM[3,1]-MatM[3,2]*MatM[2,1]*cD, 0., MatM[3,3]-MatM[3,2]*MatM[2,3]*cD, MatM[3,4]-MatM[3,2]*MatM[2,4]*cD, MatM[3,5]-MatM[3,2]*MatM[2,5]*cD],
                           [MatM[4,0]-MatM[4,2]*MatM[2,0]*cD, MatM[4,1]-MatM[4,2]*MatM[2,1]*cD, 0., MatM[4,3]-MatM[4,2]*MatM[2,3]*cD, MatM[4,4]-MatM[4,2]*MatM[2,4]*cD, MatM[4,5]-MatM[4,2]*MatM[2,5]*cD],
                           [MatM[5,0]-MatM[5,2]*MatM[2,0]*cD, MatM[5,1]-MatM[5,2]*MatM[2,1]*cD, 0., MatM[5,3]-MatM[5,2]*MatM[2,3]*cD, MatM[5,4]-MatM[5,2]*MatM[2,4]*cD, MatM[5,5]-MatM[5,2]*MatM[2,5]*cD]])
            if Elem.ShellRCFlag: return array([sig[0,0],sig[1,1],sig[2,2],sig[1,2],sig[0,2],sig[0,1]]), MatM_, [sig[0,0],sig[1,1],sig[2,2],sig[1,2],sig[0,2],sig[0,1], Eps[0], Eps[1], Eps[2], sig[0,0], sig[1,1], sig[0,1], 0.,0.]
            else:                return array([sig[0,0],sig[1,1],sig[2,2],sig[1,2],sig[0,2],sig[0,1]]), MatM_, [sig[0,0],sig[1,1],sig[2,2],sig[1,2],sig[0,2],sig[0,1]]
        else: raise NameError ("ConFemMaterials::Microplane.Sig: not implemented for this element type")

class Mises(Material):                              # elastoplastic Mises
    def __init__(self, PropMat, val):
#    def __init__(self,         SymmetricVal, RTypeVal, UpdateVal, Updat2Val, StateVarVal, NDataVal):
        Material.__init__(self, True,         None,     True,      False,     5,           8)
#        self.Symmetric = True                       # flag for symmetry of material matrices
#        self.Update = True                          # has specific update procedure for update of state variables
#        self.Updat2 = False                         # no 2 stage update procedure
#        self.StateVar = 5                           # number of state variables per integration point (may in the end be overruled by other types using mises, see RCBeam )
#                                                    # 0: current permanent strain upon unloading, 1: current yield stress, 2: current reference strain for smoothed uniaxial stress strain curve, 3: final stress of last time step used for smoothed version
#        self.NData = 8                              # number of data items

        self.Emod = PropMat[0]                      # Young's modulus
        self.nu = PropMat[1]                        # Poissons's ratio
        self.sigY = PropMat[2]                      # uniaxial yield stress
        self.sigU = PropMat[3]                      # strength
        self.epsU = PropMat[4]                      # limit strain
        self.alphaT = PropMat[5]                    # thermal expansion coefficient
        self.sfac = PropMat[6]                      # parameter to smooth the transition in the bilinear course
        if self.sfac>0.:
            denom = self.sfac*(-self.epsU*self.Emod+self.sigY)
            self.b0 = 0.25*(self.sfac**2*self.epsU*self.Emod-2*self.sfac*self.epsU*self.Emod+self.epsU*self.Emod-self.sigU+2*self.sfac*self.sigU-self.sfac**2*self.sigU)*self.sigY/denom
            self.b1 = 0.50*self.Emod*(-self.epsU*self.Emod+self.sigU-self.sfac*self.epsU*self.Emod-self.sfac*self.sigU+2*self.sigY*self.sfac)/denom
            self.b2 = 0.25*self.Emod**2*(self.epsU*self.Emod-self.sigU)/self.sigY/denom
        self.Density = PropMat[7]                   # specific mass
        self.fct = val[0]                           # concrete tensile strength for tension stiffening
        self.alpha = val[1]                         # tension stiffening parameter
        self.epsY = self.sigY/self.Emod             # uniaxial yield strain  ????
        self.Etan = (self.sigU-self.sigY)/(self.epsU-self.epsY)# tangential / hardening modulus
        self.H = self.Etan/(1-self.Etan/self.Emod)  # hardening modulus
        
        self.epsD  = self.sfac*self.epsY
        
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        if CalcType == 0: return [], [], []
        sigy = max(self.sigY,Elem.StateVar[ipI][1]) # current state parameter - current uniaxial yield stress
        Elem.StateVarN[ipI][0] = 0                  # current values of state variables have to initialized again
        Elem.StateVarN[ipI][1] = 0
        if Elem.dim==2 or Elem.dim==3 or Elem.dim==21:           # plane stress/strain biaxial
            nu = self.nu                            # Poisson's ratio
            mu = self.Emod/(2.*(1.+nu))             # shear modulus
            C0 = self.Emod*(1-nu)/((1+nu)*(1-2*nu))*array([[1,nu/(1-nu),nu/(1-nu),0,0,0],
                                                           [nu/(1-nu),1,nu/(1-nu),0,0,0],
                                                           [nu/(1-nu),nu/(1-nu),1,0,0,0],
                                                           [0,0,0,(1-2*nu)/(2*(1-nu)),0,0],
                                                           [0,0,0,0,(1-2*nu)/(2*(1-nu)),0],
                                                           [0,0,0,0,0,(1-2*nu)/(2*(1-nu))]]) # triaxial isotropic elasticity 
            if Elem.dim==2:                         # plate plane stress / plane strain
                Sig = array( [Elem.DataP[ipI,3],Elem.DataP[ipI,4],Elem.DataP[ipI,6],0.,0.,Elem.DataP[ipI,5]] ) # stress of previous increment
                dEps = array([Dps[0],Dps[1],0.,0.,0.,Dps[2]]) # total strain increment
            elif Elem.dim==21:                      # cb shell -- dEps[2] --> 0, re-evaluated in the following
                Sig = array([Elem.DataP[ipI,0],Elem.DataP[ipI,1],Elem.DataP[ipI,2],Elem.DataP[ipI,3],Elem.DataP[ipI,4],Elem.DataP[ipI,5]] ) # stress of previous increment
                dEps = array([Dps[0],Dps[1],0.,Dps[3],Dps[4],Dps[5]]) # total strain increment
            elif Elem.dim==3:
                Sig = array([Elem.DataP[ipI,0],Elem.DataP[ipI,1],Elem.DataP[ipI,2],Elem.DataP[ipI,3],Elem.DataP[ipI,4],Elem.DataP[ipI,5]] ) # stress of previous increment
                dEps = array([Dps[0],Dps[1],Dps[2],Dps[3],Dps[4],Dps[5]]) # total strain increment
            Fn = sqrt(3.*(((Sig[0]-Sig[1])**2+(Sig[0]-Sig[2])**2+(Sig[1]-Sig[2])**2)/6.+Sig[3]**2+Sig[4]**2+Sig[5]**2))-sigy # distance to yield surface of previous step
            Eflag = False
            dEpp = zeros((6),dtype=float)           # initial value plastic strain increment
            dSig = zeros((6),dtype=float)           # initial value plastic strain increment
            dLam = 0.                               # initial value plastic multiplier increment
            dsiy = 0.                               # initial value current yield stress increment
            ni = 20                                 # iteration limit
            for i in xrange(ni):
                if Elem.PlSt: dEps[2]=-( C0[2,0]*(dEps[0]-dEpp[0])
                                        +C0[2,1]*(dEps[1]-dEpp[1])
                                        +C0[2,3]*(dEps[3]-dEpp[3])
                                        +C0[2,4]*(dEps[4]-dEpp[4])
                                        +C0[2,5]*(dEps[5]-dEpp[5]))/C0[2,2]+dEpp[2] # lateral strain for plane stress
                dSig = dot(C0,dEps-dEpp)
                SigN = Sig + dSig
                J2 = ((SigN[0]-SigN[1])**2+(SigN[0]-SigN[2])**2+(SigN[1]-SigN[2])**2)/6.+SigN[3]**2+SigN[4]**2+SigN[5]**2 # 2nd stress deviator invariant
                if sqrt(3.*J2)-sigy<1.e-9:          # elastic loading, unloading or reloading
                    Eflag = True
                    break
                sm = (SigN[0]+SigN[1]+SigN[2])/3.   # 1st stress invariant / mean stress predictor stress
                rr = sqrt(3./(4.*J2))*array([SigN[0]-sm,SigN[1]-sm,SigN[2]-sm,SigN[3],SigN[4],SigN[5]]) # yield gradient predictor stress
                dL = (Fn + dot(rr,dSig)+rr[3]*dSig[3]+rr[4]*dSig[4]+rr[5]*dSig[5] - self.H*dLam)/(3.*mu+self.H) # with Voigt notation correction
                if dL<1.e-9: break
                dLam = dLam + dL                    # update plastic multiplier
                dEpp = dLam*array([rr[0],rr[1],rr[2],2.*rr[3],2.*rr[4],2.*rr[5]]) # plastic strain incremen with Voigt notation correction
            if i>=ni-1:  
                print elI, ipI, i, ni, Sig, dEps, sigy, dLam, dEpp, dSig  
                raise NameError ("ConFemMaterials::Mises.Sig: no convergence")
            Sig = SigN
            if Eflag:                               # elastic loading or unloading / reloading
                if Elem.dim==2:                     # plane stress / strain
                    if Elem.PlSt: MatM = self.Emod/(1-nu**2)*array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]) # plane stress
                    else:         MatM = self.Emod*(1-nu)/((1+nu)*(1-2*nu))*array([[1,nu/(1-nu),0],[nu/(1-nu),1,0],[0,0,(1-2*nu)/(2*(1-nu))]]) # plane strain
                elif Elem.dim==21:                  # continuum bases shell -- plane stress ???
                    MatM = self.Emod*array([[1./(1-nu**2), nu/(1-nu**2), 0., 0., 0., 0.],
                                            [nu/(1-nu**2), 1./(1-nu**2), 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 1./(2+2*nu), 0., 0.],
                                            [0., 0., 0., 0., 1./(2+2*nu), 0],
                                            [0., 0., 0., 0., 0., 1./(2+2*nu)]]) 
                else: MatM = C0
            else:                                   # plastic loading
                Elem.StateVarN[ipI][1]= sqrt(3.*J2) # new equivalent yield limit 
                A = 1./(self.H + 3.*mu)             # --> 2.*mu*dot(rr,rr) --> dot(xx,rr) --> dot(C0,rr)
                CC = C0 -  A*4.*mu**2*outer(rr,rr)  # --> A*outer(xx,xx)            # tangential material stiffness
                cD = 1./CC[2,2]
                if Elem.dim==2:
                    if Elem.PlSt: MatM=array([[CC[0,0]-CC[0,2]*CC[2,0]*cD,CC[0,1]-CC[0,2]*CC[2,1]*cD,CC[0,5]-CC[0,2]*CC[2,5]*cD],
                                              [CC[1,0]-CC[1,2]*CC[2,0]*cD,CC[1,1]-CC[1,2]*CC[2,1]*cD,CC[1,5]-CC[1,2]*CC[2,5]*cD], 
                                              [CC[5,0]-CC[5,2]*CC[2,0]*cD,CC[5,1]-CC[5,2]*CC[2,1]*cD,CC[5,5]-CC[5,2]*CC[2,5]*cD]])
                    else:         MatM=array([[CC[0,0],CC[0,1],CC[0,5]],[CC[1,0],CC[1,1],CC[1,5]],[CC[5,0],CC[5,1],CC[5,5]]])
                elif Elem.dim==21:                               # shell -- plane stress
                    MatM=array([[CC[0,0]-CC[0,2]*CC[2,0]*cD,CC[0,1]-CC[0,2]*CC[2,1]*cD,0.,CC[0,3]-CC[0,2]*CC[2,3]*cD,CC[0,4]-CC[0,2]*CC[2,4]*cD,CC[0,5]-CC[0,2]*CC[2,5]*cD],
                                [CC[1,0]-CC[1,2]*CC[2,0]*cD,CC[1,1]-CC[1,2]*CC[2,1]*cD,0.,CC[1,3]-CC[1,2]*CC[2,3]*cD,CC[1,4]-CC[1,2]*CC[2,4]*cD,CC[1,5]-CC[1,2]*CC[2,5]*cD],
                                [0.,                        0.,                        0.,0.,                        0.,                        0.],
                                [CC[3,0]-CC[3,2]*CC[2,0]*cD,CC[3,1]-CC[3,2]*CC[2,1]*cD,0.,CC[3,3]-CC[3,2]*CC[2,3]*cD,CC[3,4]-CC[3,2]*CC[2,4]*cD,CC[3,5]-CC[3,2]*CC[2,5]*cD],
                                [CC[4,0]-CC[4,2]*CC[2,0]*cD,CC[4,1]-CC[4,2]*CC[2,1]*cD,0.,CC[4,3]-CC[4,2]*CC[2,3]*cD,CC[4,4]-CC[4,2]*CC[2,4]*cD,CC[4,5]-CC[4,2]*CC[2,5]*cD], 
                                [CC[5,0]-CC[5,2]*CC[2,0]*cD,CC[5,1]-CC[5,2]*CC[2,1]*cD,0.,CC[5,3]-CC[5,2]*CC[2,3]*cD,CC[5,4]-CC[5,2]*CC[2,4]*cD,CC[5,5]-CC[5,2]*CC[2,5]*cD]])
                elif Elem.dim==3:
                    MatM=array([[CC[0,0],CC[0,1],CC[0,2],CC[0,3],CC[0,4],CC[0,5]],
                                [CC[1,0],CC[1,1],CC[1,2],CC[1,3],CC[1,4],CC[1,5]],
                                [CC[2,0],CC[2,1],CC[2,2],CC[2,3],CC[2,4],CC[2,5]],
                                [CC[3,0],CC[3,1],CC[3,2],CC[3,3],CC[3,4],CC[3,5]],
                                [CC[4,0],CC[4,1],CC[4,2],CC[4,3],CC[4,4],CC[4,5]],
                                [CC[5,0],CC[5,1],CC[5,2],CC[5,3],CC[5,4],CC[5,5]]])
            if Elem.dim==2:
                sig = array([Sig[0],Sig[1],Sig[5]])
                return sig, MatM, [Eps[0], Eps[1], Eps[2], Sig[0], Sig[1], Sig[5], Sig[2]] # data
            else:                                   # 3D, shell
                sig = array([Sig[0],Sig[1],Sig[2],Sig[3],Sig[4],Sig[5]])
                return sig, MatM, [Sig[0],Sig[1],Sig[2],Sig[3],Sig[4],Sig[5], 0.] # data
        elif Elem.dim==1 or Elem.dim==10 or Elem.dim==11: raise NameError("ConFemMat::Mises.Sig: dim mismatch") # uniaxial, Bernoulli beam, Timoshenko beam, MisesReMem
        else: raise NameError ("ConFemMaterials::Mises.Sig: not implemented for this element type")
    def UpdateStateVar(self, Elem, ff):
        for j in xrange(Elem.StateVar.shape[0]):    # loop over integration points same as for UpdateStateVar of RCBeam RList
            if Elem.StateVarN[j,1]>Elem.StateVar[j,1]:                 #Elem.StateVar[j] = Elem.StateVarN[j]
                Elem.StateVar[j,0] = Elem.StateVarN[j,0]
                Elem.StateVar[j,1] = Elem.StateVarN[j,1]
            Elem.StateVar[j,2] = Elem.StateVarN[j,2] # longitudinal strain
            if abs(Elem.StateVarN[j,4])>abs(Elem.StateVar[j,4]):
                Elem.StateVar[j,3] = Elem.StateVarN[j,3] # for smoothed uniaxial version of uniaxial mises for reinforcement
                Elem.StateVar[j,4] = Elem.StateVarN[j,4] # "
        return False

class MisesUniaxial( Mises ):                              # elastoplastic Mises  for Elem.dim==1 or Elem.dim==10 or Elem.dim==11: # uniaxial, Bernoulli beam, Timoshenko beam, MisesReMem
    def __init__(self, PropMat, val):
        Mises.__init__(self, PropMat, val)
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        def sigEpsL(eps):                           # uniaxial stress strain curve smoothing between elastic and yielding branch within a range a ... b 
            Emod = self.Emod
            ET = self.Etan
            epsD = self.epsD
            epsY_= self.epsY
            sigY_= self.sigY
            gg  = 2.                                 # gg*epsD marks right side transition range
            g2  = gg**2
            g3  = gg**3
            ff = 1./(g3+3*gg+1+3*g2)
            if Elem.StateVar[ipI,3]==0: epsY, sigY = self.epsY, self.sigY  # initialization
            else:                       epsY, sigY = Elem.StateVar[ipI,3], Elem.StateVar[ipI,4] # actual values
            b0 =  ff*(sigY-2*g2*Emod*epsD+2*g2*epsD*ET+3*gg*sigY+Emod*epsY_*g3+3*g2*Emod*epsY_)
            b1 = -ff*(-4*gg*epsD*Emod-epsD*ET+ET*gg*epsD+g2*Emod*epsD-4*g2*epsD*ET+6*gg*Emod*epsY_-6*gg*sigY-g3*epsD*Emod)/epsD
            b2 = -ff*(2*g2*Emod*epsD-2*g2*epsD*ET+3*gg*Emod*epsY_-3*gg*sigY-2*gg*epsD*Emod+2*ET*gg*epsD-3*Emod*epsY_+3*sigY-2*epsD*ET+2*Emod*epsD)/epsD**2
            b3 =  ff*(2*Emod*epsY_-2*sigY+gg*epsD*Emod+epsD*ET-Emod*epsD-ET*gg*epsD)/epsD**3
            eps1 = eps+2*epsY_-epsY
            eps2 = eps        -epsY
            if   eps < epsY-2*epsY_-gg*epsD+1.e-9:
                sig = -sigY + ET*eps1
                Emo = ET
                Elem.StateVarN[ipI,3] =  eps + 2*epsY_ + gg*epsD
                Elem.StateVarN[ipI,4] = -sig - ET*gg*epsD
#                if ipI==0: print '\nCCC', epsY,epsY-epsD, eps, eps1, 'X', sig, Emo, sigY
            elif eps < epsY-2*epsY_+epsD:
                sig =   b3*eps1**3 -  b2*eps1**2 + b1*eps1-b0
                Emo = 3*b3*eps1**2 -2*b2*eps1    + b1
#                if ipI==0: print '\nBBB', epsY,epsY-epsD, eps, eps1, 'X', sig, Emo, sigY
            elif eps <   epsY-epsD:                     # will presumably not work for cyclic loading with a change from compression to tension and vice versa
                sig = sigY_ + Emod*eps2 
                Emo = Emod
#                if ipI==0: print '\nAAA', epsY,epsY-epsD, eps, eps2, 'X', sig, Emo, sigY, sigY_  
            elif eps < epsY+gg*epsD-1.e-9:
                sig =   b3*eps2**3 +  b2*eps2**2 + b1*eps2 + b0
                Emo = 3*b3*eps2**2 +2*b2*eps2    + b1
#                if ipI==0: print '\nBB1', epsY,epsY-epsD, eps, eps2, 'X', sig, Emo, sigY
            else:
                sig = sigY + ET*eps2
                Emo = ET
                Elem.StateVarN[ipI,3] = eps - gg*epsD
                Elem.StateVarN[ipI,4] = sig - ET*gg*epsD
#                if ipI==0: print '\nCC1', epsY,epsY-epsD, eps, eps2, 'X', sig, Emo, sigY
            return sig, Emo

        if CalcType == 0: return [], [], []
        sigy = max(self.sigY,Elem.StateVar[ipI][1]) # current state parameter - current uniaxial yield stress
        Elem.StateVarN[ipI][0] = 0                  # current values of state variables have to initialized again
        Elem.StateVarN[ipI][1] = 0

        epsP = Elem.StateVar[ipI][0]            # current permanent strain with zero stress
        eps = Eps[0] - self.alphaT*Temp         # stress inducing strain
        dps = Dps[0] - self.alphaT*dTmp         # stress inducing strain increment
        epc = 0                                 # yielding correction strain in case of tension stiffening 
        if Elem.TensStiff:                      # tension stiffening
            if Elem.Type=='SH4':                            # Elem.dim was presumably changed temporarily
                if   Elem.nInt==2 and ipI>=16: jj = (ipI-16)//4 # floor division -> integer value -> index for reinforcement layer, 4 stands for number of integration points
                elif Elem.nInt==5 and ipI>=20: jj = (ipI-20)//5 # "
                rhoeff = Elem.Geom[jj+2,2]      # effective reinforcement ratio
                betat = Elem.Geom[jj+2,3]       # tension stiffening parameter betat
            else:
                rhoeff = Elem.Geom[elI+2,2]     # effective reinforcement ratio, elI presumably correct, see ConFemMaterials::RCBeam:sig - loop for reinforcement layers
                betat = Elem.Geom[elI+2,3]      # tension stiffening parameter betat
            if betat==0.: 
                eps_, Emod_, dsig = epsP, self.Emod, 0.
            else:
                if self.fct/rhoeff>sigy: raise NameError("ConFemMat::Mises.Sig: effective reinforcement ratio to less for minimum reinforcement")
                sigsr = self.fct/rhoeff
                eps_ = self.fct*(self.alpha-betat)/(self.Emod*rhoeff)
                Emod_ = self.alpha*sigsr/eps_
                dsig = betat*self.fct/rhoeff
                epc = dsig/self.Emod
        else:   eps_, Emod_, dsig = epsP, self.Emod, 0. # no tension stiffening

        if self.sfac>0.:                        # elasto-plastic with smoothing of transition but WITHOUT elasto-plastic unloading / reloading
            sig_, Emod_ = sigEpsL(eps)
            MatM = array([[Emod_,0.],[0.,0.]])
            sig = array([sig_,0.])
        elif eps<=(epsP-self.epsY):             # plastic compressive
            MatM = array([[self.Etan,0.],[0.,0.]])# tangential material stiffness
            sig  = array([-sigy + self.Etan*dps,0.])# stress
            Elem.StateVarN[ipI][0]= eps+self.epsY # update state variables
            Elem.StateVarN[ipI][1]=-sig[0]
        elif eps<(epsP):                        # elastic compressive
            MatM = array([[self.Emod,0.],[0.,0.]])
            sig  = array([self.Emod*(eps-epsP),0.])
        elif eps<(epsP+self.epsY-epc-ZeroD):    # elastic tensile with some tolerance
            if eps<eps_:
                MatM = array([[Emod_,0.],[0.,0.]])
                sig  = array([Emod_*(eps-epsP),0.])
            else:
                MatM = array([[self.Emod,0.],[0.,0.]])
                sig  = array([self.Emod*(eps-epsP) + dsig,0.])
        else:                                   # plastic tensile
            MatM = array([[self.Etan,0.],[0.,0.]])
            sig  = array([ sigy + self.Etan*dps,0.])
            Elem.StateVarN[ipI][0]= eps-self.epsY
            Elem.StateVarN[ipI][1]= sig[0]
        return sig, MatM, [ eps, sig[0], epsP] 

class Lubliner(Material):                              # elastoplastic Mises
    def __init__(self, PropMat):
#    def __init__(self,         SymmetricVal, RTypeVal, UpdateVal, Updat2Val, StateVarVal, NDataVal):
        Material.__init__(self, False,        None,     True,      False,     12,           8)
#        self.Symmetric = True                       # flag for symmetry of material matrices
#        self.Update = True                          # has specific update procedure for update of state variables
#        self.Updat2 = False                         # no 2 stage update procedure
#        self.StateVar = 1                           # number of state variables per integration point
#        self.NData = 8                              # number of data items

        self.E_0  = PropMat[0]                      # initial Young's modulus
        self.nu   = PropMat[1]                      # Poissons's ratio
        self.f_c0 = PropMat[2]                      # elastic uniaxial compressive stress (unsigned)
        self.f_cm = PropMat[3]                      # uniaxial compressive strength (unsigned)
        self.f_t0 = PropMat[4]                      # uniaxial tensile strength
        alpha_    = PropMat[5]                      # ratio of biaxial compressive strength to uniaxial compressive strength
        K_c       = PropMat[6]                      # ratio of strength on tensile meridian compared to compressive meridian
        self.G_F  = PropMat[7]                      # tensile cracking energy
        self.G_ch = PropMat[8]                      # compressive cracking energy
        self.alpha_p = PropMat[9]*pi/180.           # angle of dilatancy (input in degrees, transformed to rad)
        self.ecc  = PropMat[10]                     # eccentricity of plastic potential surface
        self.Density = PropMat[11]                  # specific mass
        self.CalDamC = 0.4                          # assumed damage for maximum uniaxial compression stress
        self.CalDamT = 0.5                          # assumed damage for half maximum uniaxial tensile stress
        self.ff = self.E_0*(1.-self.nu)/((1.+self.nu)*(1.-2*self.nu))
        # uniaxial
        xx = self.f_cm/self.f_c0
        self.a_c  = 2.*xx - 1.+2.*sqrt(xx**2-xx)
        self.dbbC = log(1.-self.CalDamC) / log((1.+self.a_c)/(2.*self.a_c)) 
        self.a_t  = 1.0
        self.dbbT = log(1.-self.CalDamT) / log((1.+self.a_t-sqrt(1.+self.a_t**2))/(2.*self.a_t))
        # multiaxial
        self.alpha = (alpha_-1.)/(2.*alpha_-1.)
        self.gamma = 3.*(1.-K_c)/(2.*K_c-1.)
#        print 'CCC', self.alpha, self.gamma, '__', self.f_c0/self.f_t0
        # elastic contributions
        nu = self.nu                                # Poisson's ratio
        self.C0 = self.E_0*(1-nu)/((1+nu)*(1-2*nu))*array([[1,nu/(1-nu),nu/(1-nu),0,0,0],
                                                       [nu/(1-nu),1,nu/(1-nu),0,0,0],
                                                       [nu/(1-nu),nu/(1-nu),1,0,0,0],
                                                       [0,0,0,(1-2*nu)/(2*(1-nu)),0,0],
                                                       [0,0,0,0,(1-2*nu)/(2*(1-nu)),0],
                                                       [0,0,0,0,0,(1-2*nu)/(2*(1-nu))]]) # triaxial isotropic elasticity
#        self.C0_ = self.E_0*(1-nu)/((1+nu)*(1-2*nu))*array([[1.,nu/(1.-nu),nu/(1.-nu)],
#                                                             [nu/(1.-nu),1.,nu/(1.-nu)],
#                                                             [nu/(1.-nu),nu/(1.-nu),1.]]) # triaxial isotropic elasticity 
        self.ZeroTol = 1.e-3
        self.KapTol  = 1.e-4
#        self.KapTol  = 1.e-5
    def fC(self, kappa):
        phi = 1. + self.a_c*(2.+self.a_c)*kappa
        fC = self.f_c0 *( (1.+self.a_c)*sqrt(phi) - phi )/self.a_c
        dfC= self.f_c0*( 0.5*(1.+self.a_c)*(2.*self.a_c+self.a_c**2)/sqrt(phi)-self.a_c*(2.+self.a_c) )/self.a_c
        return fC, dfC
    def fT(self, kappa):
        phi = 1. + self.a_t*(2.+self.a_t)*kappa
        fT = self.f_t0 *( (1.+self.a_t)*sqrt(phi) - phi )/self.a_t
        dfT= self.f_t0*( 0.5*(1.+self.a_t)*(2.*self.a_t+self.a_t**2)/sqrt(phi)-self.a_t*(2.+self.a_t) )/self.a_t
        return fT, dfT
    def FF(self, I1, J2, sig_max, fC, fT):                       # Yield function
        if   sig_max<-self.ZeroTol: 
            return (sqrt(3.*J2) + self.alpha*I1 + self.gamma*sig_max)/(1.-self.alpha)
        elif sig_max< self.ZeroTol: 
            return (sqrt(3.*J2) + self.alpha*I1 )                    /(1.-self.alpha)
        else:
            beta_ =  fC/fT*(1.-self.alpha) - (1.+self.alpha)
            return (sqrt(3.*J2) + self.alpha*I1 + beta_*sig_max)/(1.-self.alpha)
    def dFF(self, J2, sig_dev, nd_, sig_max, fC, dfC, fT, dfT, kapC, kapT, DC, DT):  # gradient of yield function
        x = 0.5*sqrt(3./J2)
        y = 1./(1.-self.alpha)
        a_c = self.a_c
        a_t = self.a_t
        dbbC = self.dbbC
        dbbT = self.dbbT
        f_c0 = self.f_c0
        phiC  = 1.+a_c*(2.+a_c)*kapC
        RphiC = sqrt(phiC)
        xx = 1. + a_c - RphiC
#        dbbC, dbbT, DC, DT = 0., 0., 0., 0.            # for control purposes
        yy = 0.5*f_c0*(2.+a_c)/(1.-DT)*pow(xx/a_c,-dbbC)/(xx*RphiC) * ( (2.-dbbC)*phiC + (dbbC+dbbC*a_c-3.-3.*a_c)*RphiC + (1.+a_c)**2 ) 
        phiT  = 1.+a_t*(2.+a_t)*kapT
        RphiT = sqrt(phiT)
        xx = 1. + a_t - RphiT 
        zz = 0.5*fC/(1.-DC) * dbbT    *pow(xx/a_t,-dbbT)/(xx*RphiT) * a_t*(2.+a_t)
        dfC_eff = array([yy,zz])                           # gradient of effective compression "cohesion" with respect to internal state variables kappa_c, kappa_t
        if   sig_max<-self.ZeroTol:
            z =  self.alpha +self.gamma
            return array([ y*(x*sig_dev[0]+self.alpha +self.gamma*nd_[0]),
                           y*(x*sig_dev[1]+self.alpha +self.gamma*nd_[1]),
                           y*(x*sig_dev[2]+self.alpha +self.gamma*nd_[2]),
                           y*(x*sig_dev[3]            +self.gamma*nd_[3]),
                           y*(x*sig_dev[4]            +self.gamma*nd_[4]),
                           y*(x*sig_dev[5]            +self.gamma*nd_[5])]), array([ -dfC_eff[0], -dfC_eff[1]])  
        elif sig_max< self.ZeroTol: 
            return array([ y*(x*sig_dev[0]+self.alpha),
                           y*(x*sig_dev[1]+self.alpha),
                           y*(x*sig_dev[2]+self.alpha),
                           y*(x*sig_dev[3]),
                           y*(x*sig_dev[4]),
                           y*(x*sig_dev[5])]), array([ -dfC_eff[0], -dfC_eff[1]])  
        else:                       
            beta_ = fC/fT*(1.-self.alpha) - (1.+self.alpha)
            return array([ y*(x*sig_dev[0]+self.alpha +beta_*nd_[0]),
                           y*(x*sig_dev[1]+self.alpha +beta_*nd_[1]),
                           y*(x*sig_dev[2]+self.alpha +beta_*nd_[2]),
                           y*(x*sig_dev[3]            +beta_*nd_[3]),
                           y*(x*sig_dev[4]            +beta_*nd_[4]),
                           y*(x*sig_dev[5]            +beta_*nd_[5])]), array([ dfC*sig_max/fT - dfC_eff[0], -sig_max*dfT*fC/(fT**2) - dfC_eff[1]]) # ???
    def dGG(self, J2, sig_dev):                                 # gradient of flow potential
        x = 1.5/sqrt( (self.ecc*self.f_t0*tan(self.alpha_p))**2 + 3.*J2)
        y = tan(self.alpha_p)/3.
        return array([x*sig_dev[0] + y,
                      x*sig_dev[1] + y,
                      x*sig_dev[2] + y,
                      x*sig_dev[3],
                      x*sig_dev[4],
                      x*sig_dev[5]])
    def GP(self, sig, fC, dfC, fT, dfT, kapC, kapT, DC, DT, Flag):                                    # whole plastic stuff for given stress state
        I1  = sig[0] + sig[1] + sig[2]
        ee, ev  = eigh(array([ [sig[0], sig[5], sig[4]] , [sig[5], sig[1], sig[3]] , [sig[4], sig[3], sig[2]] ])) # to find largest principal value
        Psig_max = ee[2]                                        # ascending order, largest (signed!) on last position 
        J2  = ((ee[0]-ee[1])**2+(ee[0]-ee[2])**2+(ee[1]-ee[2])**2)/6.
        FF  = self.FF( I1, J2, Psig_max, fC, fT)       # Yield function
        if not Flag: return FF
        pp  = I1/3.
#        if abs(ee[2]-ee[1])<self.ZeroTol:                       # same two largest principal stresses
#            ev[0,2] = ev[0,1]+ev[0,2]                           # second index indicates eigenvector
#            ev[1,2] = ev[1,1]+ev[1,2]
#            ev[2,2] = ev[2,1]+ev[2,2]
#            eL = sqrt( ev[0,2]**2 + ev[1,2]**2 + ev[2,2]**2 )
#            ev[0,2] = ev[0,2]/eL                           # second index indicates eigenvector
#            ev[1,2] = ev[1,2]/eL
#            ev[2,2] = ev[2,2]/eL
        sig_dev  = array([sig[0]-pp, sig[1]-pp, sig[2]-pp, sig[3], sig[4], sig[5]])
        nd_      = array([ev[0,2]*ev[0,2],ev[1,2]*ev[1,2],ev[2,2]*ev[2,2],ev[1,2]*ev[2,2],ev[0,2]*ev[2,2],ev[0,2]*ev[1,2]]) # gradient of largest principal stress corresponding to Voigt notation, largest eigenvalue in last position (ascending order)
        dff, dkk = self.dFF( J2, sig_dev, nd_, Psig_max, fC, dfC, fT, dfT, kapC, kapT, DC, DT)      # gradient of yield function with respect to stress
        dgg      = self.dGG( J2, sig_dev)                           # gradient of flow potential
        return FF, dff, dkk, dgg
    def CalcKapC(self, g_ch, g_F, Sig, dEpsP, kapC, kapC_old, kapT, kapT_old, ipI):
        r = (max(Sig[0],0.)+max(Sig[1],0.)+max(Sig[2],0.)) / (abs(Sig[0])+abs(Sig[1])+abs(Sig[2])) #
        dkapC  = kapC - kapC_old
        dkapT = kapT - kapT_old
        ni  = 10
        for i in xrange(ni):
            # compression
            kapC = kapC_old + dkapC
            kapC = max(0.,kapC)
            fC, dfC = self.fC(kapC)
            h0  = - fC*(1.-r)/g_ch
            xx  =   h0*dEpsP[0] 
            d0  = - dfC*(1.-r)/g_ch
            yy  =   d0*dEpsP[0] 
            dkapC_= (xx - dkapC*yy )/(1.-yy)
            # tension
            kapT = kapT_old + dkapT
            kapT = max(0.,kapT)
            fT, dfT = self.fT(kapT)
            h2  =   fT * r/g_F
            xx  =   h2*dEpsP[2]
            d2  =   dfT* r/g_F
            yy  =   d2*dEpsP[2]
            dkapT_= (xx - dkapT*yy )/(1.-yy)
            # convergence control
            if sqrt( (dkapC-dkapC_)**2 + (dkapT-dkapT_)**2 ) < self.KapTol: break
            if ipI==0 and i>=ni-1: print 'ZZZ',i,  dkapC, dkapC_, '__', dkapC, dkapT 
            dkapC = dkapC_
            dkapT = dkapT_
        if i>=ni-1: raise NameError ("ConFemMaterials::Lubliner.CalcKapC: no convergence B")
        kapC = kapC_old + dkapC
        kapT = kapT_old + dkapT
        return kapC, kapT, h0, h2, fC, dfC, fT, dfT
    def Dam(self, kapC, kapT):
        a_c = self.a_c
        phiC= 1. + a_c*(2.+a_c)*kapC
        RpC= sqrt(phiC)
        z  = 1. + a_c - RpC
        x  = z / a_c
        y  = pow(x,self.dbbC)
        DC = 1. - y 
        GC = self.dbbC * y/z * (2.*a_c+a_c**2)/(2.*RpC) 
        #
        a_t = self.a_t
        phiT= 1. + a_t*(2.+a_t)*kapT
        RpT= sqrt(phiT)
        z  = 1. + a_t - RpT
        x  = z / a_t
        y = pow(x,self.dbbT)
        DT = 1. - y 
        GT = self.dbbT * y/z * (2.*a_t+a_t**2)/(2.*RpT)
        return DC, DT, GC, GT               
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        if CalcType == 0: return [], [], []
        kapT_old = Elem.StateVar[ipI][0]
        kapT     = kapT_old
        kapC_old = Elem.StateVar[ipI][1]
        kapC     = kapC_old
        fc_old   = Elem.StateVar[ipI][2] + self.f_c0
        ft_old   = Elem.StateVar[ipI][3] + self.f_t0
        DC       = Elem.StateVar[ipI][4]                        # scalar damsge
        DT       = Elem.StateVar[ipI][5]
        DD       = 1.-(1.-DC)*(1.-DT)                           # update damage
        Lch      = Elem.Lch_
        g_ch     = self.G_ch/Lch
        g_F      = self.G_F/Lch
#        DC_old = Elem.StateVar[ipI][12]
#        DT_old = Elem.StateVar[ipI][13]
        if Elem.dim==2 or Elem.dim==3:
            if Elem.dim==2:                                     # plate plane stress / plane strain
#                Sig  = array( [Elem.DataP[ipI,3],Elem.DataP[ipI,4],Elem.DataP[ipI,6],0.,0.,Elem.DataP[ipI,5]] ) # stress of previous increment
                dEps = array([Dps[0],Dps[1],0.,0.,0.,Dps[2]])   # total strain increment
            elif Elem.dim==3:
#                Sig  = array([Elem.DataP[ipI,0],Elem.DataP[ipI,1],Elem.DataP[ipI,2],Elem.DataP[ipI,3],Elem.DataP[ipI,4],Elem.DataP[ipI,5]] ) # stress of previous increment
                dEps = array([Dps[0],Dps[1],Dps[2],Dps[3],Dps[4],Dps[5]])   # total strain increment
            EpsP_old = array([Elem.StateVar[ipI][4+2],Elem.StateVar[ipI][5+2],Elem.StateVar[ipI][6+2],Elem.StateVar[ipI][7+2],Elem.StateVar[ipI][8+2],Elem.StateVar[ipI][9+2]]) # old plastic strain in global system
            SigEff   = dot(self.C0,(Eps-EpsP_old))                   # trial stress
            fc_eff = fc_old/(1.-DD)
            FF  = self.GP( SigEff, fc_old, None, ft_old, None, None, None, DC, DT, False)          # yield condition
            i_, Flag = 0, False
            if FF>=fc_eff: 
                Flag = True
                PdEps, gg_ = eigh(array([ [dEps[0], dEps[5], dEps[4]] , [dEps[5], dEps[1], dEps[3]] , [dEps[4], dEps[3], dEps[2]] ]))  # principal values and eigenvectors of strain increment tensor, smallest (signed) 1st!
                gg_[0,2] = gg_[1,0]*gg_[2,1]-gg_[2,0]*gg_[1,1]                  # to have a right handed system
                gg_[1,2] = gg_[2,0]*gg_[0,1]-gg_[0,0]*gg_[2,1]
                gg_[2,2] = gg_[0,0]*gg_[1,1]-gg_[1,0]*gg_[0,1]
                gg = array([[gg_[0,0],gg_[1,0],gg_[2,0]],[gg_[0,1],gg_[1,1],gg_[2,1]],[gg_[0,2],gg_[1,2],gg_[2,2]]])
                TTS = array([[ gg[0,0]**2, gg[0,1]**2, gg[0,2]**2, 2.*gg[0,1]*gg[0,2], 2.*gg[0,0]*gg[0,2], 2.*gg[0,0]*gg[0,1]],
                             [ gg[1,0]**2, gg[1,1]**2, gg[1,2]**2, 2.*gg[1,1]*gg[1,2], 2.*gg[1,0]*gg[1,2], 2.*gg[1,0]*gg[1,1]],
                             [ gg[2,0]**2, gg[2,1]**2, gg[2,2]**2, 2.*gg[2,1]*gg[2,2], 2.*gg[2,0]*gg[2,2], 2.*gg[2,0]*gg[2,1]],
                             [ gg[1,0]*gg[2,0], gg[1,1]*gg[2,1], gg[1,2]*gg[2,2], gg[1,2]*gg[2,1]+gg[1,1]*gg[2,2], gg[1,2]*gg[2,0]+gg[1,0]*gg[2,2], gg[1,1]*gg[2,0]+gg[1,0]*gg[2,1]],
                             [ gg[0,0]*gg[2,0], gg[0,1]*gg[2,1], gg[0,2]*gg[2,2], gg[0,2]*gg[2,1]+gg[0,1]*gg[2,2], gg[0,2]*gg[2,0]+gg[0,0]*gg[2,2], gg[0,1]*gg[2,0]+gg[0,0]*gg[2,1]],
                             [ gg[0,0]*gg[1,0], gg[0,1]*gg[1,1], gg[0,2]*gg[1,2], gg[0,2]*gg[1,1]+gg[0,1]*gg[1,2], gg[0,2]*gg[1,0]+gg[0,0]*gg[1,2], gg[0,1]*gg[1,0]+gg[0,0]*gg[1,1]]])
                TTD = array([[ gg[0,0]**2, gg[0,1]**2, gg[0,2]**2,    gg[0,1]*gg[0,2],    gg[0,0]*gg[0,2],    gg[0,0]*gg[0,1]],
                             [ gg[1,0]**2, gg[1,1]**2, gg[1,2]**2,    gg[1,1]*gg[1,2],    gg[1,0]*gg[1,2],    gg[1,0]*gg[1,1]],
                             [ gg[2,0]**2, gg[2,1]**2, gg[2,2]**2,    gg[2,1]*gg[2,2],    gg[2,0]*gg[2,2],    gg[2,0]*gg[2,1]],
                             [ 2.*gg[1,0]*gg[2,0], 2.*gg[1,1]*gg[2,1], 2.*gg[1,2]*gg[2,2], gg[1,2]*gg[2,1]+gg[1,1]*gg[2,2], gg[1,2]*gg[2,0]+gg[1,0]*gg[2,2], gg[1,1]*gg[2,0]+gg[1,0]*gg[2,1]],
                             [ 2.*gg[0,0]*gg[2,0], 2.*gg[0,1]*gg[2,1], 2.*gg[0,2]*gg[2,2], gg[0,2]*gg[2,1]+gg[0,1]*gg[2,2], gg[0,2]*gg[2,0]+gg[0,0]*gg[2,2], gg[0,1]*gg[2,0]+gg[0,0]*gg[2,1]],
                             [ 2.*gg[0,0]*gg[1,0], 2.*gg[0,1]*gg[1,1], 2.*gg[0,2]*gg[1,2], gg[0,2]*gg[1,1]+gg[0,1]*gg[1,2], gg[0,2]*gg[1,0]+gg[0,0]*gg[1,2], gg[0,1]*gg[1,0]+gg[0,0]*gg[1,1]]])
                dEpsPL = PdEps                                              # initial guess for plastic strain increment, is already in actual local system
                ni_ = 10 # 20 
                for i_ in xrange(ni_):                                      # iteration loop for implicit compuations of kap, plastic strain and stress
                    SigL   = dot(TTS,SigEff)                                # transform stresses to actual local system
                    kapC_, kapT_, h0, h2, fC, dfC, fT, dfT = self.CalcKapC(g_ch, g_F, SigL, dEpsPL, kapC, kapC_old, kapT, kapT_old, ipI)
                    DC, DT, GC, GT = self.Dam(kapC_, kapT_)
                    # compute plastic strain increment
                    FF, dff, dkk, dgg = self.GP( SigL, fC, dfC, fT, dfT, kapC_, kapT_, DC, DT, True)             # gradients should be evaluated in local system
                    GF = array([[dgg[0]*dff[0],dgg[0]*dff[1],dgg[0]*dff[2],dgg[0]*dff[3],dgg[0]*dff[4],dgg[0]*dff[5]],
                                [dgg[1]*dff[0],dgg[1]*dff[1],dgg[1]*dff[2],dgg[1]*dff[3],dgg[1]*dff[4],dgg[1]*dff[5]],
                                [dgg[2]*dff[0],dgg[2]*dff[1],dgg[2]*dff[2],dgg[2]*dff[3],dgg[2]*dff[4],dgg[2]*dff[5]],
                                [dgg[3]*dff[0],dgg[3]*dff[1],dgg[3]*dff[2],dgg[3]*dff[3],dgg[3]*dff[4],dgg[3]*dff[5]],
                                [dgg[4]*dff[0],dgg[4]*dff[1],dgg[4]*dff[2],dgg[4]*dff[3],dgg[4]*dff[4],dgg[4]*dff[5]],
                                [dgg[5]*dff[0],dgg[5]*dff[1],dgg[5]*dff[2],dgg[5]*dff[3],dgg[5]*dff[4],dgg[5]*dff[5]]])
                    zz = dot(self.C0,dgg)
                    yy = dff[0]*zz[0]+dff[1]*zz[1]+dff[2]*zz[2]+dff[3]*zz[3]+dff[4]*zz[4]+dff[5]*zz[5]
                    zz_= -dkk[0]*h0*dgg[0] -dkk[1]*h2*dgg[2] + yy
                    EP = 1./zz_ * dot(GF,self.C0)                           # plastic operator
                    PdEps_ = array([PdEps[0],PdEps[1],PdEps[2],0.,0.,0.])   # extend principal strains to common length
                    dEpsPL = dot(EP,PdEps_)                                 # local plastic strain increment from local strain increment and plastic operator
                    dEpsP  = dot(transpose(TTS),dEpsPL)                     # transform local plastic strain increment back to global system
                    # compute effective stress
                    SigEff  = dot(self.C0,(Eps - (EpsP_old+dEpsP) ))
                    # convergence control
                    if sqrt( (kapC-kapC_)**2 + (kapT-kapT_)**2 ) < self.KapTol: break
                    if ipI==0 and i_>ni_-3: print 'ZZZ', ipI, kapC, kapC_, kapC-kapC_, kapT, kapT_ 
                    kapC = kapC_
                    kapT = kapT_
                if i_>=ni_-1: raise NameError ("ConFemMaterials::Lubliner.Sig: no convergence A")
                DD = 1.-(1.-DC)*(1.-DT)                                     # update damage
                DM = array([ (1.-DT)*GC*h0 , 0. , (1.-DC)*GT*h2, 0., 0., 0. ]) #
                Sig0L = dot(self.C0,(PdEps_-dEpsPL))                        # local elastic strains
                X = array([[Sig0L[0]*DM[0],Sig0L[0]*DM[1],Sig0L[0]*DM[2],Sig0L[0]*DM[3],Sig0L[0]*DM[4],Sig0L[0]*DM[5]],
                           [Sig0L[1]*DM[0],Sig0L[1]*DM[1],Sig0L[1]*DM[2],Sig0L[1]*DM[3],Sig0L[1]*DM[4],Sig0L[1]*DM[5]],
                           [Sig0L[2]*DM[0],Sig0L[2]*DM[1],Sig0L[2]*DM[2],Sig0L[2]*DM[3],Sig0L[2]*DM[4],Sig0L[2]*DM[5]],
                           [Sig0L[3]*DM[0],Sig0L[3]*DM[1],Sig0L[3]*DM[2],Sig0L[3]*DM[3],Sig0L[3]*DM[4],Sig0L[3]*DM[5]],
                           [Sig0L[4]*DM[0],Sig0L[4]*DM[1],Sig0L[4]*DM[2],Sig0L[4]*DM[3],Sig0L[4]*DM[4],Sig0L[4]*DM[5]],
                           [Sig0L[5]*DM[0],Sig0L[5]*DM[1],Sig0L[5]*DM[2],Sig0L[5]*DM[3],Sig0L[5]*DM[4],Sig0L[5]*DM[5]]])
                Y = dot(X,EP)
                Elem.StateVarN[ipI][0]  = kapT
                Elem.StateVarN[ipI][1]  = kapC
                Elem.StateVarN[ipI][2]  = fC - self.f_c0
                Elem.StateVarN[ipI][3]  = fT - self.f_t0
                Elem.StateVarN[ipI][4]  = DC
                Elem.StateVarN[ipI][5]  = DT
                Elem.StateVarN[ipI][6]  = Elem.StateVar[ipI][6]  + dEpsP[0]            # update of plastic strains
                Elem.StateVarN[ipI][7]  = Elem.StateVar[ipI][7]  + dEpsP[1]
                Elem.StateVarN[ipI][8]  = Elem.StateVar[ipI][8]  + dEpsP[2]
                Elem.StateVarN[ipI][9]  = Elem.StateVar[ipI][9]  + dEpsP[3]
                Elem.StateVarN[ipI][10] = Elem.StateVar[ipI][10] + dEpsP[4]
                Elem.StateVarN[ipI][11] = Elem.StateVar[ipI][11] + dEpsP[5]
                
#                Elem.StateVarN[ipI][12] = DC_old + (GC)*(kapC-kapC_old)
#                Elem.StateVarN[ipI][13] = DT_old + GT*(kapT-kapT_old)
#                if ipI==0: print 'XXX', DC, Elem.StateVarN[ipI][12], 'CCC', (DC-Elem.StateVarN[ipI][12])/DC, 'TTT', DT, Elem.StateVarN[ipI][13], (DT-Elem.StateVarN[ipI][13])/DT
#                Elem.StateVarN[ipI][14] = GC
#                Elem.StateVarN[ipI][15] = GT
#                fc_old = fC
                
            Sig = (1.-DD)*SigEff
#            if not Flag and ipI==0: print 'XXX', kapC, kapT, fc_old, DD, '__', FF, fc_eff, i_, '__', SigEff[0], Sig[0]
            if Elem.dim==2:                     # plane stress / strain
                if Elem.PlSt: MatM = self.Emod/(1-nu**2)*array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]) # plane stress
                else:         MatM = self.Emod*(1-nu)/((1+nu)*(1-2*nu))*array([[1,nu/(1-nu),0],[nu/(1-nu),1,0],[0,0,(1-2*nu)/(2*(1-nu))]]) # plane strain
                sig = array([Sig[0],Sig[1],Sig[5]])
                return sig, MatM, [Eps[0], Eps[1], Eps[2], Sig[0], Sig[1], Sig[5], Sig[2]] # data
            else: 
                if i_>0: 
#                    CP  = (1.-DD)*(self.C0 - dot(self.C0,EP))                   # tangential material stiffness in local system
                    CP  = (1.-DD)*(self.C0 - dot(self.C0,EP)) - Y                   # tangential material stiffness in local system
                    MatM = dot(transpose(TTD),dot(CP,TTD))                      # transform local stiffness into global system
                else:    
                    MatM = self.C0
                sig = array([Sig[0],Sig[1],Sig[2],Sig[3],Sig[4],Sig[5]])
                return sig, MatM, [Sig[0],Sig[1],Sig[2],Sig[3],Sig[4],Sig[5], 0.] # data
        else: raise NameError ("ConFemMaterials::Lubliner.Sig: not implemented for this element type")
    def UpdateStateVar(self, Elem, ff):
        for j in xrange(Elem.StateVar.shape[0]):    # loop over integration points 
#            if Elem.StateVarN[j,0]>Elem.StateVar[j,0]: Elem.StateVar[j,0] = Elem.StateVarN[j,0]
            ###############################################
#            for i in (1,2): Elem.StateVar[j,i] = Elem.StateVarN[j,i]
            Elem.StateVar[j] = Elem.StateVarN[j]
            ###############################################
        return False

class Spring(Material):                             # Nonlinear spring for bond
    def __init__(self, PropMat):
        self.PropMat = PropMat
        self.StateVar = None
        self.NData = 2                              # number of data items
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        if Eps[0]<0: si=-1
        else:        si= 1
        s = abs(Eps[0])
        s1 = self.PropMat[0]
        F1 = self.PropMat[1]
        s2 = self.PropMat[2]
        F2 = self.PropMat[3]
        s0 = self.PropMat[4]
        d0 = self.PropMat[5]
        if s<= s0:
            ssig=si*d0*s
            dsig=d0
        elif s<= s1:
            t5 = s*s
            t8 = s0*s0
            t9 = d0*t8
            t11 = s0*F1
            t14 = s0*s1*d0
            t17 = 3.0*s1*F1
            t18 = s1*s1
            t19 = t18*d0
            t20 = 2.0*t19
            ssig = (-(d0*s0+s1*d0-2.0*F1)*t5*s
                       +(2.0*t9-3.0*t11+2.0*t14-t17+t20)*t5
                       -s1*(t14+t19+4.0*t9-6.0*t11)*s
                       +t8*(t20+t11-t17)) / (t8*s0+3.0*s0*t18-t18*s1-3.0*s1*t8)
            ssig = si*ssig
            dsig = (-3.0*(d0*s0+s1*d0-2.0*F1)*t5+2.0*(2.0*t9-3.0*t11+2.0*t14-3.0*s1*F1+2.0*t19)*s-s1*(t14+t19+4.0*t9-6.0*t11))/(t8*s0+3.0*s0*t18-t18*s1-3.0*s1*t8)
        elif s<=s2:
            t1  = -F1+F2
            t20 = s1*s1
            t21 = t20*s1
            t14 = s2*s2
            t15 = t14*s2
            t17 = t14*s1
            t2  = s*s
            ssig = (-2.0*t1*t2*s+3.0*t1*(s2+s1)*t2-6.0*t1*s2*s1*s+t15*F1-3.0*t17*F1-F2*t21+3.0*F2*s2*t20)/(-t21-3.0*t17+t15+3.0*s2*t20)
            ssig = si*ssig
            dsig = 6.0*(-t1*t2+t1*(s2+s1)*s-t1*s2*s1)/(-t20*s1-3.0*s1*t14+t14*s2+3.0*s2*t20)
        else:
            ssig = si*F2
            dsig = 0.
        sig =  array([ssig,0], dtype=double)
        MatM = array([[dsig,0],[0,0]], dtype=double)# material tangential stiffness
        return sig, MatM, [Eps[0], sig[0]]

class RCBeam(Material):                             # Reinforced concrete beam cross section (see Script Sec. 3.1.3)
    def __init__(self, PropMat):
        if PropMat[0][4]<1.e-6: raise NameError ("ConFemMaterials::RCBEAM.__init__: non zero tensile strength has to be defined for tension stiffening")
        self.PropMat = PropMat                      # [0] concrete (see below, [0][4] tensile strength-tension stiff only, [0][5] cross section integration), [1][] reinforcement
        self.NData = 8                              # number of data items in Elem.Data / DataP
        self.NullD = 1.*10**(-6)
        self.Reinf = MisesUniaxial( PropMat[1], [PropMat[0][4],1.3] )# initialization material for reinforcement + tension stiffening parameters (tensile strength, factor for stabilized cracking)
        self.ReinfTC = None
        self.StateVar = 5                           # number of state variables per integration point per concrete/reinforcement layer
                                                    # 0: current permanent strain upon unloading, 1: current yield stress, 2: current strain (just obsolete)
                                                    # 2: current reference strain for uniaxial stress strain curve, 3: final stress of last time step, see also mises for reinforcement
                                                    # StateVar of Mises should be the same
        self.Update = True                          # has specific update procedure for update of state variables
        self.Updat2 = False                         # no 2 stage update procedure
        self.alphaT = PropMat[0][6]                 # thermal expansion coefficient
        self.phi = PropMat[0][7]                    # creep number
        self.zeta = PropMat[0][8]                   # viscosity parameter
        self.psi = (1+self.phi)*self.zeta           # auxiliary value
        self.Density = PropMat[0][9]                # specific mass
        if self.PropMat[0][5]>0 : self.Symmetric = True # flag for symmetry of material matrices, PropMat[0][5] indicates number of integration over cross section, value 0 might be creeping which probably is not symmetric
        if len(PropMat[0])==11:
            self.fct = PropMat[0][10]               # quick and dirty introduction of concrete tensile strength via corresponding strain
            self.epsRef = self.fct/PropMat[0][0]
            self.epsFctCoeff = 1.5 # 2.            # strain to tensile strength
            self.epsLimCoeff = 3. # 3. # 5. # 6.        # strain to zero
            self.epsLim = self.epsLimCoeff*self.epsRef
            self.epsFct = self.epsFctCoeff*self.epsRef
        else: 
            self.epsLim = 0.
    
    def MatUniaxCo( self, PropM, eps):              # uniaxial concrete (see DIN1045-1 Sec. 9.1.5)
        Ec0     = PropM[0]                          # Concrete Young's modulus
        fc      = PropM[1]                          # compressive strength
        epsc1   = PropM[2]                          # strain at compressive strength
        epsc1u  = PropM[3]                          # ultimate compressive strain
        if True:
            if 0. < eps <= self.epsLim:
                sig, dsig = self.MatUniaxCoTens( eps, Ec0, self.fct)
            elif epsc1u<=eps and eps<=0.:
                if True:
                    eta = eps/epsc1                                 #DIN1045-1 Eq. (63)
                    k = -Ec0*epsc1/fc                               #DIN1045-1 Eq. (64)
                    sig = -fc * (k*eta-eta**2) / (1.+(k-2)*eta)     #DIN1045-1 Eq. (62)
                    dsig = -fc * ( (k-2*eta)/(1+(k-2)*eta) - (k*eta-eta**2)/(1+(k-2)*eta)**2*(k-2) ) / epsc1
                else:
                    if eps<0: si=-1
                    else:     si= 1
                    s = abs(eps)
                    s1 = -epsc1
                    F1 = fc
                    s2 = -epsc1u
                    F2 = 1.0*fc
                    s0 = 0.
                    d0 = Ec0
                    if s<= s0:
                        sig=si*d0*s
                        dsig=d0
                    elif s<= s1:
                        t5 = s*s
                        t8 = s0*s0
                        t9 = d0*t8
                        t11 = s0*F1
                        t14 = s0*s1*d0
                        t17 = 3.0*s1*F1
                        t18 = s1*s1
                        t19 = t18*d0
                        t20 = 2.0*t19
                        sig = (-(d0*s0+s1*d0-2.0*F1)*t5*s
                                   +(2.0*t9-3.0*t11+2.0*t14-t17+t20)*t5
                                   -s1*(t14+t19+4.0*t9-6.0*t11)*s
                                   +t8*(t20+t11-t17)) / (t8*s0+3.0*s0*t18-t18*s1-3.0*s1*t8)
                        sig = si*sig
                        dsig = (-3.0*(d0*s0+s1*d0-2.0*F1)*t5+2.0*(2.0*t9-3.0*t11+2.0*t14-3.0*s1*F1+2.0*t19)*s-s1*(t14+t19+4.0*t9-6.0*t11))/(t8*s0+3.0*s0*t18-t18*s1-3.0*s1*t8)
                    elif s<=s2:
                        t1  = -F1+F2
                        t20 = s1*s1
                        t21 = t20*s1
                        t14 = s2*s2
                        t15 = t14*s2
                        t17 = t14*s1
                        t2  = s*s
                        sig = (-2.0*t1*t2*s+3.0*t1*(s2+s1)*t2-6.0*t1*s2*s1*s+t15*F1-3.0*t17*F1-F2*t21+3.0*F2*s2*t20)/(-t21-3.0*t17+t15+3.0*s2*t20)
                        sig = si*sig
                        dsig = 6.0*(-t1*t2+t1*(s2+s1)*s-t1*s2*s1)/(-t20*s1-3.0*s1*t14+t14*s2+3.0*s2*t20)
                    else:
                        sig = si*F2
                        dsig = 0.
            else:
                sig = 0
                dsig = 0
        else:                                       # parabola square
            fcd = 28.3
            eps_c2 = 0.002
            if -0.002<=eps and eps<=0.:
                xyz = 1.-(-eps)/eps_c2
                sig = -fcd*(1.-xyz*xyz)
                dsig = 2*fcd*(1+eps/eps_c2)/eps_c2
                print sig, dsig
            elif -0.0035<=eps:
                sig = -fcd
                dsig = 0.
            else:
                sig = 0.
                dsig = 0. 
        return sig, dsig
    # approach taken from Spring adopted for tensile stress strain
    def MatUniaxCoTens(self, s, E0, fct):
        si = 1.
        eps_ref = fct/E0
        s1 = self.epsFctCoeff*eps_ref
        F1 = fct
        s2 = self.epsLimCoeff*eps_ref
        F2 = 0.
        s0 = 0. 
        d0 = E0
        if s<= s0:
            ssig=si*d0*s
            dsig=d0
        elif s<= s1:
            t5 = s*s
            t8 = s0*s0
            t9 = d0*t8
            t11 = s0*F1
            t14 = s0*s1*d0
            t17 = 3.0*s1*F1
            t18 = s1*s1
            t19 = t18*d0
            t20 = 2.0*t19
            ssig = (-(d0*s0+s1*d0-2.0*F1)*t5*s
                       +(2.0*t9-3.0*t11+2.0*t14-t17+t20)*t5
                       -s1*(t14+t19+4.0*t9-6.0*t11)*s
                       +t8*(t20+t11-t17)) / (t8*s0+3.0*s0*t18-t18*s1-3.0*s1*t8)
            ssig = si*ssig
            dsig = (-3.0*(d0*s0+s1*d0-2.0*F1)*t5+2.0*(2.0*t9-3.0*t11+2.0*t14-3.0*s1*F1+2.0*t19)*s-s1*(t14+t19+4.0*t9-6.0*t11))/(t8*s0+3.0*s0*t18-t18*s1-3.0*s1*t8)
        elif s<=s2:
            t1  = -F1+F2
            t20 = s1*s1
            t21 = t20*s1
            t14 = s2*s2
            t15 = t14*s2
            t17 = t14*s1
            t2  = s*s
            ssig = (-2.0*t1*t2*s+3.0*t1*(s2+s1)*t2-6.0*t1*s2*s1*s+t15*F1-3.0*t17*F1-F2*t21+3.0*F2*s2*t20)/(-t21-3.0*t17+t15+3.0*s2*t20)
            ssig = si*ssig
            dsig = 6.0*(-t1*t2+t1*(s2+s1)*s-t1*s2*s1)/(-t20*s1-3.0*s1*t14+t14*s2+3.0*s2*t20)
        else:
            ssig = si*F2
            dsig = 0.
        return ssig, dsig        
#    @profile
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        self.PropMat[0][1]=RandomField_Routines().get_property(Elem.Set,Elem.Label,self.PropMat[0][1])[0]
        if CalcType == 0: return [], [], []
        epsc = Eps[0]                               # strain on compressive side
        epss = Eps[0]                               # maximum reinforcement strain
        NN = 0.                                     # resulting normal force
        MM = 0.                                     # resulting moment
        NDE = 0.                                    # normal force gradient strain
        NDK = 0.                                    # normal force gradient curvature
        MDK = 0.                                    # moment gradient curvature
        if Elem.CrSecPolyL<>None: z1, z2, hh = Elem.Geom[1,1], Elem.Geom[1,2], Elem.Geom[1,2]-Elem.Geom[1,1] 
        else:                     z1, z2, hh = -Elem.Geom[1,2]/2, Elem.Geom[1,2]/2, Elem.Geom[1,2] # coordinate of bottom fibre, top fibre, height of cross section    
        Tr = 0.5*(Temp[0]+Temp[1])                  # temperature of reference / middle axis
        Tg = (Temp[0]-Temp[1])/hh                   # temperature gradient
        dTr= 0.5*(dTmp[0]+dTmp[1])                  # temperature increment of reference / middle axis
        dTg= (dTmp[0]-dTmp[1])/hh                   # temperature increment gradient

        nR =Elem.Geom.shape[0]-2                    # number of reinforcement layers
        for i in range(nR):                         # reinforcement contribution
            ipL = (1+nR)*ipI+(1+i)                  # every IP has state variables for concrete and nR reinforcement layers
            As = Elem.Geom[i+2,0]                   # reinforcement cross section
            ys = Elem.Geom[i+2,1]                   # coordinates of reinforcement
            eps = Eps[0] - ys*Eps[1]                # strain in reinforcement
            dep = Dps[0] - ys*Dps[1]                # strain increment in reinforcement
            tmp = Tr - ys*Tg                        # temperature in reinforcement
            dtp = Tr - ys*Tg                        # temperature incement in reinforcement
            if Elem.RType[i]=='TC':
                if len(Eps)==4: eps = (Eps[0]-Eps[2]) - ys*(Eps[1]-Eps[3])   # for material tester
                Sig, Dsig, dataL = self.ReinfTC.Sig( ff, CalcType, Dt, i, ipL, Elem, [dep, 0], [eps, 0], dtp, tmp, None)
            else:                   
                Sig, Dsig, dataL =   self.Reinf.Sig( ff, CalcType, Dt, i, ipL, Elem, [dep, 0], [eps, 0], dtp, tmp, None)
            NN = NN   + Sig[0] *As                  # reinforcement contribution to normal force (Script -> first term in Eq. (3.17))
            MM = MM   - Sig[0] *As*ys               # reinforcement contribution to moment (Script -> first term in Eq. (3.18))
            NDE = NDE + Dsig[0,0]*As                # reinf. contrib. to normal force grad eps (Script -> first term in Eq. (3.20)1)
            NDK = NDK - Dsig[0,0]*As*ys             # reinf. contrib. to normal force grad kappa (Script -> first term in Eq. (3.20)2)
            MDK = MDK + Dsig[0,0]*As*ys**2          # reinf. contrib. to moment grad kappa (Script -> first term in Eq. (3.21))
            Elem.StateVarN[ipL,2] = eps             # 
            if eps>epss: epss=eps                   # maximum reinforcement strain

        if fabs(Eps[1])>self.NullD:                  # contributing concrete part (Script Sec. 3.1.3 -> *Concrete compressive zone)
#            z0 = Eps[0]/Eps[1]                      # zero line - no contribution of concrete tensile strength
            z0 = (Eps[0]-self.epsLim)/Eps[1]         # zero line - with contribution of concrete tensile strength
            if Eps[1]>0:                            # upward curvature
                epsc = Eps[0]-z2*Eps[1]             # minimum compressive strain
                if z0 > z2: # hh/2:                       # predominant tension
                    z1 = z2  #hh/2
                    BZ = zeros((2,2))
                elif z0 >z1: # -hh/2:                     # predominant bending upward
                    z1 = z0
                    BZ=array([[1/Eps[1],-Eps[0]/(Eps[1]**2)],[0,0]])
                else: BZ = zeros((2,2))
            else:                                   # downward curvature
                epsc = Eps[0]-z1*Eps[1]             # minimum compressive strain
                if z0 < z1: # -hh/2:                       # predominant tension
                    z2 = z1 #-hh/2
                    BZ = zeros((2,2))
                elif z0 < z2: #hh/2:                     # predominant bending downward
                    z2 = z0
                    BZ=array([[0,0],[1/Eps[1],-Eps[0]/(Eps[1]**2)]])
                else: BZ = zeros((2,2))
        else: BZ = zeros((2,2))
        if epsc<self.PropMat[0][3]:
            print(Elem.Label,elI,ipI,'__',self.PropMat[0][3],epsc,"ConFemMaterials::RCBeam.Sig: allowed concrete stress exceeded")
#            raise NameError ("ConFemMaterials::RCBeam.Sig: allowed concrete compressive strain exceeded")

        Emod = self.PropMat[0][0]
        nI = int(self.PropMat[0][5])                # number of integration points (for integration over cross section)
        ipL = (1+nR)*ipI                            # every IP has state variables for concrete and nR reinforcement layers
        Elem.StateVarN[ipL,2] = epsc                # currently used by reference strain for uniaxial smoothed stress strain curve
        
#        if 0.0021<Eps[1] and Eps[1]<0.0023: PPFlag = True
#        else: PPFlag=False
        
        if nI>0:                                    # consideration of nonlinear behavior in compressive range
            dz = (z2-z1)/nI                         # cross sectional height / number of integration points
#            zz = zeros((nI+1), dtype=double)
            nn = zeros((nI+1), dtype=double)
            mm = zeros((nI+1), dtype=double)
            nde= zeros((nI+1), dtype=double)
            ndk= zeros((nI+1), dtype=double)
            mdk= zeros((nI+1), dtype=double)
            bb = Elem.Geom[1,1]                     # width of cross section, may eventually be overridden
            z = z1                                  # cross section / strain coordinate, initial value
            
#            if Elem.Label==12 and ipI==0:
#                print 'AAA', Elem.Label, ipI, Eps[0], Eps[1], '_', NN, MM, '_', NDE, NDK, MDK 
#            print 'AAA', z1,z2
            
            for i in xrange(nI+1):                  # numerical integration of concrete contribution
                if Elem.CrSecPolyL<>None: bb = Elem.WidthOfPolyline(z) # width of cross section defined by polyline
#                if Elem.Label==1 and ipI==0: 
#                    print 'X', z, bb
                ept = self.alphaT*(Tr - z*Tg)       # temperature strain in concrete fiber
                eps = Eps[0]-z*Eps[1]
                sig, dsig =self.MatUniaxCo( self.PropMat[0], Eps[0]-z*Eps[1]-ept) # ccurrent fiber stress
                sig_ = bb*sig
                dsig_= bb*dsig
#                zz[i]  = z                          # vertical coordinate
                nn[i]  =    sig_                    # normal force
                mm[i]  = -z*sig_                    # moment
                nde[i] =       dsig_                # normal force grad eps
                ndk[i] = -z   *dsig_                # normal force grad kappa
                mdk[i] =  z**2*dsig_                # moment grad kappa

#                if PPFlag: print 'XXX', Elem.Label, ipI, i, z, '_', eps, self.epsFct, '__', sig, dsig

                z = z + dz
#            NN = NN + integrate.trapz( nn, zz)      # numerical integration normal force (Script -> last term in Eq. (3.17))
#            MM = MM + integrate.trapz( mm, zz)      # numerical integration moment (Script -> last term in Eq. (3.18))
#            NDE = NDE + integrate.trapz( nde, zz )  # numerical normal force grad eps (Script -> last term in Eq. (3.20)1)
#            NDK1= NDK + integrate.trapz( ndk, zz )  # numerical normal force grad kappa (Script -> last term in Eq. (3.20)2)
#            MDK = MDK + integrate.trapz( mdk, zz )  # numerical moment grad kappa (Script -> last term in Eq. (3.21))
            NN = NN + integrate.trapz( nn, x=None, dx=dz) # numerical integration normal force (Script -> last term in Eq. (3.17))
            MM = MM + integrate.trapz( mm, x=None, dx=dz)      # numerical integration moment (Script -> last term in Eq. (3.18))
            NDE = NDE + integrate.trapz( nde, x=None, dx=dz)  # numerical normal force grad eps (Script -> last term in Eq. (3.20)1)
            NDK1= NDK + integrate.trapz( ndk, x=None, dx=dz)  # numerical normal force grad kappa (Script -> last term in Eq. (3.20)2)
            MDK = MDK + integrate.trapz( mdk, x=None, dx=dz)  # numerical moment grad kappa (Script -> last term in Eq. (3.21))
            NDK2= NDK1
            
#            if Elem.Label==12 and ipI==0:
#                print 'BBB', Elem.Label, ipI, Eps[0], Eps[1], '_', NN, MM, '_', NDE, NDK, MDK
            
#            if abs(NDK1-NDK2)>1.e-9: raise NameError("EEE") # control for symmetry of material matrix
        else:                                       # linear behavior in compressive zone, features creep
            psiI= 1/(1+Dt*self.psi)
            eps1 = Eps[0]-z1*Eps[1] - self.alphaT*(Tr-z1*Tg) # effective strain in lower concrete fiber
            eps2 = Eps[0]-z2*Eps[1] - self.alphaT*(Tr-z2*Tg) # effective strain in upper concrete fiber
            dep1 = Dps[0]-z1*Dps[1] - self.alphaT*(dTr-z1*dTg) # effective strain increment in lower concrete fiber
            dep2 = Dps[0]-z2*Dps[1] - self.alphaT*(dTr-z2*dTg) # effective strain increment in upper concrete fiber
            if abs(eps1)>self.NullD: sic1 = psiI*( Elem.StateVar[ipL,0] + Emod*dep1 + Dt*self.zeta*Emod*(eps1) )# implicit Euler integration !!! zeta inverse to zeta in book
            else: sic1 = 0.
            if abs(eps2)>self.NullD: sic2 = psiI*( Elem.StateVar[ipL,1] + Emod*dep2 + Dt*self.zeta*Emod*(eps2) )# implicit Euler integration !!!
            else: sic2 = 0.
            if Elem.CrSecPolyL<>None:
                AA, SS, JJ, hh = Elem.CrossSecGeom(z1,z2)# area, statical moment, inertial moment
                AS = 1./(z2-z1)*array([[AA*z2-SS,-AA*z1+SS],[-SS*z2+JJ,SS*z1-JJ]]) # A_sigma in book
                NN = NN + AS[0,0]*sic1 + AS[0,1]*sic2
                MM = MM + AS[1,0]*sic1 + AS[1,1]*sic2
                AZ = Elem.CrossSecGeomA(z1,z2,sic1,sic2)
            else:
                bb = Elem.Geom[1,1]                 # width of cross section
                AS = array([[bb*(z2-z1)/2,bb*(z2-z1)/2],[-bb*(z2+2*z1)*(z2-z1)/6,-bb*(2*z2+z1)*(z2-z1)/6]]) # A_sigma in book
                NN =  NN + bb*(z2-z1)*(sic1+sic2)/2
                MM =  MM - bb*( (z2+2*z1)*(z2-z1)*sic1 + (2*z2+z1)*(z2-z1)*sic2 )/6
                AZ = array([[-bb*(sic1+sic2)/2,bb*(sic1+sic2)/2],[bb*( (4*z1-z2)*sic1 + (2*z1+z2)*sic2 )/6, -bb*( (z1+2*z2)*sic1 + (-z1+4*z2)*sic2 )/6]])
            BE = array([[1,-z1],[1,-z2]])
            SE = Emod*dot(AS,BE)                    # material tangential stiffness 
            SZ = dot(AZ,BZ)                         # geometrical tangential stiffness
            NDE = NDE + SE[0,0] + SZ[0,0]
            NDK1= NDK + SE[0,1] + SZ[0,1]
            NDK2= NDK + SE[1,0] + SZ[1,0]
            MDK = MDK + SE[1,1] + SZ[1,1]
            Elem.StateVarN[ipL,0] = sic1            # update state variables
            Elem.StateVarN[ipL,1] = sic2            #
        if Elem.dim==10:                            # Bernoulli beam
            MatM = array( [[NDE,NDK1],[NDK2,MDK]] ) # material tangential stiffness (Script Eq. (3.21))
            sig = array( [NN,MM] )                  # stress vector
#            if elI==1 or elI==8:
#                print elI,ipI, Eps, sig
            return sig, MatM, [Eps[0],Eps[1],sig[0],sig[1],1000*epss,1000*epsc]
        elif Elem.dim==11:                          # Timoshenko beam
            MatM = array( [[NDE,NDK1,0],[NDK2,MDK,0],[0,0,bb*hh*0.9*Emod/4]] )
            sig = array( [NN,MM,MatM[2,2]*Eps[2]] ) # stress vector
            return sig, MatM, [Eps[0],Eps[1],Eps[2],sig[0],sig[1],sig[2],1000*epss,1000*epsc]
    def UpdateStateVar(self, Elem, ff):
        RList = []                                  # list of integration point indices for reinforcement 
        nR =Elem.Geom.shape[0]-2                    # number of reinforcement layers
        for j in xrange(Elem.nIntL):
            for i in xrange(nR): RList.append((1+nR)*j+(1+i)) # every IP j has state variables for concrete and nR reinforcement layers
        for j in xrange(Elem.StateVar.shape[0]):    # loop over all concrete and reiforcement integration point contributions
            if j in RList:                          # reinforcement integration contribution
                if Elem.StateVarN[j,1]>Elem.StateVar[j,1]: # plastic loading
                    Elem.StateVar[j,0] = Elem.StateVarN[j,0]
                    Elem.StateVar[j,1] = Elem.StateVarN[j,1]
                Elem.StateVar[j,2] = Elem.StateVarN[j,2] # strain of fiber
                Elem.StateVar[j,3] = Elem.StateVarN[j,3] # for smoothed uniaxial version of uniaxial mises for reinforcement
                Elem.StateVar[j,4] = Elem.StateVarN[j,4] # "
            else: 
                Elem.StateVar[j] = Elem.StateVarN[j] # concrete integration contribution
        return False

class TextileR(Material):
    def __init__(self, PropMat):
#    def __init__(self,         SymmetricVal, RTypeVal, UpdateVal, Updat2Val, StateVarVal, NDataVal):
        Material.__init__(self, True,         None,     True,      False,     1,           2)
#        self.Symmetric = True                       # flag for symmetry of material matrices
#        self.Update = True                          # has specific update procedure for update of state variables
#        self.Updat2 = False                         # no 2 stage update procedure
#        self.StateVar = 1                           # number of state variables per integration point
#        self.NData = 2                              # number of data items
        self.PropMat = PropMat
#        self.StateVar = None
#        self.NData = 2                              # number of data items
        self.alphaT = 0                             # thermal expansion coefficient
        self.Density = 0                            # specific mass
        self.Emod = PropMat[0]                      # Young's modulus
        self.eps_tund = PropMat[1]                  # PropMat[1] is strain 
        self.sig_tund = self.eps_tund*self.Emod     
        self.sig_tu = PropMat[2]                    # failure stress
        self.eps_tu = PropMat[3]                    # failure strain
        if self.eps_tu==self.eps_tund:
            self.linearity=1
            self.Etan = self.Emod 
        else:
            self.linearity=2
            self.Etan = (self.sig_tu-self.sig_tund)/(self.eps_tu-self.eps_tund) 
        
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
#        self.sig_tu=RandomField_Routines().get_property(Elem.Set,Elem.Label,self.sig_tu)[-1] #sig_tu is overwritten by the random value
#                                                                                            #chose of the last value!!!
#        if self.eps_tu==self.eps_tund:
#            self.linearity=1
#            self.Etan = self.Emod 
#        else:
#            self.linearity=2
#            self.Etan = (self.sig_tu-self.sig_tund)/(self.eps_tu-self.eps_tund) 
   
        if Eps[0]<=self.eps_tund:
            MatM = array([[self.Emod,0],[0,0]])          # material stiffness uniaxial
            sig = [self.Emod*Eps[0],0.]
        elif Eps[0]<=self.eps_tu and self.linearity==2:
            MatM = array([[self.Etan,0],[0,0]]) 
            sig  = [(Eps[0]-self.eps_tund)*self.Etan+self.sig_tund,0]
        else: 
            MatM = array([[self.Etan,0],[0,0]])          # material stiffness uniaxial
            sig  = [(Eps[0]-self.eps_tund)*self.Etan+self.sig_tund,0] 
        Elem.StateVarN[ipI][0] = sig[0] 
        return sig, MatM, [Eps[0], sig[0], 0.]          # last value to make this consistent with WraRCShell

class WraTCBeam(RCBeam):                                # Textile Reinforced concrete beam
    def __init__(self, PropMat):
        RCBeam.__init__(self, [PropMat[0], PropMat[1]])
        self.ReinfTC = TextileR( PropMat[2] )           # used as additional option by RCBeam

class WraRCShell(Material):                             # Wrapper for reinforced shell
    def __init__(self, PropMat, MatPointer, ReMatType):
        if isinstance( MatPointer, ElasticLT):  
            self.Conc = ElasticLT(PropMat[0])
            fct = PropMat[0][2]                         # for tension stiffening
        elif isinstance( MatPointer, IsoDamage):
            self.Conc = IsoDamage(PropMat[0])
            fct = self.Conc.fct                         # for tension stiffening
        elif isinstance( MatPointer, Elastic):
            self.Conc = Elastic(PropMat[0])
            self.Conc.NData = 8                         # to make this consistent with ElasticLT and IsoDamage
            fct = 0.
        elif isinstance( MatPointer, MicroPlane):
            self.Conc = MicroPlane(PropMat[0])
            self.Conc.NData = 8                         # to make this consistent with ElasticLT and IsoDamage
            fct = 0.
        else: raise NameError ("ConFemMaterials::WraRCShell.__init__: material type not regarded")
        self.StateVar = self.Conc.StateVar              # number of state variables per integration point
        self.NData  = self.Conc.NData                   # number of data items, subject to change in ConFemInOut::ReadInputFile for input of SH4.
        self.Update = self.Conc.Update
        self.Updat2 = self.Conc.Updat2
        self.RType  = self.Conc.RType                   # regularization type for concrete
        self.ReinfType = None                           # reinforcement type
        if self.RType==2:
            self.bw = self.Conc.bw
            self.CrX= self.Conc.CrX
            self.CrY= self.Conc.CrY
            self.CrBwN = self.Conc.CrBwN
        if ReMatType == "TR":                           # leads to TextileR, but input syntax slightly different with corresponding input syntax of TCBEAM
            E_0, sig_1, E_1, sig_u = PropMat[1][0], PropMat[1][2], PropMat[1][3], PropMat[1][4]
            eps_1 = sig_1/E_0
            eps_u = eps_1 + (sig_u-sig_1)/E_1  
            self.Reinf  = TextileR( [ E_0 , eps_1 , sig_u , eps_u ] ) # to make this compatible with init of TextileR
            self.ReinfType = 'TR'
        else:                 
            self.Reinf  = MisesUniaxial( PropMat[1], [fct,1.3])  # initialization material for reinforcement + tension stiffening parameters
            self.ReinfType = 'Mises'
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        if ipI<Elem.nIntLi:                        # initial value of number of bulk integration points before extension for reinforcement 
            SigL, MatL, dataL = self.Conc.Sig( ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, []) # 4/5 integration points over cross section height
        else:                                       # reinforcement contributions
            if CalcType == 0: return [], [], []
            As = SampleWeightRCShell[Elem.Set,Elem.IntT,Elem.nInt-1,ipI]
            if As>ZeroD:
                Elem.dim=1                              # for uniaxial reiforcement constitutive law
                Dps_ = array([Dps[0],Dps[1],Dps[5]])
                Eps_ = array([Eps[0],Eps[1],Eps[5]])
                if   Elem.nInt==2: j = (ipI-16)//4      # floor division -> integer value -> index for reinforcement layer, 4 stands for number of integration points in base area
                elif Elem.nInt>=5: j = (ipI-20)//4   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                phi = Elem.Geom[2+j,4]*pi/180.          # angle of orientation of reinforcement
                Trans=array([[cos(phi)**2,sin(phi)**2, cos(phi)*sin(phi)],
                             [sin(phi)**2,cos(phi)**2,-cos(phi)*sin(phi)],
                             [-2*cos(phi)*sin(phi),2*cos(phi)*sin(phi),cos(phi)**2-sin(phi)**2]])# transformation matrix for strains from global to local
                MStiff=array([[cos(phi)**4,cos(phi)**2*sin(phi)**2,cos(phi)**3*sin(phi)],
                              [cos(phi)**2*sin(phi)**2,sin(phi)**4,cos(phi)*sin(phi)**3],
                              [cos(phi)**3*sin(phi),cos(phi)*sin(phi)**3,cos(phi)**2*sin(phi)**2]])
                dpsL = dot(Trans,Dps_)                  # local strain increment
                epsL = dot(Trans,Eps_)                  # local strain
                SigL, MatL, dataL = self.Reinf.Sig( ff, CalcType, Dt, elI, ipI, Elem, dpsL, epsL, dTmp, Temp, None)
                MatM = MatL[0,0]*MStiff                 # anisotropic local tangential material stiffness
                sig = dot(Trans.transpose(),[SigL[0],0.,0.]) # global stress times thickness
                # make data consistent in shell SH4 element
                Elem.dim=21
                SigL = array([sig[0],sig[1],0.,0.,0.,sig[2]]) # ([0.,0.,0.,0.,0.,0.]) #
                MatL = array([[MatM[0,0],MatM[0,1],0.,0.,0.,MatM[0,2]],
                              [MatM[1,0],MatM[1,1],0.,0.,0.,MatM[1,2]],
                              [0.,0.,0.,0.,0.,0.],
                              [0.,0.,0.,0.,0.,0.],
                              [0.,0.,0.,0.,0.,0.],
                              [MatM[2,0],MatM[2,1],0.,0.,0.,MatM[2,2]]])
            else:
                MatL = zeros((6,6), dtype=float)
                SigL = zeros((6), dtype=float)
                dataL = zeros((3), dtype=float)
            dataL = [SigL[0],SigL[1],SigL[2],SigL[3],SigL[4],SigL[5],dataL[0],dataL[1],dataL[2],0.,0.,0.,0.,0.]# trailing zeros to make this consistent with data format of ElasticLT 2d
        # different dataL for bulk and reinforcement !
        return SigL, MatL, dataL
    def UpdateStateVar(self, Elem, ff):
        Flag = self.Conc.UpdateStateVar(Elem, ff)
        if Elem.Type=='SH4': 
            if   Elem.nInt==2: nn = range(16,Elem.StateVar.shape[0]) # for reinforcement type mises/RC only
            elif Elem.nInt==5: nn = range(20,Elem.StateVar.shape[0]) # "
            if self.ReinfType == "Mises":
                for j in nn:                          # loop over integration points for reinforcement only
                    if Elem.StateVarN[j,1]>Elem.StateVar[j,1]: Elem.StateVar[j] = Elem.StateVarN[j]
            if self.ReinfType == "TR":
                for j in nn:
                    if abs(Elem.StateVarN[j,0])>abs(Elem.StateVar[j,0]): Elem.StateVar[j] = Elem.StateVarN[j]
                    if abs(Elem.StateVarN[j,0])>self.Reinf.sig_tu: print >> ff, 'WraRCShell::TR: strength exceeded', Elem.Label, j, Elem.StateVar[j,0]  
        return Flag
    def UpdateStat2Var(self, Elem, ff, Flag, LoFl):
        self.Conc.UpdateStat2Var(Elem, ff, Flag, LoFl)
    
class WraMisesReMem(Material):                      # anisotropic reinforcement membrane with elastoplastic Mises
    def __init__(self, PropMat, val):
        self.PropMat = PropMat
        self.Density = PropMat[6]                   # specific mass
        self.NData = 6                              # number of data items
        # self.Reinf = Mises( PropMat, [0.0,1.3] )
        self.Reinf = MisesUniaxial( PropMat, [0.0,1.3] )    # initialization material for reinforcement + tension stiffening parameters
        self.StateVar = self.Reinf.StateVar
        self.Update = True                          # has specific update procedure for update of state variables
        self.Updat2 = False                         # no 2 stage update procedure
        phi = PropMat[1]*pi/180.                    # angle of orientation of reinforcement
        self.Trans=array([[cos(phi)**2,sin(phi)**2, cos(phi)*sin(phi)],
                          [sin(phi)**2,cos(phi)**2,-cos(phi)*sin(phi)],
                          [-2*cos(phi)*sin(phi),2*cos(phi)*sin(phi),cos(phi)**2-sin(phi)**2]])# transformation matrix for strains from global to local
        self.MStiff=array([[cos(phi)**4,cos(phi)**2*sin(phi)**2,cos(phi)**3*sin(phi)],
                          [cos(phi)**2*sin(phi)**2,sin(phi)**4,cos(phi)*sin(phi)**3],
                          [cos(phi)**3*sin(phi),cos(phi)*sin(phi)**3,cos(phi)**2*sin(phi)**2]])
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        if CalcType == 0: return [], [], []
        Emod = self.PropMat[0]
        nu = 0.2                                    # Poissons ratio fixed here !
        if Elem.dim==2:                             # plane strain biaxial
            dpsL = dot(self.Trans,Dps)              # local strain increment
            epsL = dot(self.Trans,Eps)              # local strain
            Elem.dim=1
            SigL, MatL, dataL = self.Reinf.Sig( ff, CalcType, Dt, elI, ipI, Elem, dpsL, epsL, dTmp, Temp, EpsR)
            Elem.dim=2
            MatM = MatL[0,0]*self.MStiff            # anisotropic local tangential material stiffness
            sig = dot(self.Trans.transpose(),[SigL[0],0.,0.]) # global stress times thickness
            return sig, MatM, [Eps[0], Eps[1], Eps[2], sig[0], sig[1], sig[2]] # !!!!!!!
        else: raise NameError ("ConFemMaterials::WraMisesReMem.Sig: misesReMem does not match element type")
    def UpdateStateVar(self, Elem, ff):
        for j in xrange(Elem.StateVar.shape[0]):# loop over all integration and integration sub points
            if abs(Elem.StateVarN[j,4])>abs(Elem.StateVar[j,4]): Elem.StateVar[j] = Elem.StateVarN[j]
        return False

class NLSlab(Material):                             # Simple nonlinear Kirchhoff slab
    def __init__(self, PropMat):
        self.KX = PropMat[0]                        # initial bending stiffness
        self.KY = PropMat[1]                        # initial bending stiffness
        self.Mx_y = PropMat[2]                      # initial yield moment x
        self.My_y = PropMat[3]                      # initial yield moment y
        self.KTX = PropMat[4]                       # hardening stiffness
        self.KTY = PropMat[5]                       # hardening stiffness
        self.alpha = PropMat[6]                     # factor for twisting stiffness
        self.StateVar = 7 # 0 plastic curv x, 1 current moment x, 2 yield mom x, 3-5 same y, 6 actual twisting moment  
        self.NData = 6                              # number of data items
        self.Update = True                          # has specific update procedure for update of state variables
        self.Updat2 = False                         # no 2 stage update procedure
    def Sig(self, ff, CalcType, Dt, elI, ipI, Elem, Dps, Eps, dTmp, Temp, EpsR):
        if CalcType == 0: return [], [], []
        if Elem.dim==20:                            # Kirchhoff slab
            if Elem.StateVar[ipI][2]==0: momYx=self.Mx_y   # for first calcul
            else:                        momYx=Elem.StateVar[ipI][2]#self.Mx_y
            if Elem.StateVar[ipI][5]==0: momYy=self.My_y
            else:                        momYy=Elem.StateVar[ipI][5] # self.My_y

            epsPx, epsPy = Elem.StateVar[ipI][0], Elem.StateVar[ipI][3] # permanent curvature at the end of of previous time increment
            sigOld = array([Elem.StateVar[ipI][1],Elem.StateVar[ipI][4],Elem.StateVar[ipI][6]]) # moments at the end of of previous time increment
            MatM = array([[self.KX,0,0],[0,self.KY,0],[0,0,0]])# twisting stiffness. 33 should be strictly divided by 2 and B_xy doubled, see SB3.FormB.
            Tx, Ty = self.KTX, self.KTY             # for twisting stiffness

            if   Eps[0]<=(epsPx-momYx/self.KX): MatM[0,0] = self.KTX    # elasto-plastic range in compressive loading range
            elif Eps[0]<=(epsPx+momYx/self.KX): Tx = self.KX            # elastic in unloading range
            else:                               MatM[0,0] = self.KTX    # elasto-plastic in tensile loading range
            if   Eps[1]<=(epsPy-momYy/self.KY): MatM[1,1] = self.KTY
            elif Eps[1]<=(epsPy+momYy/self.KY): Ty = self.KY
            else:                               MatM[1,1] = self.KTY
            MatM[2,2] = self.alpha*sqrt(Tx*Ty)                 # final value twisting stiffness
            sig = sigOld + dot(MatM,Dps)
            Elem.StateVarN[ipI][1],Elem.StateVarN[ipI][4],Elem.StateVarN[ipI][6] = sig[0],sig[1],sig[2] # new stress as state variable
            if abs(sig[0])>abs(momYx):
                Elem.StateVarN[ipI][0] = Eps[0]-sig[0]/self.KX # new permanent curvature 
                Elem.StateVarN[ipI][2] = sig[0]     # new yield moment
            if abs(sig[1])>abs(momYy):
                Elem.StateVarN[ipI][3] = Eps[1]-sig[1]/self.KY # new permanent curvature 
                Elem.StateVarN[ipI][5] = sig[1]     # new yield moment
            return sig, MatM, [Eps[0],Eps[1],Eps[2],sig[0],sig[1],sig[2]]
        else: raise NameError ("ConFemMaterials::NLSlab.Sig: not implemented for this element type")
    def UpdateStateVar(self, Elem, ff):
        for j in xrange(Elem.StateVar.shape[0]):    # loop over all integration and integration sub points
            Elem.StateVar[j] = Elem.StateVarN[j]
#            Elem.StateVar[j][1] = Elem.StateVarN[j][1] # mx
#            Elem.StateVar[j][4] = Elem.StateVarN[j][4] # my
#            Elem.StateVar[j][6] = Elem.StateVarN[j][6] # mxy
        return False

class MatTester():
    def __init__(self):
        self.ZeroD = 1.e-9                                           # Smallest float for division by Zero

    def Cplot(self, x, y, z, label, col ): 
        FonSizAx='large'     # fontsize axis
        LiWiCrv=3            # line width curve in pt
        PP = plt.figure()
        P0 = PP.add_subplot(111,title=label)
        if col <>None:  
            for i in xrange(len(x)): P0.plot(x[i], y[i], col, linewidth=LiWiCrv)
        else: 
            for i in xrange(len(x)): P0.plot(x[i], y[i], linewidth=LiWiCrv)
        P0.tick_params(axis='x', labelsize=FonSizAx)
        P0.tick_params(axis='y', labelsize=FonSizAx) #'x-large')
        P0.grid()
        if z <> None:
            P1 = P0.twinx()
            for i in xrange(len(x)): P1.plot(x[i], z[i], 'blue')
        PP.autofmt_xdate()

    def AmpVal(self, Time, Data):
        nD = len(Data)
        if Time>=Data[nD-1][0]:                         # time exceeds last value
            DelT = Data[nD-1][0]-Data[nD-2][0]
            if DelT<self.ZeroD: raise NameError ("ConFemMaterials::MatTester.AmpVal: something wrong with amplitude 1")
            return (Time-Data[nD-2][0])/DelT * (Data[nD-1][1]-Data[nD-2][1]) + Data[nD-2][1]
        for i in xrange(len(Data)-1):
            if Data[i][0]<=Time and Time<=Data[i+1][0]: # time within interval
                DelT = Data[i+1][0]-Data[i][0]
                if DelT<self.ZeroD: raise NameError ("ConFemMaterials::MatTester.AmpVal: something wrong with amplitude 2")
                return (Time-Data[i][0])/DelT * (Data[i+1][1]-Data[i][1]) + Data[i][1]
    def AmpVal2(selfself, Time):                        # exponential function with decreasing derivative
        if Time<=1.:
            return Time
        else:      
            a, b  = 5.687010672, .2133556056           # see maple preprocessor
            return (a-1.)*(1-exp(-b*(Time-1.)))+1.
    def RunMatT(self, MatLi, Elem, PreL, PlotF):
        import lib_Mat
        RListeAll = []
        if isinstance(MatLi, lib_Mat.RCBeam) or isinstance(MatLi, lib_Mat.WraTCBeam) or\
           isinstance( MatLi, RCBeam) or isinstance( MatLi, WraTCBeam):
            print 'R/TCBeam', MatLi.PropMat
            if MatLi.ReinfTC<>None: print MatLi.ReinfTC.PropMat
            e1, e2 = MatLi.PropMat[0][3], 1.5*MatLi.epsLim
            nn = 200
            de = (e2-e1)/nn
            ep, si, ds = [], [], []
            for i in range(nn+1):                           # stress strain curve
                eps = e1 + i*de
                sig, dsig = MatLi.MatUniaxCo( MatLi.PropMat[0], eps)
                ep += [1000.*eps]
                si += [sig]
                ds += [dsig]
            if PlotF: 
                self.Cplot( [ep], [si], [ds], 'eps-sig','r-')
        
            kk, N, M, EJ = [], [], [], []
#            kapD = 0.0002
#            kapD = 0.00002
#            kapD = 0.00001
            kapD = 0.0002
            Deps0 = array([0.000001, 0,      0, 0])
            Deps1 = array([0,        0.00001, 0, 0])
            f1=open('Detailergebnisse.txt','w')
            for NorT in [ 0. ]: #, -1., -2.]:
                kk_, N_, M_, EJ_, sig  = [], [], [], [], [0,0]
                kap, epsL, epsLold, maxM, kap_, epsL_ = -kapD, 0, 0, 0, 0, 0
                RListe = zeros((5+Elem.Geom.shape[0]-2), dtype=double)          # storage for results
                Flag = True
                while Flag:
                    Mold= sig[1]
                    kap = kap + kapD
                    epsLold = epsL
                    for j in range(20):
                        eps = array([epsL, kap, epsL_, kap_])
                        sig, MatM, dataS = MatLi.Sig(None,1,0.0,0,0, Elem,[epsL-epsLold,kapD],eps, [0.,0.], [0.,0.], None)
                        sig0,MatM0,dataS0 = MatLi.Sig(None,1,0.0,0,0, Elem,[epsL-epsLold,kapD],eps-Deps0, [0.,0.], [0.,0.], None)
                        sig1,MatM1,dataS1 = MatLi.Sig(None,1,0.0,0,0, Elem,[epsL-epsLold,kapD],eps-Deps1, [0.,0.], [0.,0.], None)
                        MM00 = (sig[0]-sig0[0])/Deps0[0]
                        MM01 = (sig[0]-sig1[0])/Deps1[1]
                        MM10 = (sig[1]-sig0[1])/Deps0[0]
                        MM11 = (sig[1]-sig1[1])/Deps1[1]
                        Res = sig[0] - NorT
#                        if fabs(Res)<0.00001: break
                        if fabs(Res)<1.e-5: break
                        epsL = epsL - (1/MatM[0,0]) * Res
#                    Flag_ = MatLi.UpdateStateVar(Elem, None)     # for unloading only
                    if abs(sig[1])<PreL: epsL_, kap_ = epsL, kap         
                    if Elem.StateVarN[0,2]<0.95*MatLi.PropMat[0][3]: Flag = False  # keep sign in mind 0.98
#                    if Elem.StateVarN[0,2]<0.10*MatLi.PropMat[0][3]: Flag = False  # keep sign in mind 0.98
                    for i in range(Elem.Geom.shape[0]-2):       # control of reinforcement strains over number of reinforcement layers
                        if Elem.RType[i]=='RC' and abs(Elem.StateVarN[i+1,2])>0.99*MatLi.PropMat[1][4]: Flag = False
                        if Elem.RType[i]=='TC' and abs(Elem.StateVarN[i+1,2])>0.99*MatLi.ReinfTC.PropMat[3]: Flag = False
                    if not Flag: break
#                        kapD = -kapD #break                      # for unloading only
#                        Flag = True                                  # for unloading only
#                        print 'Hello', kapD, sig[1]                   # for unloading only
#                    if (kapD<0 and sig[1]<0.05): break               # for unloading only
                    dMom1 = sig[1]-Mold
                    dMom2 = MatM[1,0]*(epsL-epsLold)+MatM[1,1]*kapD
                    dMomE = ((dMom1-dMom2)/dMom1)
                    print j, PreL, eps, sig, '__',   MatM[0,0], MM00, '_', MatM[0,1], MM01,'_', MatM[1,0], MM10, '_', MatM[1,1], MM11, '__', dMom1, dMom2, dMomE
                    if abs(dMomE)>0.10: print dMomE
                    print >> f1, j, PreL, eps[1], sig, dataS[4], dataS[5], 'B', sig[1]/eps[1]
                    kk_ += [kap]
                    N_  += [sig[0]]
                    M_  += [sig[1]]
                    EJ_ += [MatM[1,1]]
                    if abs(sig[1])>RListe[3]:                   # intermediate storage of actually relevant results
                        RListe[0]=epsL; RListe[1]=kap; RListe[2]=sig[0]; RListe[3]=sig[1]; RListe[4]=Elem.StateVarN[0,2]
                        for i in range(Elem.Geom.shape[0]-2): RListe[5+i]=Elem.StateVarN[i+1,2]
                kk += [kk_]
                N  += [N_]
                M  += [M_]
                EJ += [EJ_]
                RListeAll += [RListe]
            f1.close
            if PlotF: self.Cplot(kk,N, None, 'normal force-curvature', None); self.Cplot(kk,M,EJ,'moment-curvature', None)
            if PlotF: plt.show()
            print 'finished R/TCBeam'
        elif isinstance(MatLi, lib_Mat.ElasticLT):
            print 'ElasticLT', MatLi.Emod, MatLi.nu, MatLi.fct, MatLi.Gf, MatLi.epsct 
            N = 100
            de = MatLi.epsct
            deps= [de/N,0,0]
            Eps, Sig, Emo, WW, Eps_, Sig_, Emo_, WW_, Tim, Tim_ = [], [], [], [], [], [], [], [], [], []
            Eps1, Sig1, Emo1, WW1, Eps1_, Sig1_, Emo1_, WW1_ = [], [], [], [], [], [], [], []
#           Amp = [[0.,0.],[9.,9.]]
#            Amp = [[0.,0.],[4.,4.],[8.5,-0.5],[9.,0.]]
            Amp = [[0.,0.],[2.,2.],[4.5,-0.5],[5.,0.]]
            Amp1 = [[0.,0.],[23.,0.],[30.,7.],[32.,-0.5],[33.,0.]]
            State_, State1_  = 0, 0
            for i in xrange(int(15*N)):                          # 9, 23, 50 
                fac, fac1 = self.AmpVal(float(i)/N, Amp), self.AmpVal(float(i)/N, Amp1)
#                fac, fac1 = self.AmpVal2(float(i)/N), 0
                index, eps = 0, [fac*de,fac1*de,0]
                sigI, C, Data = MatLi.Sig( None, 0, 0.01,0,0, Elem, deps, eps, 0, 0, None)
                sigI, C, Data = MatLi.Sig( None, 1, 0.01,0,0, Elem, deps, eps, 0, 0, None)# stress, stress incr, tangential stiffness (Script Eq. (1.20) -> linear-elastic material;
                Flag_ = MatLi.UpdateStateVar(Elem, None)
                MatLi.UpdateStat2Var(        Elem, None, True, True)
                for k in xrange(len(Data)): Elem.DataP[0][k] = Data[k]

                State = int(Elem.StateVarN[0,0])
                State1 = int(Elem.StateVarN[0,6])
                if State<>State_:
                    Eps.append(Eps_); Sig.append(Sig_); Emo.append(Emo_); WW.append(WW_); Tim.append(Tim_) 
                    Eps_, Sig_, Emo_, WW_, Tim_ = [], [], [], [], []
                    State_ = State
                if State1<>State1_:
                    Eps1.append(Eps1_); Sig1.append(Sig1_); Emo1.append(Emo1_); WW1.append(WW1_)
                    Eps1_, Sig1_, Emo1_, WW1_ = [], [], [], []
                    State1_ = State1
                    
                Sig_.append(sigI[0]); Eps_.append(eps[0]); Emo_.append(C[0,0]); WW_.append(Data[6]); Tim_.append(float(i)/N)
                Sig1_.append(sigI[1]); Eps1_.append(eps[1]); Emo1_.append(C[1,1]); WW1_.append(Data[6])
                print i,State,fac,eps, sigI, Data[6], C[0,0] 
            
            Eps.append(Eps_); Sig.append(Sig_); Emo.append(Emo_); WW.append(WW_); Tim.append(Tim_)
            Eps1.append(Eps1_); Sig1.append(Sig1_); Emo1.append(Emo1_); WW1.append(WW1_)
            if PlotF: 
                self.Cplot(Eps,Sig,'eps0-sig0',None)
                self.Cplot(Eps1,Sig1,'eps1-sig1',None)
                self.Cplot(Eps,Emo,'eps-Emod',None)
                self.Cplot(WW,Sig,'w-sig',None)
                self.Cplot(Eps,WW,'eps-w',None)
                self.Cplot(Tim,Eps,'time-eps',None)
                plt.show()
            print 'Finished ElasticLT'
        else:
            print 'Unknown material type'
        return RListeAll 

# for material testing purpose
if __name__ == "__main__":
    from ConFEM2D_InOut import *
    X = MatTester()
#    Name, MType, ElSet = "../DataExamples/E6-03/E6-03", 'MAT2', 'EL1'                                # choose material type !
#    Name, MType, ElSet ="../TestIn/Sample0_", 'CONCRETE', 'B23E0' 
#    Name, MType, ElSet ="../DataTest/Sandwich/KernPU", 'MAT1', 'ELU'           # choose material type
    f1=open( Name+".in.txt", 'r')
    NodeList, ElList, MatList, StepList = ReadInputFile(f1, False)
    f1.close()
    MatLi = MatList[MType]
    for Elem in ElList:                       # loop over all elements
        if Elem.Set == ElSet: 
            if Elem.MatN==MType:
                Elem.Ini2( NodeList, MatList )
                print dir(Elem)
                X.RunMatT( MatLi, Elem, 0, True )
                break
