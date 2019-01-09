# ConPlaD
"""Special module for reinforcement design of plate/slab element type"""
from math import *
from time import *
from numpy import *
from scipy.linalg import *
from scipy import sparse
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse.linalg import aslinearoperator
import matplotlib.pyplot as plt
from os import path as pth

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

def Reinforcement2D( ElList, NodeList, ReinfDes, VecScales, Key, ff):
    def MSIG(pp):
        if pp[0]<0:
            phi = arctan(pp[2]/pp[1])                   # direction of principal compressive stress
            sig = pp[0]                                 # value of principal compressive stress
        elif pp[3]<0:
            if abs(pp[4])>ZeroD: phi = atan(pp[5]/pp[4])
            else:                phi = -(0.5*pi-atan(pp[4]/pp[5]))
            sig = pp[3]
        else:
            phi = None
            sig = None
        return phi, sig
    def KIXC( sigxy, rhox, xx):                         # NR iteration matrix for reinforcement in x-direction sigsy prescribed, sigsx unknown 
        sigc=xx[0]
        phi = xx[2]
        KI = array([[-2*sigc**2*cos(phi)/(-2*sigxy*cos(phi)-sin(phi)**3*sigc+sin(phi)*sigc*cos(phi)**2),0,
(-sin(phi)**2+cos(phi)**2)/sin(phi)/(-2*sigxy*cos(phi)-sin(phi)**3*sigc+sin(phi)*sigc*cos(phi)**2)*sigc],
[2*cos(phi)*sigc**2*(cos(phi)**2+sin(phi)**2)/rhox/(-2*sigxy*cos(phi)-sin(phi)**3*sigc+sin(phi)*sigc*cos(phi)**2),1/rhox,
 -cos(phi)*(2*sigxy*sin(phi)-cos(phi)*sigc*sin(phi)**2+cos(phi)**3*sigc)/rhox/sin(phi)/(-2*sigxy*cos(phi)-sin(phi)**3*sigc+sin(phi)*sigc*cos(phi)**2)],
[sin(phi)/(-2*sigxy*cos(phi)-sin(phi)**3*sigc+sin(phi)*sigc*cos(phi)**2)*sigc,0,
 -sigxy/sigc/sin(phi)/(-2*sigxy*cos(phi)-sin(phi)**3*sigc+sin(phi)*sigc*cos(phi)**2)]])
        return KI
    def KIYC( sigxy, rhoy, xx):                         # NR iteration matrix for reinforcement in x-direction sigsx prescribed, sigsy unknown 
        sigc=xx[0]
        phi = xx[2]
        KI = array([[2*sigc**2*sin(phi)/(2*sigxy*sin(phi)-cos(phi)*sigc*sin(phi)**2+cos(phi)**3*sigc),
-(sin(phi)**2-cos(phi)**2)*sigc/cos(phi)/(2*sigxy*sin(phi)-cos(phi)*sigc*sin(phi)**2+cos(phi)**3*sigc),0],
 [-2*sigc**2*sin(phi)*(cos(phi)**2+sin(phi)**2)/rhoy/(2*sigxy*sin(phi)-cos(phi)*sigc*sin(phi)**2+cos(phi)**3*sigc),
  sin(phi)*(2*sigxy*cos(phi)+sin(phi)**3*sigc-sin(phi)*sigc*cos(phi)**2)/cos(phi)/rhoy/(2*sigxy*sin(phi)-cos(phi)*sigc*sin(phi)**2+cos(phi)**3*sigc),1/rhoy],
 [cos(phi)/(2*sigxy*sin(phi)-cos(phi)*sigc*sin(phi)**2+cos(phi)**3*sigc)*sigc,
  -sigxy/sigc/cos(phi)/(2*sigxy*sin(phi)-cos(phi)*sigc*sin(phi)**2+cos(phi)**3*sigc),0]])
        return KI
    def SLABRE(mx, my, mxy, fy, fcd, dd):               # bending reinforcement for slabs
        Tol = 1.e-6
        pp = PrinC( mx, my, mxy)                        # principal stresses
        if pp[0]<=0 and pp[3]<=0: return 0,0,0,0
        chi = 0.95
        kk = 0.8
        fc_ = chi*kk*fcd
        N1 = 21
        if mxy>0: phic=-pi/4                            # prescription of compressive principal stress direction
        else:     phic= pi/4
        mc = -2*abs(mxy)                                # default value for concrete moment
        indi = 2*kk*mc/fc_/dd**2                           # indicator for concrete demand
        if indi<-0.99: NameError ("Reinforcement2D: slab concrete demand")
        zz = 0.5*dd*(1+sqrt(1+indi))                    # internal lever arm
        as1 = (mx -mc*cos(phic)**2)/(zz*fy)             # reinforcement x - direction
        as2 = (my -mc*sin(phic)**2)/(zz*fy)             # reinforcement y - direction
        if as1<0 and as2<0: raise NameError ("Reinforcement2D: unexpected condition")
        if as1<0 or as2<0:                              # not a reasonable reinforcement value, iteration for concrete strut direction necessary
            if abs(mxy)<0.01:                           # improve iteration starter
                phic = atan(pp[5]/pp[4])
                mc = pp[3]
            if as1<0: xx = array([mc,as2,zz,phic])      # iteration initial value
            else:     xx = array([mc,as1,zz,phic])
            for k in xrange(N1):                        # Newton Raphson iteration 
                indi = 2*xx[0]/fc_/dd**2                # indicator for concrete demand
                if indi<-0.99: NameError ("Reinforcement2D: slab concrete demand")
                if as1<0:
                    RR = array([cos(xx[3])*sin(xx[3])*xx[0]-mxy,
                                -mx+xx[0]*cos(xx[3])**2,
                                fy*xx[2]*xx[1]-my+xx[0]*sin(xx[3])**2,
                                xx[2]-0.5*dd*(1+sqrt(1+2*kk*xx[0]/fc_/dd**2))])
                    KM = array([[cos(xx[3])*sin(xx[3]),0,0,-xx[0]*sin(xx[3])**2+xx[0]*cos(xx[3])**2],
                                [cos(xx[3])**2,0,0,-2*cos(xx[3])*sin(xx[3])*xx[0]],
                                [sin(xx[3])**2,fy*xx[2], fy*xx[1],2*cos(xx[3])*sin(xx[3])*xx[0]],
                                [-1/(dd*sqrt(1+2*kk*xx[0]/fc_/dd**2)*fc_),0,1,0]])
                else:
                    RR = array([cos(xx[3])*sin(xx[3])*xx[0]-mxy,
                                fy*xx[2]*xx[1]-mx+xx[0]*cos(xx[3])**2,
                                -my+xx[0]*sin(xx[3])**2,
                                xx[2]-0.5*dd*(1+sqrt(1+2*kk*xx[0]/fc_/dd**2))])
                    KM = array([[cos(xx[3])*sin(xx[3]),0,0,-xx[0]*sin(xx[3])**2+xx[0]*cos(xx[3])**2],
                                [cos(xx[3])**2,fy*xx[2],fy*xx[1],-2*cos(xx[3])*sin(xx[3])*xx[0]],
                                [sin(xx[3])**2,0,fy*as2,2*cos(xx[3])*sin(xx[3])*xx[0]],
                                [-1/(dd*sqrt(1+2*kk*xx[0]/fc_/dd**2)*fc_),0,1,0]]) 
                if norm(RR)<Tol: break                # found solution
                xn = xx - solve(KM,RR) #,overwrite_a=True,overwrite_b=True)
                xx = copy(xn)
            if k<N1-1:
                mc = xx[0]                              # concrete moment
                if as1<0: 
                    as1 = 0
                    as2 = xx[1]                         # reinforcement
                else:
                    as1 = xx[1]
                    as2 = 0
                zz = xx[2]                              # internal lever arm
                phic = xx[3]                            # concrete moment direction
            else: raise NameError ("Reinforcement2D: no convergence") # no convergence reached
        x = 2*(dd-zz)/kk                                # compression zone height
        if (fcd*chi*kk*x+mc/zz)>100*Tol: raise NameError ("Reinforcement2D: moment control")
        return phic, zz, as1, as2
    def SLABSH(mx, my, mxy, qx, qy, as1B, as2B, as1T, as2T, fy, fcd, dd): # shear calc for slabs
        pp = PrinC( mx, my, mxy)                        # principal stresses
        for i in (0,3):
            mm, phi = pp[i], atan(pp[i+2]/pp[i+1])
            qphi = qx*cos(phi)+qy*sin(phi)
            if mm>0: rhosphi, Id = (as1B*cos(phi)**2+asB2*sin(phi)**2)/dd, 'B' # effective reinforcement ratio bottom 
            else:    rhosphi, Id = (as1T*cos(phi)**2+asT2*sin(phi)**2)/dd, 'T' # effective reinforcement ratio top
            vrdct = 0.1*1.89*(100*rhosphi*30)**0.3333*dd    # ad hoc
        return abs(qphi/vrdct), Id, rhosphi
    def RHOITER( stri, sig, phi, N1):
        xx = array([sig,0,phi])         # iteration initial value
        rho = 1
        for k in xrange(N1):            # Newton Raphson iteration for sigsy --> xx[1]
            if   stri=='rhoy': RR =array([cos(xx[2])*sin(xx[2])-sigxy/xx[0],     -sigx+xx[0]*cos(xx[2])**2,xx[1]-sigy+xx[0]*sin(xx[2])**2])
            elif stri=='rhox': RR =array([cos(xx[2])*sin(xx[2])-sigxy/xx[0],xx[1]-sigx+xx[0]*cos(xx[2])**2,     -sigy+xx[0]*sin(xx[2])**2])
            if norm(RR)<1.e-4: break    # found solution
            if   stri=='rhoy': KI = KIYC( sigxy, rho, xx)
            elif stri=='rhox': KI = KIXC( sigxy, rho, xx)
            xn = xx - dot(KI,RR)
            xx = copy(xn)
        if k<N1-1:
            sigc = xx[0]
            rho = xx[1]/fy
            phic = xx[2]
        else: raise NameError ("Reinforcement2D: no convergence") # no convergence reached
        return sigc, rho, phic, k
    fc = -10                                            # admissible concrete compressive stress
    fy = 500                                            # admissible reinforcement tensile stress
    N1 = 10                                             # max number of iterations for NR
    ASx = 0                                             # sum reinforcement x direction 
    ASy = 0                                             # sum reinforcement y direction 
    Vol = 0                                             # volume
    hh = 0.3
    scaleP = 1.5*VecScales[Key]
    P0 = plt.figure().add_subplot(111,title='plate reinforcement / slab lower bending reinforcement: ')
    P0.axis('equal')
    P0.grid()
    P1 = plt.figure().add_subplot(111,title='slab: upper bending reinforcement')
    P1.axis('equal')
    P1.grid()
    P2 = plt.figure().add_subplot(111,title='slab: shear reinforcement')
    P2.axis('equal')
    P2.grid()
    for i in xrange(len(ElList)):                       # loop over elements
        Elem = ElList[i]
        if (not Elem.Type=='CPE4') and (not Elem.Type=='CPE3') and  (not Elem.Type=='CPS4') and (not Elem.Type=='CPS3') and (not Elem.Type=='SB3'): continue
        if Elem.Type=='CPE4' or Elem.Type=='CPS4':
            xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[3]].XCo,NodeList[Elem.Inzi[0]].XCo]
            yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[3]].YCo,NodeList[Elem.Inzi[0]].YCo]
            P0.plot(xN,yN, 'b--')
            xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[3]].XCo]
            yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[3]].YCo]
        elif Elem.Type=='CPE3' or Elem.Type=='CPS3' or Elem.Type=='SB3':
            xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo,NodeList[Elem.Inzi[0]].XCo]
            yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo,NodeList[Elem.Inzi[0]].YCo]
            P0.plot(xN,yN, 'b--')
            if Elem.Type=='SB3':
                P1.plot(xN,yN, 'b--')
                P2.plot(xN,yN, 'b--')
            xN = [NodeList[Elem.Inzi[0]].XCo,NodeList[Elem.Inzi[1]].XCo,NodeList[Elem.Inzi[2]].XCo]
            yN = [NodeList[Elem.Inzi[0]].YCo,NodeList[Elem.Inzi[1]].YCo,NodeList[Elem.Inzi[2]].YCo]
        xC = dot( Elem.FormX(0,0,0), xN)
        yC = dot( Elem.FormX(0,0,0), yN)
   #     plt.text(xC,yC,str(i),ha='center',va='center',color='black',fontsize='small')
        for j in xrange(Elem.nIntLi):                    # build element stiffness with integration loop
            r = SamplePoints[Elem.IntT,Elem.nInt-1,j][0]
            s = SamplePoints[Elem.IntT,Elem.nInt-1,j][1]
            t = SamplePoints[Elem.IntT,Elem.nInt-1,j][2]

            Flag = True
            #if (i==61 and j==1) or (i==118 and j==0) or (i==91 and j==3) or (i==48 and j==2): Flag=True # E401-plate
            #else:Flag= False
            #if (i==0 and j==0) or (i==12 and j==0) or (i==59 and j==0) or (i==128 and j==0): Flag=True # E7-01
            #else: Flag=False

            f = Elem.Geom[1,0]*Elem.Geom[0,0]*SampleWeight[Elem.IntT,Elem.nInt-1,j]*Elem.JacoD(r,s, t) # weighting factor
            xI = dot( Elem.FormX(r,s,0), xN)              # global integration point coordinate
            yI = dot( Elem.FormX(r,s,0), yN)              # global integration point coordinate
            sigx =Elem.Data[j,3]                        # retrieve stresses / loading
            sigy =Elem.Data[j,4]
            sigxy=Elem.Data[j,5]
            pp = PrinC( sigx, sigy, sigxy)              # principal stresses
            if Flag:
                SS = pp[0]                              # principal stress plot
                if SS>=0: P0.plot([xI-SS*scaleP*pp[1],xI+SS*scaleP*pp[1]],[yI-SS*scaleP*pp[2],yI+SS*scaleP*pp[2]],'r-')
                else:     P0.plot([xI-SS*scaleP*pp[1],xI+SS*scaleP*pp[1]],[yI-SS*scaleP*pp[2],yI+SS*scaleP*pp[2]],'g-')
                SS = pp[3]
                if SS>=0: P0.plot([xI-SS*scaleP*pp[2],xI+SS*scaleP*pp[2]],[yI+SS*scaleP*pp[1],yI-SS*scaleP*pp[1]],'r-')
                else:     P0.plot([xI-SS*scaleP*pp[2],xI+SS*scaleP*pp[2]],[yI+SS*scaleP*pp[1],yI-SS*scaleP*pp[1]],'g-')
            if Elem.Type=='CPE4' or Elem.Type=='CPE3' or Elem.Type=='CPS4' or Elem.Type=='CPS3':
                rhox, rhoy = 0, 0
                if pp[0]<0 and pp[3]<0: sigc = min(pp[0],pp[3]) # compressive principal stresses only
                else:                                       # all other stress states
                    phi, sig = MSIG(pp)                     # direction and value of maximum principal stress
                    if sigxy>0: phic=-pi/4                  # prescription of compressive principal stress direction
                    else:       phic= pi/4
                    sigc = sigxy/(cos(phic)*sin(phic))      # corresponding concrete stress
                    rhox = (sigx - sigc*cos(phic)**2)/fy    # reinforcement ratio from equilibrium condition
                    rhoy = (sigy - sigc*sin(phic)**2)/fy
                    if rhoy>=0 and rhox<0:                  # undesired case, iterate for better solution with rhox=0
                        rhox = 0
                        sigc, rhoy, phic, k = RHOITER( 'rhoy', sig, phi, 10)
                        ff.write('%8.4f%8.4f rhoy iteration %3i\n'%(xI,yI,k))
                    elif rhoy<0 and rhox>=0:                # undesired case, iterate for better solution with rhoy=0
                        rhoy = 0
                        sigc, rhox, phic, k = RHOITER( 'rhox', sig, phi, 10)
                        ff.write('%8.4f%8.4f rhox iteration %3i\n'%(xI,yI,k))
                    elif rhoy<0 and rhox<0: raise NameError ("Reinforcement2D: an exceptional case")
    #                    print rhox,rhoy,phic, sigxy, sigc
                   
                    if Flag and rhox>0.0015: P0.text(xI,yI,format(100*rhox,".2f"),ha='center',va='bottom',color='blue',fontsize='x-small') # x-large, medium xx-large, small
                    if Flag and rhoy>0.0015: P0.text(xI,yI,format(100*rhoy,".2f"),ha='center',va='top',color='blue', rotation=90,fontsize='x-small')
                    rxy = sigc*cos(phic)*sin(phic) - sigxy   # equilibrium control
                    rxx = fy*rhox - sigx + sigc*cos(phic)**2 #
                    ryy = fy*rhoy - sigy + sigc*sin(phic)**2 #
                    if sqrt(rxy**2+rxx**2+ryy**2)>1.e-3: print(i,j,j1,'no equilibrium',rxx,ryy,rxy,norm(RR))
                ASx = ASx + rhox*f                           # total reinforcement volume in x direction
                ASy = ASy + rhoy*f                           # total reinforcement volume in y direction
                Vol = Vol + f
                ff.write('%8.4f%8.4f  %8.5f%8.5f'%(xI,yI,100*rhox,100*rhoy))
                if sigc<fc: 
                    ff.write(' concrete stress %8.4f'%(sigc))
                    print 'allowed concrete stress exceeded',xI,yI,sigc
                ff.write('\n')
            elif Elem.Type=='SB3':
                phicB,zzB,asB1,asB2 =SLABRE( sigx, sigy, sigxy, ReinfDes[0], ReinfDes[1], ReinfDes[3])   # reinforcement bottom side
                phicT,zzT,asT1,asT2 =SLABRE(-sigx,-sigy,-sigxy, ReinfDes[0], ReinfDes[1], ReinfDes[3]) # reinforcement top side
                qR, qI, rP = SLABSH(sigx,sigy,sigxy,Elem.Data[j,6],Elem.Data[j,7],asB1,asB2,asT1,asT2,ReinfDes[0],ReinfDes[1],ReinfDes[3]) # reinforcement top side
                if Flag:
                    as1B = "%.2f"%(10000*asB1)               # plot lower reinforcement
                    as2B = "%.2f"%(10000*asB2)
                    if asB1>0: P0.text(xI,yI,as1B,ha='center',va='bottom',color='black')#,fontsize='small')
                    if asB2>0: P0.text(xI,yI,as2B,ha='center',va='top',color='black', rotation=90)#,fontsize='small')
                    as1T = "%.2f"%(10000*asT1)               # plot upper reinforcement
                    as2T = "%.2f"%(10000*asT2)
                    if asT1>0: P1.text(xI,yI,as1T,ha='center',va='bottom',color='black')#,fontsize='small')
                    if asT2>0: P1.text(xI,yI,as2T,ha='center',va='top',color='black', rotation=90)#,fontsize='small')
#                    if qR>1 and qI=='B': P2.text(xI,yI,"%.1f/%.1f"%(qR,10000*rP)) # plot shear, compute cm2 from m2 ?
#                    if qR>1 and qI=='T': P2.text(xI,yI,"%.1f/%.1f"%(qR,10000*rP),color='red') # compute cm2 from m2 ?
                    if qR>1 and qI=='B': P2.text(xI,yI,"%.1f"%(qR)) # plot shear, compute cm2 from m2 ?
                    if qR>1 and qI=='T': P2.text(xI,yI,"%.1f"%(qR)) # compute cm2 from m2 ?
                ff.write('%8.4f%8.4f  %8.5f%8.5f%8.5f  %8.5f%8.5f%8.5f\n'%(xI,yI,zzB/ReinfDes[3],10000*asB1,10000*asB2,zzT/ReinfDes[3],10000*asT1,10000*asT2))
                ASx = ASx + (asB1+asT1)*f                           # total reinforcement volume in x direction
                ASy = ASy + (asB2+asT2)*f                           # total reinforcement volume in y direction
                Vol = Vol + f*hh
            else: print 'Element type not supported'
    ff.write('\nasx volume     %8.4f\nasy volume     %8.4f\nconcrete volume%8.2f\n'%(ASx,ASy,Vol))
    return ASx, ASy, Vol

class ConPlaD:
    def __init__(self):
        pass
    def Run(self, Name, Key, LinAlgFlag, PloF):
        f1=open( Name+".in.txt", 'r')
        f2=open( Name+".elemout.txt", 'w')
        f3=open( Name+".nodeout.txt", 'w')
        f7=None
        print "ConPlaD: ", Name
        NodeList, ElList, MatList, StepList = ReadInputFile(f1, False) # read input file and create node, element, material and step lists -> in SimFemInOut.py
        N, Mask, Skyline, SDiag, SLen = AssignGlobalDof( NodeList, ElList, MatList)              # assign degrees of freedom (dof) to nodes and elements -> see above
        f1.close()
        if pth.isfile(Name+".opt.txt"):                     # read options, mandatory for reinforcement design
            f4=open( Name+".opt.txt", 'r')
            WrNodes, LineS, ReDes, MaxType = ReadOptionsFile(f4, NodeList)
            f4.close()
        else: raise NameError ("ConPlaD: options file missing")
        # Initializations
        # with N = Index = total number of Dofs
        VecU = zeros((N),dtype=double)                      # current displacement vector
        VecI = zeros((N),dtype=double)                      # internal nodal forces vector
        VecA = zeros((N),dtype=double)                      # nodal forces vector from prescribed constrained dofs
        VecP = zeros((N),dtype=double)                      # load vector
        VecP0= zeros((N),dtype=double)                      # dummy load vector
        VecZ = zeros((N),dtype=double)                      # zero vector
        BCIn = ones((N),dtype=int)                          # for compatibility with SimFem
        BCIi = zeros((N),dtype=int)                         # for compatibility with SimFem
        # FE Calculation
        stime = time()
        Time = StepList[0].TimeTarg                         # set time
        if LinAlgFlag:
            KVecU = zeros(SLen, dtype=float)                    # Initialize upper right part of stiffness vector
            KVecL = zeros(SLen, dtype=float)                    # Initialize lower left part of stiffness vector
            IntForces( N, MatList, ElList, Time, VecZ, VecU, VecZ, VecZ, VecI, None,None,KVecU,KVecL,Skyline,SDiag, 2, None, False, False, False)# internal nodal forces / stiffness matrix
        else:
            MatK = sparse.lil_matrix((N, N))                    # sparse stiffness matrix initialization
            IntForces( N, MatList, ElList, Time, VecZ, VecU, VecZ, VecZ, VecI, MatK,None,None,None,None,None, 2, None, False, False, False)# internal nodal forces / stiffness matrix
        StepList[0].NodalLoads( N, Time, Time, NodeList, VecP, VecP0)# introduce concentrated loads into system -> in SimFemSteps.py
        StepList[0].ElementLoads( Time, Time, ElList, NodeList, VecP, VecP0)# introduce distributed loads into system -> in SimFemSteps.py
        if LinAlgFlag:
            StepList[0].BoundCond( N, Time, 0, Time, NodeList, VecU, VecI, VecP, VecP0, BCIn, BCIi, None,KVecU,KVecL,Skyline,SDiag, 2, False)# introduce boundary conditions
        else:
            StepList[0].BoundCond( N, Time, 0, Time, NodeList, VecU, VecI, VecP, VecP0, BCIn, BCIi, MatK,[],None,None,None, 2, False)# introduce boundary conditions
        VecR = VecP + VecA - VecI                           # residual vector
        print StepList[0].TimeTarg, norm(VecR)
        if LinAlgFlag:
            LinAlg2.sim0_lu(      KVecU, KVecL, SDiag, Skyline, N)
            LinAlg2.sim0_so(VecR, KVecU, KVecL, SDiag, Skyline, N)
            VecU = copy(VecR)
        else:
            K_LU = linsolve.splu(MatK.tocsc(),permc_spec=3)     #triangulization of stiffness matrix
            VecU = K_LU.solve( VecR )                           # solution of K*u=R -> displacement increment
        IntForces( N, MatList, ElList, Time, VecZ, VecU, VecZ, VecZ, VecI, None,None,None,None,None,None, 1, None, False, False, False)# internal nodal forces
        WriteElemData( f2, f7, Time, ElList, NodeList, MatList, [])          # write element data
        if LinAlgFlag: NodeList.sort(key=lambda t: t.Label)
        WriteNodalData( f3, Time, NodeList, VecU, VecU)     # write nodal data
        if LinAlgFlag: NodeList.sort(key=lambda t: t.CMLabel_)
        f3.close()
        # post processing
        f2.close()
        f2=open( Name+".elemout.txt", 'r')                  #
        Sc, Sc1 = PostScales( ElList, NodeList, f2 )
        # Limit analysis
        f4=open( Name+".reinforcement.txt", 'w')
        ASx, ASy, Vol = Reinforcement2D( ElList, NodeList, ReDes, Sc, Key, f4)
        f4.close()
        print ASx,ASy,Vol,ASx*7810/Vol,ASy*7810/Vol
        print time()-stime
        if PloF:
            plt.show()
            return 0
        else:
            import hashlib
            mmm = hashlib.md5()
            fp= file( Name+".reinforcement.txt", "rb")
            while True:
                data= fp.read(65536)
                if not data: break
                mmm.update(data)
            fp.close()
            RC = mmm.hexdigest()
            return RC

if __name__ == "__main__":
#    numpy.seterr(all='raise')
    #Name="C:/Users/regga/Desktop/testcode/E4-01plate"                         # input data name
    #Key='EL11.0000'
    #Name="C:/Users/regga/Desktop/testcode/E7-01"                         # input data name
    #Key='PROP11.0000'
    Name="C:/Users/regga/Desktop/testcode/E6-02"                 # ConPlad Plate E4_01 linear elastic, reinforcement design        11s
    Key = 'EL11.0000'                               # Key concatenates label of element set and time to be evaluated
    ConPlaD_ = ConPlaD()
    LinAlgFlag = False                                   # True: store matrices as vector and use own c routines for LU-decomposition and backward substitution
    ConPlaD_.Run(Name, Key, LinAlgFlag, True)
