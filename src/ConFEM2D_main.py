"""The main module of ConFEM2D """
from time import time
from numpy import ones, copy
from scipy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse.linalg import aslinearoperator
sparse.linalg.use_solver(useUmfpack=True)
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

try:
    import LinAlg2
    reload(LinAlg2)
    from LinAlg2 import *
    Linalg_possible = True
except ImportError:
    Linalg_possible = False

import logging
logging.basicConfig(filename='Log_file.txt', level=logging.DEBUG) #format=' %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)  # enable/disable logging

class ConFem:
    def __init__(self):
        pass  # flag for symmetric system

    #    @profile
    def Run(self, Name, LogName, PloF, LinAlgFlag, Restart, ResType):
        print Linalg_possible, ConFemElemCFlag, ConFemMatCFlag
        print "ConFem: ", Name
        logger.critical('Start of program')
        if Restart:
            fd = open(Name + '.pkl', 'r')  #
            uuu = cPickle.Unpickler(fd)
            NodeList, ElList, MatList, StepList, N, WrNodes, LineS, Flag, \
            VecU, VecC, VecI, VecP, VecP0, VecP0old, VecBold, VecT, VecS, VeaU, VevU, VeaC, VevC, VecY, BCIn, BCIi, Time, TimeOld, TimeEl, TimeNo, TimeS, i, Mask, Skyline, SDiag, SLen, SymSys \
                = uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load(), uuu.load()
            fd.close()
            f1 = open(Name + ".in.txt", 'r')
            MatList, StepList = ReadInputFile(f1, Restart)  # read input file
            f1.close()
            f2 = open(Name + ".elemout.txt", 'a')  #
            f3 = open(Name + ".nodeout.txt", 'a')  #
            f6 = open(Name + ".restart.txt", 'w')  #
            f5 = open(Name + ".timeout.txt", 'a')  #
            f7, MaxType = None, None
        else:
            f1 = open(Name + ".in.txt", 'r')
            NodeList, ElList, MatList, StepList = ReadInputFile(f1, Restart)  # read input file # print NodeList.__dict__
            f1.close()
            #if LinAlgFlag:
            CuthillMckee(NodeList, ElList)  # to reduce the skyline
            NodeList.sort(key=lambda t: t.CMLabel_)
            N, Mask, Skyline, SDiag, SLen = AssignGlobalDof(NodeList, ElList, MatList)  # assign degrees of freedom (dof) to nodes and elements -> see above
            f2 = open(Name + ".elemout.txt", 'w')
            f3 = open(Name + ".nodeout.txt", 'w')
            f6 = open(Name + ".protocol.txt", 'w')
            print >> f6, "ConFem: ", Name
            WrNodes, LineS = None, None
            MaxType = []
            f5, f7 = None, None
            if pth.isfile(Name + ".opt.txt"):  # read options file if there is any
                f4 = open(Name + ".opt.txt", 'r')
                WrNodes, LineS, ReDes, MaxType = ReadOptionsFile(f4, NodeList)
                f4.close()
                f5 = open(Name + ".timeout.txt", 'w')
                if len(MaxType) > 0: f7 = open(Name + ".elemmax.txt", 'w')
            for i in list(MatList.values()):  # check, whether particular material types are in system
                #                Flag = (isinstance(i,ConFemMat.ElasticLT) or isinstance(i,ConFemMat.WraElasticLTReShell) or isinstance( i.Conc, ConFemMat.ElasticLT)) and i.Used
                Flag = (isinstance(i, lib_Mat.ElasticLT) or isinstance(i.Conc, lib_Mat.ElasticLT)) and i.Used
                if Flag: break

            # Initializations
            VecU = zeros((N), dtype=double)  # current displacement vector
            VevU = zeros((N), dtype=double)  # current velocities
            VeaU = zeros((N), dtype=double)  # current accelerations
            VecC = zeros((N), dtype=double)  # displacement vector of previous time step
            VevC = zeros((N), dtype=double)  # previous time step velocities
            VeaC = zeros((N), dtype=double)  # previous time step accelerations
            VecI = zeros((N), dtype=double)  # internal nodal forces vector
            VecP = zeros((N), dtype=double)  # load vector of current Time
            VecP0 = zeros((N), dtype=double)  # nominal load vector of TimeTarg
            VecP0old = zeros((N), dtype=double)  # nominal load vector of previous calculation step
            VecBold = zeros((N), dtype=double)  # holds reaction forces of previous step
            VecT = zeros((N), dtype=double)  # current temperatures vector
            VecS = zeros((N), dtype=double)  # temperatures vector of previous time step
            VecY = zeros((N), dtype=double)  # previous time step displacement increment (for line search only) or for dynamics
            BCIn = ones((N), dtype=int)  # indices for dofs with prescribed displacements --> 0, --> 1 otherwise
            BCIi = zeros((N), dtype=int)  # indices for dofs with prescribed displacements --> 1, --> 0 otherwise
            Time = 0.
            TimeS = 0.  # Target time of previous step
            TimeOld = 0.
            SymSys = IsSymSys(MatList)  # if there is at least one un-symmetric material the whole system is un-symmetric
            i = 0  # step counter
        print 'symmetric system', SymSys
        print >> f6, 'symmetric system', SymSys
        # Calculations
        stime = time()
        while i < len(StepList):
            StLi = StepList[i]
            logger.critical('Step %s starts with SolType = %s' % (i,StLi.SolType))
            print i, 'step starts', StLi.SolType
            print >> f6, i, 'step starts', StLi.SolType
            if StLi.varTimeSteps:
                TimeVarL = len(StLi.TimeTargVar)
                TimeTarg = StLi.TimeTargVar[-1]
                if i > 0:
                    TimeS = StepList[i - 1].TimeTargVar[-1]  # time target of previous step
                else:
                    TimeS = 0.
            else:
                TimeTarg = StLi.TimeTarg  # time target for step
                if i > 0:
                    TimeS = StepList[i - 1].TimeTarg  # time target of previous step
                else:
                    TimeS = 0.
            TS = TimeTarg - TimeS
            if not StLi.Buckl:
                if TS < ZeroD:
                    raise NameError("ConFem: TimeS <= TimeTarg")
                else:
                    DTS = 1. / TS
            logger.critical('The computation below is from previous time step TimeS=%s to TimeTarg=%s' %(TimeS,TimeTarg))
            logger.critical('At this step(i=%s), constant DTS=1/(TimeTar-TimeS)= %s' %(i,DTS))
            TimeEl = TimeS  # !!!!!
            TimeNo = TimeS  # !!!!
            StLi.current = i

            MatM = None
            Stop, Stop_ = False, False  # flag to stop computation
            if LineS <> None and LineS[0] > 1:  # parameters for line search iteration
                LinS = LineS[0]
                LinT = LineS[1]
            else:
                LinS = 1
            counter = 0
            StLi.BoundOffset(NodeList, VecU)  # add offset for prescribed displacement from current displacement for OPT=ADD
            logger.critical('reset counter=%s' %(counter))
            logger.critical('enter in loop over time steps - TimeS=%s -->TimeTarg=%s' %(TimeS,TimeTarg))
            while not Stop:  # loop over time steps
                counter += 1
                logger.info('')
                logger.info('Increase counter+=1, so now counter=%s, Time=%s' % (counter, Time))
                #                if counter == 4: Stop = True   # 266
                if len(StLi.ElFilList) > 0 and Time + 1.e-6 >= TimeEl: TimeEl = Time + StLi.ElFilList[0].OutTime  # set time for element output
                if len(StLi.NoFilList) > 0 and Time + 1.e-6 >= TimeNo: TimeNo = Time + StLi.NoFilList[0].OutTime  # set time for element output
                A_BFGS = []  # BFGS auxiliary list of vectors
                B_BFGS = []  # BFGS auxiliary list of vectors
                rho_BFGS = []  # BFGS auxiliary list of scalars
                VecD = zeros((N), dtype=double)  # displacement incr vector
                VecR = zeros((N), dtype=double)  # residual nodal forces vector
                VecRP = zeros((N), dtype=double)  # residual nodal forces vector
                dt = 0  # initial value for time step in case of quasistatic computation
                tt = 0  # time increment in each equilibrium iteration
                LoFl = False  # flag to proceed with time / loading
                CalcType = 2  # 0: check system 1: internal forces only 2: internal forces and tangential stiffness matrix
                En0, En1 = 1., 1.  # initialization initial energy
                logger.info('reinitialize VecD,VecR,VecRP, and BFGS parameters')
                logger.info('reinitialize dt=0-initial value for time steps, and tt=0-time incre in each iteration')

                logger.info('enter in equilibrium Iteration loop of this counter=%s' %(counter))
                for j in xrange(StLi.IterNum):  # equilibrium iteration loop
                    logger.info('%s-th iteration of counter %s' %(j,counter))
                    S = [0.]  # line search auxiliary value
                    G = [dot(VecRP, VecD)]  # line search auxiliary value
                    sds = 1.0  # line search scalar
                    VecUP = copy(VecU)  # remember displacement result of last load increment
                    k_ = -1  # auxiliary counter for line search
                    logger.debug('before each equilibrium iteration, reinitialize Line search parameters -S,G,sds,k_')
                    logger.debug('in each equilibrium iteration, a small line search procedure is required \n'
                                  '        for better oriented dipls direction. Enter in Line search iteration')
                    for k in xrange(LinS):  # line search iteration
                        logger.debug('in xrange(LinS = %s), iteration k = %s' %(LinS,k))
                        logger.debug('update displacement VecU = VecUP + sds*VecD, displ incre dU = norm(VecU-VecC)')
                        VecU = VecUP + sds * VecD  # update displacements
                        dU = norm(VecU - VecC)  # displacement increment compared to last step
                        # make system vectors and matrices
                        logger.debug('Make system vectors and matrices')
                        StLi.NodalTemp(N, Time, NodeList, VecT)  # introduce nodal temperatures into system with computation of nodal temperatures -> in SimFemSteps.py
                        if Flag and j == 0:
                            IntForces(N, MatList, ElList, Time - TimeOld, VecC, VecU, VecS, VecT, VecI,
                                                      None, None, None, None, None, None, 0, f6, StLi.NLGeom, SymSys,
                                                      False)  # check system state for certain materials
                        if LinAlgFlag:
                            if CalcType == 2:
                                KVecU = zeros(SLen, dtype=float)  # Initialize upper right part of stiffness vector
                                if not SymSys:
                                    KVecL = zeros(SLen, dtype=float)  # Initialize lower left part of stiffness vector
                                else:
                                    KVecL = None
                            IntForces(N, MatList, ElList, Time - TimeOld, VecC, VecU, VecS, VecT, VecI, None, None,
                                      KVecU, KVecL, Skyline, SDiag, CalcType, f6, StLi.NLGeom, SymSys,
                                      False)  # internal nodal forces / stiffness matrix
                        else:
                            if CalcType == 2:
                                MatK = sparse.lil_matrix((N, N))  # sparse stiffness matrix initialization
                            IntForces(N, MatList, ElList, Time - TimeOld, VecC, VecU, VecS, VecT, VecI, MatK, MatM,
                                      None, None, None, None, CalcType, f6, StLi.NLGeom, SymSys,
                                      StLi.Buckl)  # internal nodal forces / stiffness matrix

                        if CalcType == 2 : logger.debug('compute IntForces(VecC,VecU) and tangential Kt-these make update in VecI,VecP,VecP0,VecR')
                        else :             logger.debug('compute only IntForcecs(VecC,VecU), no more tangential Kt matricx')

                        logger.debug('Introduce c/d Loads and Boundary conditions into system')
                        StLi.NodalLoads(N, Time, TimeTarg, NodeList, VecP, VecP0)  # introduce concentrated loads into system -> in SimFemSteps.py
                        StLi.ElementLoads(Time, TimeTarg, ElList, NodeList, VecP, VecP0)  # introduce distributed loads into system -> in SimFemSteps.py
                        StLi.NodalPrestress(N, Time, ElList, NodeList, VecP, VecU, StLi.NLGeom)  # introduce prestressing
                        StLi.NodalPrestress(N, TimeTarg, ElList, NodeList, VecP0, VecU, StLi.NLGeom)  # nominal prestressing (maybe this works not optimal with P0-approach)
                        VecP0__ = copy(VecP0)
                        VecP0 = VecP0 - VecP0old  # only nominal load change compared to last step is relevant
                        VecB = copy(VecI - VecP)  # keep boundary forces from internal forces

                        if LinAlgFlag:
                            StLi.BoundCond(N, Time, TimeS, TimeTarg, NodeList, VecU, VecI, VecP, VecP0, BCIn, BCIi,
                                           None, KVecU, KVecL, Skyline, SDiag, CalcType, SymSys)  # introduce boundary conditions
                        else:
                            StLi.BoundCond(N, Time, TimeS, TimeTarg, NodeList, VecU, VecI, VecP, VecP0, BCIn, BCIi,
                                           MatK, [], None, None, None, CalcType, SymSys)  # introduce boundary conditions
                        if CalcType == 2:
                            VecP0_ = copy(VecP0)  # remember nominal load in case stiffness matrix is not updated upcoming
                        else:
                            VecP0 = copy(VecP0_)  # use previous value of nominal load in case stiffness matrix was not updated
                        VecR = VecP - VecI  # residual vector
                        logger.debug('These introductions make update in KvecU,VecI,VecP,VecP0 \n'
                                      '        then compute the new residual vector VecR = VecP-VecI')
                        # line search scaling, if prescribed
                        logger.debug('Base on these new (VecR,VecP,VecP0)&(VecRP,VecU,VecD)=cte in line search procedure \n'
                                      '        Compute line search parameters S,G,sds of this k=%sth line search iteration' %(k))
                        if j > 1 and LinS > 1:
                            G_ = dot(VecR, VecD)
                            G0 = dot(VecRP, VecD)
                            if G_ * G[k_] < 0.:
                                k_ = k_ + 1
                                S += [sds]
                                G += [G_]
                            if abs(G_) > LinT * abs(G0):
                                sds = S[k_] - G[k_] * (sds - S[k_]) / (G_ - G[k_])
                                if sds < 0.0:
                                    sds = 1.0
                                elif sds < 0.2:
                                    sds = 0.2
                                elif sds > 5.0:
                                    sds = 5.0
                                print 'line search', k_, 'G/G0', G_ / G0, 'step', sds
                                print >> f6, 'line search', k_, 'G/G0', G_ / G0, 'step', sds
                                logger.debug('line seach : sds=%s' %(sds))
                            else:
                                logger.debug('end of line search iteration, sds=%s' %(sds))
                                break  # end of line search iteration
                        else:
                            logger.debug('no line search iteration, sds=%s' %(sds))
                            break  # no line search iteration

                    logger.debug('end of Line search iteration')
                    # residuum
                    logger.debug('Calculate residuum and equilibrium control')
                    logger.debug('VecRD = VecRP-VecR+tt*DTS*VecP0 with tt=%s \n'
                                  '        then VecRP=copy(VecR)' %(tt))
                    Resi = norm(VecR)  # residual norm
                    VecRD = VecRP - VecR + tt * DTS * VecP0  # BFGS auxiliary vector
                    VecRP = copy(VecR)  # residual nodal forces vector of previous iteration

                    # equilibrium control
                    if j > 0:
                        VecBold = BCIi * (VecB - VecBold)
                        En1 = dot(VecD, (VecR + VecBold))
                        VecBold = copy(VecB)
                        if j == 1: En0 = En1  # for convergence control via energy criterion
                    if En0 > ZeroD:
                        Resi_ = abs(En1 / En0)  # energy based convergence indicator
                    else:
                        Resi_ = 1.
                    print counter, i, TimeTarg, Time, j, Resi, dU, Resi_  # En1, En0 # report state to screen
                    print >> f6, counter, i, TimeTarg, Time, j, Resi, dU, Resi_  # report state to log file
                    logger.debug('Report : TimeTarg %s, counter %s, Time %s, eq iter j= %s, Resi %s, dU %s, Resi_ %s' %(TimeTarg,counter,Time,j,Resi,dU,Resi_))

                    if j > 0 or StLi.Dyn:
                        if Resi < StLi.IterTol:
                            logger.debug('j=%s>0. Convergence criterium reached - Resi<StLi.IterTol. Break, not need anymore compute new solution, new time increment, new displ' % j)
                            break  # convergence criterium reached, continue after for-loop
                        else: logger.debug('at j=%s, condition Resi<StLi.IterTol not meet' % j)
                    else:  # initial state of time step / loading increment
                        logger.debug('j=%s. Need to consider Stop_ even if Resi<StLi.IterTol' % j)
                        if Resi < StLi.IterTol:  # equilibrium found for initial state
                            if Stop_:
                                Stop = True
                                logger.debug('Break ! finish step in case of Flag and in case j=0 and Resi<StLi.IterTol. Stop_,Stop=%s, %s. Jump out of Equilibrium iteration loop' %(Stop_,Stop))
                                break  # finish step in case of Flag
                            LoFl = True  # initial system is in static equilibrium, proceed with time step / loading
                            logger.debug('at j=%s, Resi<StLi.IterTol satisfied, but Break in oder to jump out of Equilib iteration loop still not meet' % (j))
                        else: logger.debug('at j=%s. Condition Resi<StLi.IterTol not meet. LoFl=%s' % (j, LoFl))

                    # determine new solution
                    logger.debug('equilibrium isnt satisfied, determine new solution using %s method : compute new VecD, VecDI' %(StLi.SolType))
                    if j == 0 or StLi.SolType == 'NR':  # Newton Raphson
                        if LinAlgFlag:
                            if SymSys:
                                LinAlg2.sim1_lu(KVecU, SDiag, Skyline, N)
                            else:
                                LinAlg2.sim0_lu(KVecU, KVecL, SDiag, Skyline, N)
                            VecD = copy(VecR)
                            if SymSys:
                                LinAlg2.sim1_so(VecD, KVecU, SDiag, Skyline, N)
                            else:
                                LinAlg2.sim0_so(VecD, KVecU, KVecL, SDiag, Skyline, N)
                            VecDI = copy(VecP0)
                            if SymSys:
                                LinAlg2.sim1_so(VecDI, KVecU, SDiag, Skyline, N)
                            else:
                                LinAlg2.sim0_so(VecDI, KVecU, KVecL, SDiag, Skyline, N)
                        else:
                            K_LU = linsolve.splu(MatK.tocsc(), permc_spec=0)  # triangulization of stiffness matrix
                            VecD = K_LU.solve(VecR)  # solution of K*u=R -> displacement increment
                            VecDI = K_LU.solve(VecP0)  # solution contribution for arc length control
                        #                        for jj in xrange(len(VecRD)):sys.stdout.write('%5i%5i%16.8f%16.8f%16.8f%16.8f%16.8f%16.8f\n'%(j,jj,VecD_[jj],VecD[jj],VecD_[jj]-VecD[jj],VecDI_[jj],VecDI[jj],VecDI_[jj]-VecDI[jj]))
                        if StLi.SolType <> 'NR': CalcType = 1  # stiffness matrix not built anymore
                    elif StLi.SolType == 'BFGS':  # BFGS according to Matthies & Strang 1979 (S. 1617) for indefinite matrices
                        VecRD = VecRD * BCIn  # mask prescribed dofs
                        VecD = VecD * BCIn  # mask prescribed dofs
                        A_BFGS += [VecRD]       # append to list
                        B_BFGS += [sds * VecD]  # append to list
                        rho_BFGS += [1. / dot(transpose(VecRD), sds * VecD)]    # scalar
                        alpha_BFGS = []
                        for k in range(j - 1, -1, -1):
                            alpha = rho_BFGS[k] * dot(B_BFGS[k], VecR)      # scalar
                            VecR = VecR - alpha * A_BFGS[k]
                            alpha_BFGS += [alpha]               # append to list
                        alpha_BFGS.reverse()
                        if LinAlgFlag:
                            for jj in xrange(N): VecD[jj] = VecR[jj]
                            if SymSys:
                                LinAlg2.sim1_so(VecD, KVecU, SDiag, Skyline, N)
                            else:
                                LinAlg2.sim0_so(VecD, KVecU, KVecL, SDiag, Skyline, N)
                        else:
                            VecD = K_LU.solve(VecR)
                        for k in range(j):
                            beta_BFGS = rho_BFGS[k] * dot(A_BFGS[k], VecD)          # scalar
                            VecD = VecD + (alpha_BFGS[k] - beta_BFGS) * B_BFGS[k]  # end BFGS
                    else:
                        if LinAlgFlag:
                            if SymSys:
                                LinAlg2.sim1_so(VecR, KVecU, SDiag, Skyline, N)
                            else:
                                LinAlg2.sim0_so(VecR, KVecU, KVecL, SDiag, Skyline, N)
                            VecD = copy(VecR)
                            if SymSys:
                                LinAlg2.sim1_so(VecP0, KVecU, SDiag, Skyline, N)
                            else:
                                LinAlg2.sim0_so(VecP0, KVecU, KVecL, SDiag, Skyline, N)
                            VecDI = copy(VecP0)
                        else:
                            VecD = K_LU.solve( VecR)  # solution of K*u=R -> displacement increment, modified Newton-Raphson
                            VecDI = K_LU.solve(VecP0)  # solution contribution for arc length control

                    # determine time increment / step and displacement increment
                    logger.debug('determine time incr/step and displacement incr')
                    logger.debug('set tt=0-initialize time incr for this iteration step. The TimeOld=%s, dt=%s' % (TimeOld,dt))
                    tt = 0  # initialize time increment for this iteration step
                    if not StLi.Dyn and LoFl:
                        if StLi.ArcLen:
                            tt = TS * ArcLength(StLi.ArcLenV, VecDI, VecD, VecU - VecC, VecY,
                                                Mask)  # time increment with arc length control
                        elif j == 0:
                            if StLi.varTimeSteps:
                                for targ_i in xrange(TimeVarL):  #
                                    if Time < StLi.TimeTargVar[targ_i]:
                                        tt = StLi.TimeStepVar[targ_i]
                                        break
                            else:
                                tt = StLi.TimeStep  # time increment prescribed
                        dt = dt + tt  # update time increment
                        Time = TimeOld + dt  # update time (dt = 0 in case of arc length control)
                        logger.debug('LoFl=%s, compute new tt=%s, dt=dt+tt=%s' %(LoFl,tt,dt))
                    logger.debug('tt is only incremented one time at j=0 and LoFl=true. After computing: LoFl=%s, tt = %s, dt=dt+tt=%s, Time=TimeOld+dt=%s' %(LoFl,tt,dt,Time))
                    VecD = VecD + tt * DTS * VecDI  # new displacement increment with two contributions
                    logger.debug('compute new VecD=VecD+tt*DTS*VecDI')

                # if j>=0: LogResiduals(LogName, counter, j, NodeList, VecRP, VecU)  #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                logger.info('End of Equilibrium iteration loop in Time=%s' %(Time))
                TimeOld = Time
                VecY = VecU - VecC  # store current displacement time step increment
                VecC = copy(VecU)  # store current displacements as previous for next time step
                VecS = copy(VecT)  # store current temperatures as previous for next time step
                VevC = copy(VevU)  # store current velocities as previous for next time step
                VeaC = copy(VeaU)  # store current accelerations as previous for next time step
                logger.info('store current displ as previous for next step : TimeOld=Time=%s,VecY,VecC' % TimeOld)
                if j == (StLi.IterNum - 1):
                    LogResiduals(LogName, counter, j, NodeList, VecRP, VecU)
                    logger.info('j=%s reaches StLi.IterNum')
                logger.info('FinishEquilibIteration. Check if there is changed state (by func.UpdateStateVar) in certain material')
                FinishEquilibIteration(MatList, ElList, f6, StLi.NLGeom, LoFl)  # update state variables etc.

                if Time > TimeTarg - 1.e-6:
                    if not Flag or StLi.Dyn:
                        Stop = True  # time target reached, finalize computation, Flag for material ElasticLT, not Flag for all other materials
                        logger.info('Stop=True-time target reached')
                    else:
                        Stop_ = True  # eventually with one more equilibrium iteration for ElasticLT in case of quasistatic computation, Stop is set elsewhere
                        logger.info('Stop_=True-eventually with one more equilibrium iteration for ElasticLT in case of quasistatic computation, Stop is set elsewhere')

                if ((Time + 1.e-6 >= TimeEl or Stop) and not Stop_) or (Stop and Stop_):
                    #                if counter in (582,553,526,1094,1111,1128): #   # data set E1-01 only
                    #                if counter in [iii for iii in range(1000) if iii%50==0 ]:
                    logger.debug('Vi Time=%s > TimeEL=%s, write data to file elemout' %(Time,TimeEl))
                    WriteElemData(f2, f7, Time, ElList, NodeList, MatList, MaxType)  # write element data
                    fd = open(Name + '.pkl', 'w')  # Serialize data and store for restart
                    ppp = cPickle.Pickler(fd)
                    ppp.dump(NodeList);
                    ppp.dump(ElList);
                    ppp.dump(MatList);
                    ppp.dump(StepList);
                    ppp.dump(N);
                    ppp.dump(WrNodes);
                    ppp.dump(LineS);
                    ppp.dump(Flag); \
                            ppp.dump(VecU);
                    ppp.dump(VecC);
                    ppp.dump(VecI);
                    ppp.dump(VecP);
                    ppp.dump(VecP0);
                    ppp.dump(VecP0old);
                    ppp.dump(VecBold);
                    ppp.dump(VecT);
                    ppp.dump(VecS);
                    ppp.dump(VeaU);
                    ppp.dump(VevU);
                    ppp.dump(VeaC);
                    ppp.dump(VevC);
                    ppp.dump(VecY); \
                            ppp.dump(BCIn);
                    ppp.dump(BCIi);
                    ppp.dump(Time);
                    ppp.dump(TimeOld);
                    ppp.dump(TimeEl);
                    ppp.dump(TimeNo);
                    ppp.dump(TimeS);
                    ppp.dump(i);
                    ppp.dump(Mask);
                    ppp.dump(Skyline);
                    ppp.dump(SDiag);
                    ppp.dump(SLen);
                    ppp.dump(SymSys)
                    fd.close()
                    f2.flush()
                    print >> f6, 'Element Data written', Time, TimeEl
                if ((Time + 1.e-6 >= TimeNo or Stop) and not Stop_) or (Stop and Stop_):
                    if LinAlgFlag: NodeList.sort(key=lambda t: t.Label)
                    WriteNodalData(f3, Time, NodeList, VecU, VecB)  # write nodal data
                    if LinAlgFlag: NodeList.sort(key=lambda t: t.CMLabel_)
                    f3.flush()
                    print >> f6, 'Nodal Data written', Time, TimeNo
                if f5 <> None:
                    WriteNodes(f5, WrNodes, Time, VecU, VecB, VecP)
                    f5.flush()
            VecP0old = copy(VecP0__)  # remember nominal load of this step
            i += 1  # next step list
            logger.critical('i+=1=%s - next step list' %(i))

        print time() - stime  # end of computation loop, look for computation time
        print >> f6, time() - stime
        f2.close()
        f3.close()
        if f5 <> None: f5.close()
        f6.close()
        if f7 <> None: f7.close()
        #        fX.close()
        print 'Characteristic size numbers: ', N, len(ElList), len(NodeList), '__', 1. * SLen / (
                    N ** 2), 1. * SLen / 1024
        #

        RC = FinishAllStuff(PloF, Name, ElList, NodeList, MatList, f2, VecU, WrNodes, "elemout")
        logger.critical('End of program')
        return RC


if __name__ == "__main__":
    LogName = "C:/Users/regga/Desktop/ConFEM2D/examples/ex1/LogFiles"  # to log temporary data
    Name = "C:/Users/regga/Desktop/ConFEM2D/examples/ex1/ex1"
    ConFem_ = ConFem()
    if Linalg_possible:
        LinAlgFlag = True
    else:
        LinAlgFlag = False
    #LinAlgFlag = False  # intentionally
    Restart = False
    Plot = True
    RC = ConFem_.Run(Name, LogName, Plot, LinAlgFlag, Restart, "elemout")  # flags for plot, system of equation solution, restart
    print RC
