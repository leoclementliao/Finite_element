#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 15:48:48 2018

@author: liaomeng
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from time import time
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
sin,cos,sqrt,pi = np.sin,np.cos,np.sqrt,np.pi

"""  Constructure Information  """
L = 1
nx = 32
nbi = 1
r = 0.3

""" Mechanics parameter """
E = np.array([1, 10]) # [Phase1(Matrix),Phase2(Inclusion)]
nu = np.array([0.3, 0.3])
mu = 0.5*E/(1+nu)
lambd = E*nu/((1+nu)*(1-2*nu)) # Etat plan deformation

start = time()
"""  Mesh Construction  """
def fun_Mesh_fix(L,nx):
    h = L
    ny = nx
    [Mx,My] = np.meshgrid(np.linspace(0,L,nx),np.linspace(0,h,ny))
    Nx = np.reshape(Mx.T,(nx*ny))-L/2
    Ny = np.reshape(My.T,(nx*ny))-h/2
    M_Nxy = np.array([Nx,Ny]).T
    M_tri = Delaunay(M_Nxy).simplices.copy()
#    Nn = M_Nxy.shape[0]
    Ne = M_tri.shape[0]
    M_GDof = np.array([2*M_tri[:,0],2*M_tri[:,0]+1,
                       2*M_tri[:,1],2*M_tri[:,1]+1,
                       2*M_tri[:,2],2*M_tri[:,2]+1]).T
    
    M_EleCentre = np.zeros((Ne,2))
    for i in range(0,Ne):
        noeuds = M_tri[i,:]
        X = np.mean(M_Nxy[noeuds,0])
        Y = np.mean(M_Nxy[noeuds,1])
        M_EleCentre[i,:] = np.array([X,Y])
    
    MinX = min(Nx)
    MaxX = max(Nx)
    MinY = min(Ny)
    MaxY = max(Ny)
    Borda = np.where(abs(Ny-MinY)<1e-6)[0]
    Bordb = np.where(abs(Nx-MaxX)<1e-6)[0]
    Bordc = np.where(abs(Ny-MaxY)<1e-6)[0]
    Bordd = np.where(abs(Nx-MinX)<1e-6)[0]
    
    Bord = Borda[Nx[Borda].argsort()]
    BordD = Bord[1:]
    
    Bord = Bordb[Ny[Bordb].argsort()]
    BordR = Bord
    
    Bord = Bordc[Nx[Bordc].argsort()]
    BordU = Bord[1:]
    
    Bord = Bordd[Ny[Bordd].argsort()]
    BordL = Bord

    return M_tri,M_Nxy,M_GDof,M_EleCentre,BordD,BordR,BordU,BordL


M_tri,M_Nxy,M_GDof,M_EleCentre,BordD,BordR,BordU,BordL = fun_Mesh_fix(L,nx)
Ne = M_tri.shape[0]
Nn = M_Nxy.shape[0]
GDof = 2*Nn
Ae = L**2/Ne
FGlob = np.zeros(GDof)

M3_B = np.zeros((Ne,3,6));
for i in range(Ne):
    noeuds = M_tri[i,:]
    x = M_Nxy[noeuds,0]
    y = M_Nxy[noeuds,1]
    M3_B[i,:,:] = 0.5/Ae*np.array([[y[1]-y[2], 0, y[2]-y[0], 0, y[0]-y[1], 0],
                                   [0, x[2]-x[1], 0, x[0]-x[2], 0, x[1]-x[0]],
                 [x[2]-x[1],y[1]-y[2],x[0]-x[2],y[2]-y[0],x[1]-x[0],y[0]-y[1]]]) 


""" Boundary Conditions """
def fun_Boundary_Perio(BordD,BordU,BordL,BordR,GDof):
    C = np.zeros((2*(len(BordD)+len(BordL)),GDof))
    Ud1 = np.zeros(2*(len(BordD)+len(BordL)))
    Ud2 = np.zeros(2*(len(BordD)+len(BordL)))
    Ud3 = np.zeros(2*(len(BordD)+len(BordL)))
    cnt = 0
    for i in range(len(BordD)):
        Ba = BordD[i]
        Bc = BordU[i]
        C[2*cnt,2*np.array([Ba,Bc])] = np.array([-1,1])
        C[2*cnt+1,2*np.array([Ba,Bc])+1] = np.array([-1,1])
        
        Ud1[[2*cnt,2*cnt+1]] = [-M_Nxy[Ba,0]+M_Nxy[Bc,0],0]
        Ud2[[2*cnt,2*cnt+1]] = [0,-M_Nxy[Ba,1]+M_Nxy[Bc,1]]
        Ud3[[2*cnt,2*cnt+1]] = [0.5*(-M_Nxy[Ba,1]+M_Nxy[Bc,1]),
                                0.5*(-M_Nxy[Ba,0]+M_Nxy[Bc,0])]
        cnt+=1
        
    for i in range(len(BordL)):
        Ba = BordL[i]
        Bc = BordR[i]
        C[2*cnt,2*np.array([Ba,Bc])] = np.array([-1,1])
        C[2*cnt+1,2*np.array([Ba,Bc])+1] = np.array([-1,1])
        
        Ud1[[2*cnt,2*cnt+1]] = [-M_Nxy[Ba,0]+M_Nxy[Bc,0],0]
        Ud2[[2*cnt,2*cnt+1]] = [0,-M_Nxy[Ba,1]+M_Nxy[Bc,1]]
        Ud3[[2*cnt,2*cnt+1]] = [0.5*(-M_Nxy[Ba,1]+M_Nxy[Bc,1]),
                                0.5*(-M_Nxy[Ba,0]+M_Nxy[Bc,0])]
        cnt+=1
    return C, Ud1, Ud2, Ud3

def fun_Boundary_KUBC(BordD,BordU,BordL,BordR,GDof):
    Bord = np.unique(np.hstack([BordD,BordU,BordL,BordR]))
    Nd = len(Bord)
    C = np.zeros((2*Nd,GDof))
    Ud1 = np.zeros(2*Nd)
    Ud2 = np.zeros(2*Nd)
    Ud3 = np.zeros(2*Nd)
    for i in range(Nd):
        C[i,2*Bord[i]] = 1
        C[Nd+i,2*Bord[i]+1] = 1
        
        Ud1 = np.hstack([M_Nxy[Bord,0],np.zeros((Nd))])
        Ud2 = np.hstack([np.zeros((Nd)),M_Nxy[Bord,1]])
        Ud3 = 0.5*np.hstack([M_Nxy[Bord,1],M_Nxy[Bord,0]])

    return C, Ud1, Ud2, Ud3

C, Ud1, Ud2, Ud3 = fun_Boundary_Perio(BordD,BordU,BordL,BordR,GDof)
#C, Ud1, Ud2, Ud3 = fun_Boundary_KUBC(BordD,BordU,BordL,BordR,GDof)

"""  Phase Classification  """
def fun_Phase_RandomCycle(M_tri,M_Nxy,M_EleCentre,nbi,r):
    M_centP = np.zeros((9*nbi,2))
    M_Trans = np.array([[-L,-L],[0,-L],[L,-L],
                        [-L, 0],[0, 0],[L, 0],
                        [-L, L],[0, L],[L, L]])
#    centre = L*np.random.rand(2)-0.5*L
    centre = np.zeros(2)
    M_centP[0:9,:] = centre+M_Trans
    
    for i in range(1,nbi):
        centre = L*np.random.rand(2)-0.5*L
        V_dcc = sqrt((M_centP[0:9*i,0]-centre[0])**2 
                    +(M_centP[0:9*i,1]-centre[1])**2)
        while any(V_dcc<2.05*r):
            centre = L*np.random.rand(2)-0.5*L
            V_dcc = sqrt((M_centP[0:9*i,0]-centre[0])**2 
                         +(M_centP[0:9*i,1]-centre[1])**2)
        
        print(i)
        M_centP[9*i:9*(i+1),:] = centre+M_Trans
        
    V_group1 = (abs(M_centP[:,0])<0.5*L+r) & (abs(M_centP[:,1])<0.5*L)
    V_group2 = (abs(M_centP[:,0])<0.5*L) & (abs(M_centP[:,1])<0.5*L+r)
    V_group3 = np.sum((abs(M_centP)-0.5*L)**2,axis=1)<r**2
    V_varie = V_group1 | V_group2| V_group3
    M_CentFin = M_centP[V_varie,:]
    nbi = M_CentFin.shape[0]
    Ne = M_tri.shape[0]
    
    V_Phase = np.zeros(Ne).astype(int)
    
    for i in range(nbi):
        Mxy = M_EleCentre-M_CentFin[i,:]
        V_ph1 = (Mxy[:,0]**2+Mxy[:,1]**2)<r**2;
        V_Phase[V_ph1] = 1; # Matrix
    return V_Phase

V_Phase = fun_Phase_RandomCycle(M_tri,M_Nxy,M_EleCentre,nbi,r)

"""  Stress Matrix  """
M3_C = np.zeros((2,3,3))
Lambda = lambd[0]
Mu = mu[0]
M3_C[0,:,:] = np.array([[Lambda+2*Mu, Lambda, 0],
                        [Lambda, Lambda+2*Mu, 0],
                        [0 , 0 , Mu ]])

Lambda = lambd[1]
Mu = mu[1]
M3_C[1,:,:] = np.array([[Lambda+2*Mu, Lambda, 0],
                        [Lambda, Lambda+2*Mu, 0],
                        [0 , 0 , Mu ]])
KGlob = np.zeros((GDof,GDof))
for i in range(Ne):
    M_B = M3_B[i,:,:]
    KGlob[np.ix_(M_GDof[i,:],M_GDof[i,:])] = (KGlob[np.ix_(M_GDof[i,:],M_GDof[i,:])]
    +Ae*np.dot(np.dot(M_B.T,M3_C[V_Phase[i],:,:]),M_B))

Nd = len(Ud1)
Klag = np.vstack((np.hstack((KGlob,C.T)),np.hstack((C,np.zeros((Nd,Nd))))))

Flag1 = np.hstack((FGlob,Ud1));
Ulag1 = np.linalg.solve(Klag,Flag1)
UX1 = Ulag1[np.arange(0,GDof,2)]
UY1 = Ulag1[np.arange(1,GDof,2)]
p_def1 = M_Nxy + 1*np.array([UX1, UY1]).T

Flag2 = np.hstack((FGlob,Ud1));
Ulag2 = np.linalg.solve(Klag,Flag2)
UX2 = Ulag2[np.arange(0,GDof,2)]
UY2 = Ulag2[np.arange(1,GDof,2)]
p_def2 = M_Nxy + 1*np.array([UX2, UY2]).T

Flag3 = np.hstack((FGlob,Ud3));
Ulag3 = np.linalg.solve(Klag,Flag3)
UX3 = Ulag3[np.arange(0,GDof,2)]
UY3 = Ulag3[np.arange(1,GDof,2)]
p_def3 = M_Nxy + .1*np.array([UX3, UY3]).T
"""  Plot  """
BordA = BordR
plt.figure(1)
plt.triplot(M_Nxy[:,0],M_Nxy[:,1], M_tri[V_Phase==0,:])
plt.triplot(M_Nxy[:,0],M_Nxy[:,1], M_tri[V_Phase>0,:])
V = 0.5*np.array([-L,L,-L,L])
plt.axis(V)
plt.axis('square')
#plt.plot(M_EleCentre[:,0], M_EleCentre[:,1], 'o')
#plt.plot(M_Nxy[BordA,0], M_Nxy[BordA,1], 'o')
plt.figure(2)
plt.triplot(p_def1[:,0],p_def1[:,1], M_tri)
plt.triplot(p_def1[:,0],p_def1[:,1], M_tri[V_Phase==1,:])

plt.figure(3)
plt.triplot(p_def3[:,0],p_def3[:,1], M_tri)
plt.triplot(p_def3[:,0],p_def3[:,1], M_tri[V_Phase==1,:])

stop = time()
print(str(stop-start) + "s")