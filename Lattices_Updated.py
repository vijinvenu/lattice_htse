# -*- coding: utf-8 -*-
"""
Created on Sun May  7 21:27:43 2017

Lattices_Updated.py

Try to make lattices class-based and more user friendly.

Probably, the ideal way to do this would be to have, for a given lattice depth
and dimensionality, a well-defined set of quasimomenta and energies stored as 
a property of the lattice, with some constants as well. 

Should properties such as tunnel coupling, interaction energy, and the 
Wannier functions/Bloch waves be considered properties of the lattice, or 
quantities to be calculated via functions? Maybe the tunnel coupling can be a 
well-defined property, but everything else should only be calculated when 
necessary. 

Some files should also be saved for bandstructure, and maybe for a density of
states? These things could be referred to later without the need for constant
recalculation. 
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

class Lattice:
    
    #Define some constants.
    N_Bands = 20 #Number of bands.
    Wavelength = 1053.6E-9 #Lattice spacing in metres.
    hbar = 6.626E-34/2/np.pi #Planck's constant.
    m_K = 0.039964/6.022E23 #Mass of potassium.
    x_Vector = np.linspace(-3,3,num=501) #Position vector used for Wannier.
    Order_J = 20 #Number of tunnel couplings to consider.
    
    #Extend this to allow for variable depths in each dimension.
    def __init__(self,Depth=2,Dim=3,N_Quasimomenta=101,w=np.array([60.0,60.0,85.0])):
        self.Depth = Depth
        self.Dim = Dim
        self.N_Quasimomenta = N_Quasimomenta
        self.Waists = w * 1E-6
        self.Er = self.hbar**2 * (np.pi / (self.Wavelength/2))**2 / (2 * self.m_K)
        self.Band_Structure()
        self.Tunneling()
        self.Bandwidth()
        self.Bandgap()
        self.Harmonic_Trap_Frequencies()
        self.Quasimomentum_Dependent_Effective_Mass()
        
    def Band_Structure(self):
        #Define some Vectors for convenience.
        self.q_Vector = np.linspace(-np.pi,np.pi,num=self.N_Quasimomenta) #Quasimomentum Vector.
        p_Vector = np.linspace(-self.N_Bands,self.N_Bands,num=2*self.N_Bands+1) #Band index Vector.
        self.Energies = np.zeros((self.N_Quasimomenta,2*self.N_Bands+1),dtype=float)
        self.States = np.zeros((self.N_Quasimomenta,2*self.N_Bands+1,2*self.N_Bands+1),dtype=complex)
    
        for q in range (0,self.N_Quasimomenta):
            H = np.zeros((2*self.N_Bands+1,2*self.N_Bands+1), dtype=float)
            for p in range (0,2*self.N_Bands+1):
                H[p][p] = (2*p_Vector[p] + self.q_Vector[q]/np.pi)**2
                if p>0:
                    H[p][p-1] = self.Depth/4
                if p<2*self.N_Bands:
                    H[p][p+1] = self.Depth/4
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            sort_indices = np.argsort(eigenvalues)
            #eigenvectors = eigenvectors*np.exp(-1j*np.angle(eigenvectors[self.N_Bands+2-p%2,p]))
            #Could include multiple band in this at some later date.
            self.States[q,:,:] = eigenvectors[:,sort_indices] #Get the eigenvector corresponding to lowest band.
            self.Energies[q,:] = eigenvalues[sort_indices]
        
        return
        
    def Tunneling(self):
        Phase_Matrix = np.zeros((self.Order_J,self.N_Quasimomenta),dtype=complex)
        for Counter in range(0,self.Order_J):
            Phase_Matrix[Counter,:] = np.exp(-1j*(Counter+1) * self.q_Vector) * (-1)**(Counter+1)
        self.J = np.dot(Phase_Matrix, self.Energies[:,0])
        self.J = np.real(self.J - Phase_Matrix[:,0] * self.Energies[0,0])/(self.N_Quasimomenta-1)
        return
    
    def Bandwidth(self):
        self.Width = self.Dim*(np.max(self.Energies[:,0]) - np.min(self.Energies[:,0]))
        return
        
    def Bandgap(self):
        self.Gap1D = np.min(self.Energies[:,1]) - np.max(self.Energies[:,0])
        self.GapDim = np.max([np.min(self.Energies[:,1]) - ((self.Dim-1)/self.Dim*self.Width+np.max(self.Energies[:,0])),0.0])
        return
    
    def Bloch_Waves(self,q,x_Vector,Band_Index):
        Bloch_Wave = np.zeros(len(x_Vector),dtype=complex)
        for l in range (0,2*self.N_Bands+1):
            Bloch_Wave = Bloch_Wave + self.States[q,l,Band_Index] * np.exp(2 * 1j * (l-self.N_Bands) * np.pi * (x_Vector - 0.5))
        return Bloch_Wave
        
    def Wannier_Functions(self,x_Vector,Band_Index):
        Wannier = np.zeros(len(x_Vector),dtype=complex)
        for q in range (0,len(self.q_Vector)-1):
            Bloch_Wave = self.Bloch_Waves(q,x_Vector,Band_Index)
            Wannier = Wannier + np.exp(1j * self.q_Vector[q] * x_Vector) * Bloch_Wave
        Wannier = Wannier / len(self.q_Vector)
        return Wannier
    
    def Interaction_Energy(self,B_Field):
        f = np.abs(self.Wannier_Functions(self.x_Vector,0)/np.sqrt(self.Wavelength/2))**4
        x = self.x_Vector * self.Wavelength/2
        U = (4*np.pi*self.hbar**2*Scattering_Length(B_Field)/self.m_K) * Simpsons_Integration(f,x) ** self.Dim
        U = U / self.Er
        return U
    
    #Harmonic trap frequenices in Hz.
    def Harmonic_Trap_Frequencies(self):
        self.nu = 2*self.Er/(np.pi/(self.Wavelength/2) * self.Waists) * np.sqrt(2*self.Depth - np.sqrt(self.Depth))/self.hbar/(2*np.pi)
        self.Mean_nu = 1
        for Counter in range (0,len(self.nu)):
            self.Mean_nu = self.Mean_nu * self.nu[Counter]
        self.Mean_nu = self.Mean_nu ** (1/len(self.nu))
        
    #Include XDT trap frequency here. 
    def Fermi_Energy(self,N_Atoms=5000,nu_XDT=38):
        l_osc = np.sqrt(self.hbar/self.m_K/(self.Mean_nu*2*np.pi))
        Omega_D = 2 * np.pi**(self.Dim/2) / sp.special.gamma(self.Dim/2)
        E_F = (self.hbar * self.Mean_nu * 2 * np.pi) / 2 * (self.Wavelength/2/l_osc)**2 * (self.Dim*N_Atoms/Omega_D)**(2.0/self.Dim) / self.Er
        return E_F

    #Maybe make this a function that recalculates bandstructure, or interpolates, with a finer grain?
    def Quasimomentum_Dependent_Effective_Mass(self):
        #First, estimate the second derivative.
        dq = 2 / (self.N_Quasimomenta-1)
        Second_Derivative = (np.vstack((self.Energies[1:,:],self.Energies[0,:])) + np.vstack((self.Energies[-1,:],self.Energies[0:-1,:]))-2*self.Energies)/dq**2
        #Contains all effective masses, except those of the very lightest holes.
        Second_Derivative = Second_Derivative[1:-1]
        self.Mass_Ratio_q = 2/Second_Derivative
        return
    
    def Generate_Density_of_States(self):
        #How to do this? We have a matrix of eigenenergies in 1D, with rows being quasimomentum in 1D,
        #columns being band.
        #One way to do this involves a 3, 3D meshgrids for each band. These hold the index in k-space.
        #The energy is a 3D cube where the energy at each point is the sum of 3. 
        start = time.time()
        E_x, E_y, E_z = np.meshgrid(self.Energies[:,0],self.Energies[:,0],self.Energies[:,0])
        E = E_x + E_y + E_z
        Min_E = np.min(E)
        Max_E = np.max(E)+np.finfo(float).eps
        Num_Points = 101
        E_Vector = np.linspace(Min_E,Max_E,num=Num_Points)
        dE = (Max_E - Min_E) / (Num_Points - 1)
        #gE = np.zeros(len(E_Vector)-1)
        gE = np.array([np.sum((E>=el1)*(E<el2)) for el1, el2 in zip(E_Vector,E_Vector[1:])])/dE
#        for Counter in range(0,Num_Points-1):
#            gE[Counter] = np.sum((E>E_Vector[Counter])*(E<E_Vector[Counter+1])) / dE

        plt.plot(E_Vector[0:-1],gE)
        end = time.time()
        print('Elapsed time is ', end-start)
        return gE, dE
    
    def Generate_YZ_Density_of_States(self):
        #How to do this? We have a matrix of eigenenergies in 1D, with rows being quasimomentum in 1D,
        #columns being band.
        #One way to do this involves a 3, 3D meshgrids for each band. These hold the index in k-space.
        #The energy is a 3D cube where the energy at each point is the sum of 3. 
        start = time.time()
        E_x, E_y, E_z = np.meshgrid(self.Energies[:,0],self.Energies[:,0],self.Energies[:,0])
        E = E_x + E_y + E_z
        Min_E = np.min(E)
        Max_E = np.max(E)+np.finfo(float).eps
        Num_Points = 101
        E_Vector = np.linspace(Min_E,Max_E,num=Num_Points)
        dE = (Max_E - Min_E) / (Num_Points - 1)
        #gE = np.zeros(len(E_Vector)-1)
        gYZ = np.array([np.sum((E>=el1)*(E<el2),axis=(0,1)) for el1, el2 in zip(E_Vector,E_Vector[1:])])/dE
#        for Counter in range(0,Num_Points-1):
#            gE[Counter] = np.sum((E>E_Vector[Counter])*(E<E_Vector[Counter+1])) / dE

        print(np.sum(gYZ*dE))
        #plt.plot(E_Vector[0:-1],)
        end = time.time()
        print('Elapsed time is ', end-start)
        return gYZ, dE
    
    
def Scattering_Length(B_Field):
    a_s = 167.0*.52198*1E-10 #Background scattering length.
    Delta = 6.9 #Width of resonance.
    B_0 = 202.1 #s-wave FB resonance.
    a = a_s * (1 - Delta / (B_Field - B_0))
    return a  

#This should not really be part of the lattice class... just its own function.
def Simpsons_Integration(f,x):
    Odd_Indices = (np.mod(np.linspace(0,len(x)-1,len(x)),2)==1)
    Odd_Indices[-1] = 0
    #Define even indices, except never the first or last point.
    Even_Indices = (np.mod(np.linspace(0,len(x)-1,len(x)),2)==0)
    Even_Indices[0] = 0
    Even_Indices[-1] = 0
    I = (x[1]-x[0])/3 * (f[0] + 4 * np.sum(f[Odd_Indices]) + 2 * np.sum(f[Even_Indices]) + f[-1])
    return I

        
    
    
    