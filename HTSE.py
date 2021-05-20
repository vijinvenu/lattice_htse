# -*- coding: utf-8 -*-
"""
Spyder Editor

Creates a HTSE for a given lattice. Can also fit a radial averaged image.
"""
import numpy as np
import matplotlib.pyplot as plt
import Lattices_Updated as Lat_Up
import scipy.optimize as opt

class HTSE:
    
    #Define some constants.
    a = 526.8E-9 #Lattice spacing, nm.
    hbar = 6.626E-34/2/np.pi #Planck's constant.
    m_K = 0.039964/6.022E23 #Mass of potassium.
    E_R = hbar**2 * (np.pi / a)**2 / (2 * m_K)
    
    #Initialize the class. An HTSE has properties like entropy and number
    #derived from more basic input parameters like chemical potential,
    #trapping potential, temperature, and tunnelling rate. Initialize the class
    #to have a well-defined set of parameters. Provide a method to fit 
    #certain parameters to a series of images input. Have methods for display
    #of fit curves. 
    def __init__(self,B,V0,nu_XDT,mu,T): #Can we choose between specifying depth+B vs specifying U,t,and nu?
        self.B = B #Magnetic field.
        self.V0 = V0 #Lattice depth in units of recoil energy
        self.nu_XDT = nu_XDT #XDT radial trap frequency.
        self.mu = mu #Chemical potential.
        self.T = T #Temperature.
        self.beta = 1/self.T #Inverse temperature.
        self.nu = np.array([0.0,0.0,0.0])
        self.Lattice_Parameters()
        self.Order = 4
        self.N = []
        self.n_Av = []
        self.S = []

    #Get lattice parameters.
    def Lattice_Parameters(self):
        L = Lat_Up.Lattice(self.V0,3,1001)
        self.t = L.J[0]
        self.U = L.Interaction_Energy(self.B)
        self.nu[0] = np.sqrt(self.nu_XDT**2 + L.nu[1]**2 + L.nu[2]**2) #Radial trap frequency in units of Er
        self.nu[1] = np.sqrt(self.nu_XDT**2 + L.nu[0]**2 + L.nu[2]**2) #Radial trap frequency.
        self.nu[2] = np.sqrt((self.nu_XDT*7)**2 + L.nu[0]**2 + L.nu[1]**2) #Axial trap frequency.
        return
    
    def Set_Order(self, New_Order):
        self.Order = New_Order
        return
    
    def Fit_Parity(self,Sites,Radial_Average,Peak):
        #Fit parameters are mu, T, A, and b.
        fit_bounds=((-5,0,1500,-50),(5,5,2000,50))
        (res,cov) = opt.curve_fit(self.Parity,Sites,Radial_Average,p0=(self.mu,self.T,Peak,0.0),bounds=fit_bounds)
        self.mu = res[0]
        self.T = res[1]
        self.beta = 1/self.T
        self.N=[]
        self.S=[]
        return res

    def Number(self):
        if not bool(self.N):
            Filling = np.array([self.Filling(np.array([x,y,z])) for x in range(0,51) for y in range(0,51) for z in range(0,11)])
            self.N = 8*np.sum(Filling)
        return self.N
    
    def Average_Filling(self):
        if not bool(self.n_Av):
            Filling = np.array([self.Filling(np.array([x,y,z])) for x in range(0,51) for y in range(0,51) for z in range(0,11)])
            self.n_Av = np.sum(Filling**2)/np.sum(Filling)
        return self.n_Av
    
    def Average_Entropy(self):
        if not bool(self.S):
            [Site_Entropy,Particle_Entropy] = np.array([self.Entropy(np.array([x,y,z])) for x in range(0,51) for y in range(0,51) for z in range(0,11)]).T
            self.S = 8*np.sum(Site_Entropy)/self.Number()
        return self.S

    #The harmonic trapping potential at a specific set of site indices.
    def Potential(self, Site):
        V = 1/2 * self.m_K * np.dot((2 * np.pi * self.nu)**2, np.transpose((Site * self.a)**2)) / self.E_R
        return V
    
    #The fugacity.
    def Fugacity(self,Site,mu=None,T=None):
        if mu is None and T is None:
            mu = self.mu
            T = self.T
        z = np.exp(1/T * (mu - self.Potential(Site)))
        return z
    
    #Something that looks like fugacity, but uses U.
    def Ugacity(self,T=None):
        if T is None:
            T = self.T
        w = np.exp(-1/T * self.U)
        return w
    
    def Entropy(self, Site):
        t = self.t
        b = self.beta
        w = self.Ugacity()
        z = self.Fugacity(Site)
        if (self.Order <= 2):
            if self.Order == 0:
                t = 0
            s = (b**2*((-6*t**2*z*(1 + 2*z + w*z**2)*b**2*(2*(-1 + w)*z + (1 + w*z**2)*np.log(w)) - 
            z*(1 + 2*z + w*z**2)**2*np.log(w)*(w*z*np.log(w) + 2*(1 + w*z)*np.log(z)) + 
            12*t**2*z**2*b**2*(2*(-1 + w)*z + (1 + w*z**2)*np.log(w))*(w*z*np.log(w) + 2*(1 + w*z)*np.log(z)) - 
            6*t**2*z*(1 + 2*z + w*z**2)*b**2*(w*z**2*np.log(w)**2 + 2*(-1 + w)*z*(-1 + 2*np.log(z)) + 
            np.log(w)*(2*w*z + np.log(z) + 3*w*z**2*np.log(z))) + (1 + 2*z + w*z**2)**3*np.log(w)*
            np.log(1 + 2*z + w*z**2))/((1 + 2*z + w*z**2)**3*b**2*np.log(w))))
        elif (self.Order > 2):
            s = (b**2*((-(w*z**2*(2 + 2*w**4*z**8 + z*(16 - 12*t**2*b**2 - 27*t**4*b**4) + 
            w**3*z**7*(16 - 12*t**2*b**2 - 15*t**4*b**4) + 
            2*w**2*z**6*(24 + 4*w - 12*t**2*b**2 + 149*t**4*b**4) + 
            z**3*(64 + 48*t**2*b**2 - 496*t**4*b**4 + w*(48 - 36*t**2*b**2 - 33*t**4*b**4)) + 
            w*z**5*(64 + 48*t**2*b**2 - 560*t**4*b**4 - 3*w*(-16 + 12*t**2*b**2 + 7*t**4*b**4)) + 
            z**2*(8*w + 6*(8 - 4*t**2*b**2 + 51*t**4*b**4)) + 
            4*z**4*(3*w**2 + 8*(1 + 3*t**2*b**2 + 3*t**4*b**4) + 
            w*(24 - 12*t**2*b**2 + 113*t**4*b**4)))*np.log(w)**4) + 
            240*t**4*z**2*(1 - 2*z + w**4*z**6 + w*(-1 + 2*z - 7*z**2) - w**3*z**4*(7 + 2*z + z**2) + 
            w**2*z**2*(7 + 7*z**2 + 2*z**3))*b**4*np.log(z) + 48*t**4*z**2*b**4*np.log(w)*
            (-(w**4*z**6) - w**3*z**4*(17 + 20*z + 3*z**2) - 3*(-2 - 4*z + z**2 + 2*z**3) + 
            2*w**2*z**2*(9 + 15*z + 10*z**2 + 8*z**3) + w*(-6 - 12*z - 15*z**2 - 20*z**3 + z**4) + 
            (7*w**4*z**6 - 12*(-1 + 2*z + z**2) - w**3*z**4*(37 + 20*z + 4*z**2) + 
            w**2*z**2*(37 + 10*z + 40*z**2 + 2*z**3) + w*(-7 + 14*z - 60*z**2 + 10*z**3 + 16*z**4))*
            np.log(z)) + 24*t**2*z**2*b**2*np.log(w)**2*
            (t**2*(19 + 20*z - 36*z**2 + 7*w**4*z**6 + w**3*z**4*(-37 - 28*z + 15*z**2) + 
            w**2*z**2*(37 + 80*z + 89*z**2 + 36*z**3 - 4*z**4) - 
            w*(7 - 4*z + 43*z**2 + 128*z**3 + 8*z**5))*b**2 + 
            (2 + 2*w**4*z**6 + 19*t**2*b**2 + z*(8 - 65*t**2*b**2) + 2*z**2*(4 + 9*t**2*b**2) + 
            w**3*z**4*(2 + z*(8 - 33*t**2*b**2) - z**2*(2 + 23*t**2*b**2)) - 
            w*(2 - 10*t**2*z**3*b**2 + 8*t**2*z**5*b**2 + z*(8 - 27*t**2*b**2) + 
            z**4*(8 + 26*t**2*b**2) + z**2*(6 + 151*t**2*b**2)) + 
            w**2*z**2*(-2 + 10*t**2*z*b**2 + 4*t**2*z**4*b**2 + z**3*(-8 + 67*t**2*b**2) + 
            z**2*(6 + 167*t**2*b**2)))*np.log(z)) + 
            np.log(w)**3*(-3*t**2*z*b**2*(4 + 15*t**2*b**2 + z*(24 + 8*w - 74*t**2*b**2) + 
            w**4*z**8*(4 + 15*t**2*b**2) + 2*w*z**5*(48 + 68*w - 4*w**2 + 112*t**2*b**2 - 
            603*t**2*w*b**2) + 8*w*z**6*(4*t**2*b**2 + 28*t**2*w**2*b**2 + w*(14 - 87*t**2*b**2)) - 
            8*z**2*(-6 + 20*t**2*b**2 + w*(-8 + 3*t**2*b**2)) + 
            2*w**2*z**7*(-4*w**2 - 24*t**2*b**2 + 5*w*(4 + 11*t**2*b**2)) + 
            2*z**3*(16 + 4*w**2 + 48*t**2*b**2 + w*(92 + 49*t**2*b**2)) + 
            2*w*z**4*(112 - 92*t**2*b**2 + w*(28 + 53*t**2*b**2))) - 
            z*(4 + 8*(8 + 40*w + 15*w**2)*z**4 + 4*w**5*z**9 + 12*t**2*b**2 + 15*t**4*b**4 + 
            z*(32 + 4*w + 24*t**2*b**2 - 298*t**4*b**4) - 
            3*w**4*z**8*(-12 + 4*t**2*b**2 + 5*t**4*b**4) + 
            2*w**3*z**7*(64 + 8*w - 12*t**2*b**2 + 149*t**4*b**4) + 
            2*w**2*z**6*(8*(14 + 3*t**2*b**2 - 35*t**4*b**4) + w*(56 - 12*t**2*b**2 + 3*t**4*b**4)) + 
            2*z**3*(8*w**2 + w*(96 + 12*t**2*b**2 - 73*t**4*b**4) - 
            16*(-4 + 3*t**2*b**2 + 3*t**4*b**4)) + z**2*(-6*w*(-8 - 4*t**2*b**2 + t**4*b**4) + 
            16*(6 - 3*t**2*b**2 + 35*t**4*b**4)) + 
            2*w*z**5*(12*w**2 + 48*(2 + t**2*b**2 + t**4*b**4) + 
            w*(144 - 12*t**2*b**2 + 73*t**4*b**4)))*np.log(z) + 2*(1 + 2*z + w*z**2)**5*
            np.log(1 + 2*z + w*z**2)))/(2*(1 + 2*z + w*z**2)**5*b**2*np.log(w)**3)))
        return s, s/self.Filling(Site) #Entropy per site, and entropy per particle.
    
    #Take derivative of grand potential to get filling. 2nd order approx for now.
    def Filling(self, Site, mu=None, T=None):
        if mu is None and T is None:
            mu = self.mu
            T = self.T
        t = self.t
        b = 1/T
        w = self.Ugacity(T)
        z = self.Fugacity(Site,mu,T)
        if (self.Order <= 2):
            if self.Order == 0:
                t = 0
            n = (-((-2*z*(-12*t**2*z*(1 + w**2*z**2 - w*(1 + z**2))*b**2 + 
            (1 + (4 + 6*w)*z**2 + w**3*z**5 + 3*t**2*b**2 + z*(4 + w - 6*t**2*b**2) + 
            w**2*z**4*(5 - 3*t**2*b**2) + 2*w*z**3*(4 + w + 3*t**2*b**2))*np.log(w)))/
            ((1 + 2*z + w*z**2)**3*np.log(w))))
        elif (self.Order > 2):
            n = (-(-(z*(-240*t**4*z*(1 - 2*z + w**4*z**6 + w*(-1 + 2*z - 7*z**2) - w**3*z**4*(7 + 2*z + z**2) + 
            w**2*z**2*(7 + 7*z**2 + 2*z**3))*b**4 - 48*t**4*z*(7*w**4*z**6 - 12*(-1 + 2*z + z**2) - 
            w**3*z**4*(37 + 20*z + 4*z**2) + w**2*z**2*(37 + 10*z + 40*z**2 + 2*z**3) + 
            w*(-7 + 14*z - 60*z**2 + 10*z**3 + 16*z**4))*b**4*np.log(w) + 
            24*t**2*z*b**2*(-2 - 2*w**4*z**6 - 19*t**2*b**2 - 2*z**2*(4 + 9*t**2*b**2) + 
            z*(-8 + 65*t**2*b**2) + w**3*z**4*(-2 + z**2*(2 + 23*t**2*b**2) + 
            z*(-8 + 33*t**2*b**2)) + w*(2 - 10*t**2*z**3*b**2 + 8*t**2*z**5*b**2 + 
            z*(8 - 27*t**2*b**2) + z**4*(8 + 26*t**2*b**2) + z**2*(6 + 151*t**2*b**2)) - 
            w**2*z**2*(-2 + 10*t**2*z*b**2 + 4*t**2*z**4*b**2 + z**3*(-8 + 67*t**2*b**2) + 
            z**2*(6 + 167*t**2*b**2)))*np.log(w)**2 + (4 + 8*(8 + 40*w + 15*w**2)*z**4 + 4*w**5*z**9 + 
            12*t**2*b**2 + 15*t**4*b**4 + z*(32 + 4*w + 24*t**2*b**2 - 298*t**4*b**4) - 
            3*w**4*z**8*(-12 + 4*t**2*b**2 + 5*t**4*b**4) + 
            2*w**3*z**7*(64 + 8*w - 12*t**2*b**2 + 149*t**4*b**4) + 
            2*w**2*z**6*(8*(14 + 3*t**2*b**2 - 35*t**4*b**4) + w*(56 - 12*t**2*b**2 + 3*t**4*b**4)) + 
            2*z**3*(8*w**2 + w*(96 + 12*t**2*b**2 - 73*t**4*b**4) - 
            16*(-4 + 3*t**2*b**2 + 3*t**4*b**4)) + z**2*(-6*w*(-8 - 4*t**2*b**2 + t**4*b**4) + 
            16*(6 - 3*t**2*b**2 + 35*t**4*b**4)) + 
            2*w*z**5*(12*w**2 + 48*(2 + t**2*b**2 + t**4*b**4) + 
            w*(144 - 12*t**2*b**2 + 73*t**4*b**4)))*np.log(w)**3))/(2*(1 + 2*z + w*z**2)**5*np.log(w)**3)))
        return n
    
    #Derivative of grand potential with respect to interaction energy gives doublon fraction.
    def Doublons(self, Site, mu=None, T=None):
        if mu is None and T is None:
            mu = self.mu
            T = self.T
        t = self.t
        b = 1/T
        w = self.Ugacity(T)
        z = self.Fugacity(Site,mu,T)
        if (self.Order <= 2):
            if self.Order == 0:
                t = 0
            d = (z**2*(w*(1 + 2*z + w*z**2)**2 - (12*t**2*w*z*b**2*(2*(-1 + w)*z + (1 + w*z**2)*np.log(w)))/
            np.log(w) + (6*t**2*(1 + 2*z + w*z**2)*b**2*(2 - 2*w + 2*w*np.log(w) + w*z*np.log(w)**2))/
            np.log(w)**2))/(1 + 2*z + w*z**2)**3
        elif (self.Order > 2):
            d = ((z**2*(-360*t**4*(-1 + w*z**2)**2*(-1 + w - 2*z + 2*w*z - w*z**2 + w**2*z**2)*b**4 - 
            24*t**4*(19*w**4*z**6 - w**3*z**4*(37 + 10*z + 18*z**2) + 12*(-2 - 4*z + z**2 + 2*z**3) + 
            w*(9 + 18*z - 30*z**2 + 20*z**3 - 4*z**4) + w**2*(33*z**2 + 40*z**4 - 4*z**5))*b**4*
            np.log(w) - 12*t**2*b**2*(2*w**4*z**6*(1 + 7*t**2*b**2) - 
            (1 + 2*z)*(2 + 8*z**2 + 19*t**2*b**2 + z*(8 - 18*t**2*b**2)) - 
            w**3*z**4*(-6 + 74*t**2*b**2 + z**2*(2 + 39*t**2*b**2) + 2*z*(-6 + 61*t**2*b**2)) + 
            w*(2 - 14*t**2*b**2 + 8*t**2*z**5*b**2 + z**2*(18 - 137*t**2*b**2) + 
            z*(12 - 46*t**2*b**2) + 24*z**4*(-1 + 4*t**2*b**2) + z**3*(-8 + 68*t**2*b**2)) + 
            w**2*z**2*(6 + 74*t**2*b**2 + 4*t**2*z**4*b**2 - 12*z**3*(1 + 2*t**2*b**2) + 
            8*z*(3 + 5*t**2*b**2) + z**2*(18 + 91*t**2*b**2)))*np.log(w)**2 + 
            24*t**2*w*b**2*(1 + z*(6 - 9*t**2*b**2) + z**2*(14 + w + 39*t**2*b**2) + 
            z**4*(8 - w**2 + 16*t**2*b**2 + w*(8 - 124*t**2*b**2)) + 
            z**3*(16 + 9*t**2*b**2 + w*(4 + 5*t**2*b**2)) + 
            z**5*(4*t**2*b**2 + w*(8 - 67*t**2*b**2) + w**2*(-2 + 22*t**2*b**2)) - 
            w*z**6*(w**2 + 6*t**2*b**2 - w*(2 + 23*t**2*b**2)))*np.log(w)**3 + 
            w*(2 + 2*w**4*z**8 + z*(16 - 12*t**2*b**2 - 27*t**4*b**4) + 
            w**3*z**7*(16 - 12*t**2*b**2 - 15*t**4*b**4) + 
            2*w**2*z**6*(24 + 4*w - 12*t**2*b**2 + 149*t**4*b**4) + 
            z**3*(64 + 48*t**2*b**2 - 496*t**4*b**4 + w*(48 - 36*t**2*b**2 - 33*t**4*b**4)) + 
            w*z**5*(64 + 48*t**2*b**2 - 560*t**4*b**4 - 3*w*(-16 + 12*t**2*b**2 + 7*t**4*b**4)) + 
            z**2*(8*w + 6*(8 - 4*t**2*b**2 + 51*t**4*b**4)) + 
            4*z**4*(3*w**2 + 8*(1 + 3*t**2*b**2 + 3*t**4*b**4) + 
            w*(24 - 12*t**2*b**2 + 113*t**4*b**4)))*np.log(w)**4))/(2*(1 + 2*z + w*z**2)**5*np.log(w)**4))               
        return d
    
    #Compute parity projected filling.
    def Parity(self, Site, mu=None, T=None, A=None, b=None):
        if A is None and b is None and mu is None and T is None:
            p = self.Filling(Site) - 2 * self.Doublons(Site)
        elif A is None and b is None:
            mu = self.mu
            T = self.T
            p = (self.Filling(Site, mu, T) - 2 * self.Doublons(Site, mu, T))
        else:
            p = A * (self.Filling(Site, mu, T) - 2 * self.Doublons(Site, mu, T)) + b
        return p
    
    