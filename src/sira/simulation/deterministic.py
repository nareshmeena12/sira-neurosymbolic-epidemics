"""
Deterministic ODE Models for SIRA.
Provides the continuous-time baselines for SIR, SEIR, and SIRS dynamics.
"""
import numpy as np

class DeterministicEpidemic:
    """
    A collection of static methods representing the continuous vector fields 
    of various compartmental epidemic models. 
    These are formatted specifically for scipy.integrate.odeint.
    """
    @staticmethod
    def sir(y,t,N,beta,gamma):
        """
        Standard SIR Model.
        y: [S, I, R]
        """
        S,I,R=y
        dSdt=-beta*S*I/N
        dIdt=beta*S*I/N-gamma*I
        dRdt=gamma*I
        return [dSdt,dIdt,dRdt]
    
    @staticmethod
    def seir(y,t,N,beta,gamma,omega):
        """
        SIRS Model (Waning Immunity / Endemic Loop).
        y: [S, I, R]
        """
        S, I, R = y
        
        dSdt = -(beta * S * I) / N + omega * R
        dIdt = (beta * S * I) / N - gamma * I
        dRdt = gamma * I - omega * R
        
        return [dSdt, dIdt, dRdt]
    @staticmethod
    def sirs(y, t, N, beta, gamma, omega):
        """
        SIRS Model (Waning Immunity / Endemic Loop).
        y: [S, I, R]
        """
        S, I, R = y
        
        dSdt = -(beta * S * I) / N + omega * R   
        dIdt = (beta * S * I) / N - gamma * I
        dRdt = gamma * I - omega * R             
        
        return [dSdt, dIdt, dRdt]
        
    @staticmethod
    def time_varying_sir(y, t, N, beta_func, gamma):
        """
        Time-Varying SIR Model (Lockdown/Intervention).
        beta_func: A callable function of time, e.g., beta(t)
        """
        S, I, R = y
        
        current_beta = beta_func(t)
        
        dSdt = -(current_beta * S * I) / N
        dIdt = (current_beta * S * I) / N - gamma * I
        dRdt = gamma * I
        
        return [dSdt, dIdt, dRdt]