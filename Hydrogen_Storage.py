"""
Code name : Compressor-Hydrogen Tank-Hydrogen Vehicles model
Data : 2021.09.08
"""

import numpy as np
import pandas as pd
import os
import random

class H2System:
    """
    Parameters
    ----------
    Pout_H2 : Compressor Exit Hydrogen Pressure [Pa]
    Tout_H2 : Compressor Exit Hydrogen Temperature [K]
    R : Gas Constant
    R1 : Gas Constant
    """
    def __init__(self,NH2_ele, NH2_HV, P_H2_ele, T_H2_ele):
        self.NH2_ele = NH2_ele
        self.NH2_HV = NH2_HV
        self.P_H2_ele = P_H2_ele
        self.T_H2_ele = T_H2_ele
        
        self.R = 8.3144 #[J/K/mol]
        self.R1 = 0.08266 #[atm*liter/mol/K]
        
        P_H2_comp, T_H2_comp = self.fCompressor(self.NH2_ele, self.P_H2_ele, self.T_H2_ele)
        self.H2 = self.fH2Tank(self.NH2_ele, self.NH2_HV, T_H2_comp)
        
    def fCompressor(self, NH2_ele, P_H2_ele, T_H2_ele):
        """
        Parameters
        ----------
        n_comp : compressor efficiency
        
        Time_Varying
        ------------
        m : Polytropic exponent
        '''
        0이면 등압과정
        1이면 등온과정
        r이면 등엔트로피과정
        infinite이면 등적과정
        '''
        Input
        -----
        NH2_ele : Electrolyzer Hydrogen flow rate [mol/s]
        P_H2_ele : Electrolyzer Hydrogen Pressure [Pa]
        T_H2_ele : Electrolyzer Hydrogen Temperature [K]
        
        Output
        ------
        P_H2_comp :Compressor Hydrogen Pressure [Pa]
        T_H2_comp : Compressor Hydrogen Temperature [K]
        """
        n_comp = 1
        P_H2_comp = 200 * 101325 #[Pa = atm * 101325]
        T_H2_comp = 25 + 273.15 #[K]
        m = np.log(P_H2_ele/P_H2_ele)/np.log(P_H2_ele*T_H2_comp/P_H2_comp/T_H2_ele)
        Pcomp = 2*NH2_ele*m*self.R*T_H2_ele/(m-1)/n_comp*((P_H2_comp/(P_H2_ele*P_H2_comp)**(1/2))**((m-1)/m)-1)
        return P_H2_comp, T_H2_comp
    
    def fH2Tank(self,NH2_ele, NH2_HV, T_H2_comp):
        """
        Parameters
        ----------
        Vtank : Hydrogen Tank Volume [liter]
        T_H2_comp : Compressor Hydrogen Temperature [K]
        t : Time Interval [s]
        A0, B0, a, b, c : Volume-mol parameters
        
        Time_Varying
        ------------
        Ptank : Tank Pressure [atm] 
        
        Input
        -----
        NH2_ele : Electrolyzer Hydrogen flow rate [mol/s]
        NH2_HV : Hydrogen Vehicles Hydrogen flow rate [mol/s]
        
        Output
        ------
        n : Number of molse of H2 in the tank [mole]
        """
        Vtank = 20 * 1e3 #[liter = m^3 * 1e3]
        A0 = 0.1975 # [atm*liter^2/mol^2] = a4
        B0 = 0.02096 # [liter/mol] = a2
        a = -0.00506 #[liter/mol] = a5
        b = -0.04359 #[liter/mol] = a3
        c = 0.0504*1e4 # [liter*K^3/mol] = a1
        t = 1 #[s]
        n = NH2_ele * t - NH2_HV * t # [mol]
        Ptank = n**2*self.R1*T_H2_comp/Vtank**2*(1-c*n/Vtank/T_H2_comp**3)*(Vtank/n+B0*(1-b*n/Vtank)) \
        - A0*(1-a*n/Vtank)*n**2/Vtank**2 # [atm]
        return n
    
def Hydrogen_Storage_System(NH2_ele, NH2_HV, P_H2_ele, T_H2_ele):
    Mode = H2System(NH2_ele, NH2_HV, P_H2_ele, T_H2_ele)
    output = Mode.H2
    return output


if __name__=="__main__" : 
    # Example
    filepath_mac = '/Users/jeonseungchan/OneDrive/OneDrive - 한양대학교/3. Codes/1. DataSet/5. KETI_Example_Data'
    filepath = 'C:/Users/jsc95/OneDrive - 한양대학교/3. Codes/1. DataSet/3. 2021_KIEE_Data'
    filename = ['Hydrogen','Vehicle','P_operate','T_operate']
    filepath = filepath_mac # MAC 사용 시 ON

    df_Hydrogen = pd.read_csv(os.path.join(filepath, "%s.csv" %filename[0]))
    df_Vehicle = pd.read_csv(os.path.join(filepath, "%s.csv" %filename[1]))
    df_P = pd.read_csv(os.path.join(filepath, "%s.csv" %filename[2]))
    df_T = pd.read_csv(os.path.join(filepath, "%s.csv" %filename[3]))

    Hydrogen = df_Hydrogen.values
    Vehicle = df_Vehicle.values
    P = df_P.values
    T = df_T.values

    H2 = Hydrogen_Storage_System(Hydrogen,Vehicle,P,T)