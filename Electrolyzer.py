"""
Code name : HRI Alkaline Electrolysis cell model
Data : 2021.09.08

    HRI Electrolyser

    Operating information
    ---------------------
    Max. operating voltage : 48-56 [V]
    Operating Pressure : 1 [bar]
    Operating temperature : 0-353.15 [K]
    Hydrogen production rate : 1Nm**3/h at 80C
    Electrical power reference : 5kW at 60C

    Electrolyser's component
    ------------------------
    Membrane : Zirfon
    Anode : Nickel 99.99%
    Cathode : Nickel 99.99%
    Electrolyte : KOH at 30 wt.%
"""

import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

class HRI:
    def __init__(self, V,I,P=7,T=50):
        """
        Parameters
        ----------
        ncell : Number of electrolytic cells
        dac : Anode-Cathode gap [mm]
        dam/dcm : Membrane-Anote/Cathode gap [mm]
        Sm : Membrane surface area [m**2]
        em : Membrane thickness [mm]
        Sa/Sc : Anode/Cathode surface area [m**2]
        ea/ec : Anode/Cathode thickness [mm]
        La/Lc : Anode/Cathode height [cm]
        J0 : Anode/Cathode exchange current density [A/m**2]
        a : Anode/Cathode transfer coefficient
        R : The universal gas constant
        n : The number of electrons transferred in the electrolysis reaction
        F : The Faraday constant
        wt : KOH density in wt.%
        
        Variables
        ---------
        T : Temperature [K]
        P : Pressure [bar]
        I : Current [A]
        V : Terminal Voltage [V]
        """
        
        self.V = V
        self.I = I
        self.P = P
        self.T = T
        
        self.dam, self.dcm = 1.25, 1.25 #[mm]
        self.Sa, self.Sc, self.Sm = 0.03, 0.03, 0.03 #[m**2]
        #ea,ec, em = 2, 2, 0.5 #[mm]
        self.La, self.Lc = 45, 45 #[cm]
        self.R = 8.315 #[J/K/mol]
        self.n = 2
        self.F = 96485 #[C/mol]
        self.wt = 30 #[%]
        self.ncell = 24
        
        Eth = self.fEth(self.P, self.T)  # 가역전압
        Vact_a, Vact_c = self.fVact(self.P, self.T, self.I)  # 활성화 전압
        r = self.fR(self.I, self.T)  # 옴닉 저항
        Vcell = Eth + Vact_c + Vact_a + r * self.I  # 셀 전압 추정
        Vele = self.ncell * Vcell  # 수전해설비 전압 추정
        self.NH2_out = self.fNH2_out(self.I)
        
    ##### 입력 변수(T(온도), I(전류), P(압력))에 대한 종속 파라미터 정의
    def fm(self, T):
        """
        Parameters
        ----------
        m : molarity of KOH at 30 wt%
        wt : density of KOH in wt.%

        Varaibles
        ---------
        T : Temperature [C]
        """
        m = self.wt*(183.1221 - 0.56845*T + 984.5679*np.exp(self.wt/115.96277))/100/56.105
        return m

    def ftheta(self, I,S):
        """
        Parameters
        ----------
        S : Surface

        Variables
        _________
        I : Current [A]
        """
        theta = 0.023*(I/S)**0.3 # Caution! : I/A must be [A/m**2]
        return theta

    ##### 가역 전압 정의 (Reversible Voltage, Eth) #####
    def fEth(self, P,T):
        """
        Electrolytic cell model - Theoretical(Reversible) Voltage (Eth)

        Parameters
        ----------
        Erev0 : Temperature effect on reversible potential
        m : molarity of KOH at 30 wt%
        R : The universal gas constant
        n : The number of electrons transferred in the electrolysis reaction
        F : The Faraday constant
        ppw(pH2O*) : The partial pressure of pure water vapor
        pw(PH2O) : The electrolyte partial pressure (electrolyte having a molarity m)

        Variables
        ---------
        P : pressure [bar]
        T : temperature [C]
        """
        Erev0 = 1.50342 - 9.956*1e-4*T + 2.5*1e-7*T**2

        pw = T**(-3.498)*np.exp(37.93 - 6426.32/T)*np.exp(0.016214 - 0.13802*self.fm(T) + 0.19330*self.fm(T)**0.5)
        ppw = T**(-3.4159)*np.exp(37.043 - 6275.7/T)

        Erev = Erev0 + self.R*T*np.log((P-pw)**1.5*ppw/pw)/self.n/self.F + (P-pw)*(21.661*1e-6 - 5.471*1e-3/T) + \
            (P-pw)**2*(-6.289*1e-6/T + 0.135*1e-3/T**1.5 + 2.547*1e-3/T**2 - 0.4825/T**3)

        Eth = Erev0 - self.R*T*((P - pw)**(3/2)*ppw/pw)/self.n/self.F
        return Eth
    ##### 활성화 전압(Activation Voltage, Vact) 정의 #####
    def fVact(self, P,T,I):
        """
        Electrolytic cell model - Electrochemical electrodes' activation overvoltage (Vact)
        Assumes that the parameters are from HRI electrolyser.

        Parameters
        ----------
        R : The universal gas constant
        n : The number of electrons transferred in the electrolysis reaction
        F : The Faraday constant
        a : Transfer coefficient
        b : Tafel slope
        theta : Fractional electrode converage of the gas bubbles adhering on the electrodes.
        S : The nominal electrode surface in cm**2
        Seff : The effective electrode surface in cm**2
        J : Current density of the electrodes
        J0 : Exchange current densities of the electrodes

        Variables
        ---------
        P : Pressure[bar]
        T : Temperature [C]
        I : Current [A]
        """
        aa = 0.0675 + 0.00095*T
        ac = 0.1175 + 0.00095*T

        ba = 2.303*self.R*T/self.n/self.F/aa
        bc = 2.303*self.R*T/self.n/self.F/ac

        theta_a, theta_c = self.ftheta(I,self.Sa), self.ftheta(I,self.Sc) # Caution! : I/A must be [A/m**2]
        Seff_a, Seff_c = self.Sa*(1-theta_a), self.Sc*(1-theta_c)

        Ja = I/Seff_a #[mA/cm**2]
        Jc = I/Seff_c #[mA/cm**2]
        J0a = 30.4 - 0.206*T + 0.00035*T**2 #[mA/cm**2]
        J0c = 13.72491 - 0.09055*T + 0.09055*T**2 #[[mA/cm**2]]

        Vact_a = ba*np.log10(Ja/J0a) + ba*np.log10(1-theta_a)
        Vact_c = bc*np.log10(Jc/J0c) + bc*np.log10(1-theta_c)
        return Vact_a, Vact_c
    ##### 옴닉 저항 (Ohmmic Resistance, r) 정의 #####
    def fR(self, I, T):
        """
        Electrolytic cell model - Electrical resistance phenomenon

        Parameters
        ----------
        omega_Ni : The electrical conductivity of the Ni [S/cm]
        omega_KOH : The ionic conductivity of the KOH [S/cm]
        e : The new ionic conductivity of the electrolyte as the result of the bubbles
        L : The electrode thickness [cm]
        S : The electrode cross-sections [m**2]
        d : distance between the electrodes and the membranes [mm]
        m : molarity of KOH at 30 wt%
        Ra : The anode resistance [ohm]
        Rc : The cathode resistance [ohm]
        Rele : Electrolyte's resistance [ohm]
        Rmem : The membrane resistance [ohm]

        Variables
        ---------
        T : Temperature [K]
        """
        # Electrodes
        omega_Ni = 6*1e6 - 279650*T + 532*T**2 - 0.38057*T**3 #[S/cm]
        Ra = self.La/omega_Ni/self.Sa*1e-4 #[ohm]
        Rc = self.Lc/omega_Ni/self.Sc*1e-4 #[ohm]
        # Electrolyte
        omega_KOH = -2.04*self.fm(T) - 0.0028*(self.fm(T))**2 + 0.005332*self.fm(T)*T + 207.2*self.fm(T)/T + \
            0.001043*(self.fm(T))**3 - 0.0000003*(self.fm(T))**2*T**2
        e = 2/3*self.ftheta(I,self.Sa)
        Rele_free = 1/omega_KOH*(self.dam/self.Sa+self.dcm/self.Sc)*1e-5 #[ohm]
        Rele_e = Rele_free*(1/(1-e)**(3/2)-1) #[ohm]
        Rele = Rele_free + Rele_e #[ohm]
        # Membrane
        Rmem = (0.060 + 80*np.exp(T/50))/1e8/self.Sm # 0.5mm thickness of membrane

        r = Ra + Rc + Rele + Rmem
        return r
    
    def fNH2_out(self, I):
        """
        parameters
        __________
        n : The number of electrons transferred in the electrolysis reaction
        """
        I_loss = 0
        nF = (I-I_loss)/I
        H2_out = self.ncell*I*nF/self.n/self.F
        return H2_out


if __name__=="__main__" :
    def WED(V,I,P,T,state = 'on'):
        if state == 'on':
            Mode = HRI(V,I,P,T)
            output = Mode.NH2_out
        else:
            output = 0
        return output

    # Example
    filepath_mac = '/Users/jeonseungchan/OneDrive/OneDrive - 한양대학교/3. Codes/1. DataSet/5. KETI_Example_Data'
    filepath = 'C:/Users/jsc95/OneDrive - 한양대학교/3. Codes/1. DataSet/3. 2021_KIEE_Data'
    filename = ['V_operate','I_operate','P_operate','T_operate']
    filepath = filepath_mac # MAC 사용 시 ON

    df_V = pd.read_csv(os.path.join(filepath, "%s.csv" %filename[0]))
    df_I = pd.read_csv(os.path.join(filepath, "%s.csv" %filename[1]))
    df_P = pd.read_csv(os.path.join(filepath, "%s.csv" %filename[2]))
    df_T = pd.read_csv(os.path.join(filepath, "%s.csv" %filename[3]))

    V = df_V.values
    I = df_I.values
    P = df_P.values
    T = df_T.values

    Hydrogen = WED(V,I,P,T,state='on')