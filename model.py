import numpy as np
from scipy import special
from constants import *



class FittingModel:
    def __init__(self):
        pass
    
    # define integrated gaussian evaluated at x2 and x1
    def erf(self, a, b, c, x1, x2):
        return a / 2 * (special.erf((x2 - b) / (c * np.sqrt(2))) - special.erf((x1 - b) / (c * np.sqrt(2))))


    # define integrated linear continuum evaluated at x2 and x1
    def lin(self, mu, m, d, x1, x2):
        return m / 2 * (x2**2 - x1**2) * 1e+3 + (d - m * mu) * (x2 - x1) * 1e+4 # [d] = 1e-19 erg/s/cm^2/A, [m] = 1e-14 erg/s/cm^2/A^2
 

    # -----------------------------
    # Ha region models
    # -----------------------------
    # 1-component fit function
    def model_Ha_1(self,theta,x1,x2,disp_lsf):
        disp,z,a1,a2,a4,a5,m,d = theta
        mu = (1+z) * l_Ha
        #combine LSF with true broadening:
        disp_tot = np.sqrt(disp_lsf**2+disp**2)
        dl = disp_tot  * (1+z) * lines_Ha / c               # convert [km/s] into [microns]
        b = (1+z)*lines_Ha                                  # assumed same redshift for all lines
        #a3 = 3.05*a1                                                             # fixed line ratio for NII doublet 
        a3 = 2.95*a1
        amp = [a1,a2,a3,a4,a5]
        model = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp)):
            model+= self.erf(amp[i],b[i],dl[i],x1,x2)
        return model

    # 2-component AGN fit function
    def model_Ha_2(self,theta,x1,x2,disp_lsf):
        disp_n,z,a1_n,a2_n,a4_n,a5_n,m,d,disp_b,a2_b = theta
        mu = (1+z) * l_Ha
        #combine LSF with true broadening:
        disp_tot_n = np.sqrt(disp_lsf**2+disp_n**2)
        disp_tot_b = np.sqrt(disp_lsf**2+(disp_n+disp_b)**2)
        dl_n = disp_tot_n  * (1+z) * lines_Ha / c               # convert [km/s] into [microns]
        dl_b = disp_tot_b  * (1+z) * lines_Ha[1] / c
        b_n = (1+z)*lines_Ha                                  # assumed same redshift for all lines
        b_b = [(1+z)*lines_Ha[1]]
        #a3_n = 3.05*a1_n                                                             # fixed line ratio for NII doublet 
        a3_n = 2.95*a1_n
        amp_n = [a1_n,a2_n,a3_n,a4_n,a5_n]
        amp_b = [a2_b]
        model = self.lin(mu,m,d,x1,x2)
        model_n = self.lin(mu,m,d,x1,x2)
        model_b = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp_n)):
            model+= self.erf(amp_n[i],b_n[i],dl_n[i],x1,x2)
            model_n+= self.erf(amp_n[i],b_n[i],dl_n[i],x1,x2)
        for i in range(len(amp_b)):
            model+= self.erf(amp_b[i],b_b[i],dl_b[i],x1,x2)
            model_b+= self.erf(amp_b[i],b_b[i],dl_b[i],x1,x2)
        return model, model_n, model_b

    # 2-component gas-flow fit-function
    def model_Ha_3(self,theta,x1,x2,disp_lsf):
        disp_n,z,a1_n,a2_n,a4_n,a5_n,m,d,disp_b,a1_b,a2_b,a4_b,a5_b = theta
        mu = (1+z) * l_Ha
        #combine LSF with true broadening:
        disp_tot_n = np.sqrt(disp_lsf**2+disp_n**2)
        disp_tot_b = np.sqrt(disp_lsf**2+(disp_n+disp_b)**2)
        dl_n = disp_tot_n  * (1+z) * lines_Ha / c               # convert [km/s] into [microns]
        dl_b = disp_tot_b  * (1+z) * lines_Ha / c
        b = (1+z)*lines_Ha                                  # assumed same redshift for all lines
        #a3_n = 3.05*a1_n                                                             # fixed line ratio for NII doublet 
        #a3_b = 3.05*a1_b
        a3_n = 2.95*a1_n                                                             # fixed line ratio for NII doublet 
        a3_b = 2.95*a1_b
        amp_n = [a1_n,a2_n,a3_n,a4_n,a5_n]
        amp_b = [a1_b,a2_b,a3_b,a4_b,a5_b]
        model = self.lin(mu,m,d,x1,x2)
        model_n = self.lin(mu,m,d,x1,x2)
        model_b = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp_n)):
            model+= self.erf(amp_n[i],b[i],dl_n[i],x1,x2)
            model+= self.erf(amp_b[i],b[i],dl_b[i],x1,x2)
            model_n+= self.erf(amp_n[i],b[i],dl_n[i],x1,x2)
            model_b+= self.erf(amp_b[i],b[i],dl_b[i],x1,x2)
        return model, model_n, model_b
        
    # -----------------------------
    # Ha region models
    # -----------------------------
    # 1-component fit function
    def model_OIII_1(self,theta,x1,x2,disp_lsf):
        disp,z,a1,a2,m,d = theta
        mu = (1+z) * l_OIII_1
        #combine LSF with true broadening:
        disp_tot = np.sqrt(disp_lsf**2+disp**2)
        dl = disp_tot  * (1+z) * lines_OIII / c               # convert [km/s] into [microns]
        b = (1+z)*lines_OIII                                  # assumed same redshift for all lines
        a3 = 2.98*a2                                                            # fixed line ratio for OIII doublet 
        amp = [a1,a2,a3]
        model = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp)):
            model+= self.erf(amp[i],b[i],dl[i],x1,x2)
        return model

    # 2-component AGN fit function
    def model_OIII_2(self,theta,x1,x2,disp_lsf):
        disp_n,z,a1_n,a2_n,m,d,disp_b,a1_b = theta
        mu = (1+z) * l_OIII_1
        #combine LSF with true broadening:
        disp_tot_n = np.sqrt(disp_lsf**2+disp_n**2)
        disp_tot_b = np.sqrt(disp_lsf**2+(disp_n+disp_b)**2)
        dl_n = disp_tot_n  * (1+z) * lines_OIII / c               # convert [km/s] into [microns]
        dl_b = disp_tot_b  * (1+z) * lines_OIII[0] / c
        b_n = (1+z)*lines_OIII                                  # assumed same redshift for all lines
        b_b = [(1+z)*lines_OIII[0]]
        a3_n = 2.98*a2_n                                                            # fixed line ratio for OIII doublet 
        amp_n = [a1_n,a2_n,a3_n]
        amp_b = [a1_b]
        model = self.lin(mu,m,d,x1,x2)
        model_n = self.lin(mu,m,d,x1,x2)
        model_b = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp_n)):
            model+= self.erf(amp_n[i],b_n[i],dl_n[i],x1,x2)
            model_n+= self.erf(amp_n[i],b_n[i],dl_n[i],x1,x2)
        for i in range(len(amp_b)):
            model+= self.erf(amp_b[i],b_b[i],dl_b[i],x1,x2)
            model_b+= self.erf(amp_b[i],b_b[i],dl_b[i],x1,x2)
        return model, model_n, model_b

    # 2-component gas-flow fit-function
    def model_OIII_3(self,theta,x1,x2,disp_lsf):
        disp_n,z,a1_n,a2_n,m,d,disp_b,a1_b,a2_b = theta
        mu = (1+z) * l_OIII_1
        #combine LSF with true broadening:
        disp_tot_n = np.sqrt(disp_lsf**2+disp_n**2)
        disp_tot_b = np.sqrt(disp_lsf**2+(disp_n+disp_b)**2)
        dl_n = disp_tot_n  * (1+z) * lines_OIII / c               # convert [km/s] into [microns]
        dl_b = disp_tot_b  * (1+z) * lines_OIII / c
        b = (1+z)*lines_OIII                                  # assumed same redshift for all lines
        a3_n = 2.98*a2_n                                                            # fixed line ratio for OIII doublet 
        a3_b = 2.98*a2_b
        amp_n = [a1_n,a2_n,a3_n]
        amp_b = [a1_b,a2_b,a3_b]
        model = self.lin(mu,m,d,x1,x2)
        model_n = self.lin(mu,m,d,x1,x2)
        model_b = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp_n)):
            model+= self.erf(amp_n[i],b[i],dl_n[i],x1,x2)
            model+= self.erf(amp_b[i],b[i],dl_b[i],x1,x2)
            model_n+= self.erf(amp_n[i],b[i],dl_n[i],x1,x2)
            model_b+= self.erf(amp_b[i],b[i],dl_b[i],x1,x2)
        return model, model_n, model_b


    # -----------------------------
    # P region models
    # -----------------------------
    # 1-component fit function
    def model_P_1(self,theta,x1,x2,disp_lsf):
        disp,z,a1,a2,a3,m,d = theta
        mu = (1+z) * l_P
        #combine LSF with true broadening:
        disp_tot = np.sqrt(disp_lsf**2+disp**2)
        dl = disp_tot  * (1+z) * lines_P / c               # convert [km/s] into [microns]
        b = (1+z)*lines_P                                  # assumed same redshift for all lines
        amp = [a1,a2,a3]
        model = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp)):
            model+= self.erf(amp[i],b[i],dl[i],x1,x2)
        return model

    # 2-component fit function
    def model_P_2(self,theta,x1,x2,disp_lsf):
        disp_n,z,a1_n,a2_n,a3_n,m,d,disp_b,a1_b,a2_b,a3_b = theta
        mu = (1+z) * l_P
        #combine LSF with true broadening:
        disp_tot_n = np.sqrt(disp_lsf**2+disp_n**2)
        disp_tot_b = np.sqrt(disp_lsf**2+(disp_n+disp_b)**2)
        dl_n = disp_tot_n  * (1+z) * lines_P / c               # convert [km/s] into [microns]
        dl_b = disp_tot_b  * (1+z) * lines_P / c
        b = (1+z)*lines_P                                  # assumed same redshift for all lines
        amp_n = [a1_n,a2_n,a3_n]
        amp_b = [a1_b,a2_b,a3_b]
        model = self.lin(mu,m,d,x1,x2)
        model_n = self.lin(mu,m,d,x1,x2)
        model_b = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp_n)):
            model+= self.erf(amp_n[i],b[i],dl_n[i],x1,x2)
            model+= self.erf(amp_b[i],b[i],dl_b[i],x1,x2)
            model_n+= self.erf(amp_n[i],b[i],dl_n[i],x1,x2)
            model_b+= self.erf(amp_b[i],b[i],dl_b[i],x1,x2)
        return model, model_n, model_b


    # -----------------------------
    # Pb region models
    # -----------------------------
    # 1-component fit function
    def model_Pb_1(self,theta,x1,x2,disp_lsf):
        disp,z,a1,m,d = theta
        mu = (1+z) * l_Pb
        #combine LSF with true broadening:
        disp_tot = np.sqrt(disp_lsf**2+disp**2)
        dl = disp_tot  * (1+z) * lines_Pb / c               # convert [km/s] into [microns]
        b = (1+z)*lines_Pb                                  # assumed same redshift for all lines
        amp = [a1]
        model = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp)):
            model+= self.erf(amp[i],b[i],dl[i],x1,x2)
        return model

    # 2-component fit function
    def model_Pb_2(self,theta,x1,x2,disp_lsf):
        disp_n,z,a1_n,m,d,disp_b,a1_b = theta
        mu = (1+z) * l_Pb
        #combine LSF with true broadening:
        disp_tot_n = np.sqrt(disp_lsf**2+disp_n**2)
        disp_tot_b = np.sqrt(disp_lsf**2+(disp_n+disp_b)**2)
        dl_n = disp_tot_n  * (1+z) * lines_Pb / c               # convert [km/s] into [microns]
        dl_b = disp_tot_b  * (1+z) * lines_Pb / c
        b = (1+z)*lines_Pb                                  # assumed same redshift for all lines
        amp_n = [a1_n]
        amp_b = [a1_b]
        model = self.lin(mu,m,d,x1,x2)
        model_n = self.lin(mu,m,d,x1,x2)
        model_b = self.lin(mu,m,d,x1,x2)
        for i in range(len(amp_n)):
            model+= self.erf(amp_n[i],b[i],dl_n[i],x1,x2)
            model+= self.erf(amp_b[i],b[i],dl_b[i],x1,x2)
            model_n+= self.erf(amp_n[i],b[i],dl_n[i],x1,x2)
            model_b+= self.erf(amp_b[i],b[i],dl_b[i],x1,x2)
        return model, model_n, model_b

    # -----------------------------
    # combined fit functions for any line
    # -----------------------------

    # 1-component fit function 
    def fitfct_1(self,theta,wave,disp_lsf,line):
        dw = wave[1] - wave[0]
        if line == 'Ha':
            disp,z,a1,a2,a4,a5,m,d = theta
            fct = self.model_Ha_1(theta,wave-dw/2,wave+dw/2,disp_lsf)
        if line == 'OIII':
            disp,z,a1,a2,m,d = theta
            fct = self.model_OIII_1(theta,wave-dw/2,wave+dw/2,disp_lsf)
        if line == 'P':
            disp,z,a1,a2,a3,m,d = theta
            fct = self.model_P_1(theta,wave-dw/2,wave+dw/2,disp_lsf)
        if line == 'Pb':
            disp,z,a1,m,d = theta
            fct = self.model_Pb_1(theta, wave-dw/2,wave+dw/2,disp_lsf)
        return fct

    # 2-component AGN fit function 
    def fitfct_2(self,theta,wave,disp_lsf,line):
        dw = wave[1] - wave[0]
        if line == 'Ha':
            disp_n,z,a1,a2,a4,a5,m,d,disp_b,d_a2 = theta
            fct = self.model_Ha_2(theta,wave-dw/2,wave+dw/2,disp_lsf)[0]
        if line == 'OIII':
            disp_n,z,a1,a2,m,d,d_disp,d_a1 = theta
            fct = self.model_OIII_2(theta,wave-dw/2,wave+dw/2,disp_lsf)[0]
        if line == 'P':
            disp_n,z,a1,a2,a3,m,d,disp_b,d_a1,d_a2,d_a3 = theta
            fct = self.model_P_2(theta,wave-dw/2,wave+dw/2,disp_lsf)[0]
        if line == 'Pb':
            disp_n,z,a1,m,d,disp_b,d_a1 = theta
            fct = self.model_Pb_2(theta, wave-dw/2,wave+dw/2,disp_lsf)[0]
        return fct

    # 2-component AGN fit function , only narrow
    def fitfct_2_n(self,theta,wave,disp_lsf,line):
        dw = wave[1] - wave[0]
        if line == 'Ha':
            disp_n,z,a1,a2,a4,a5,m,d,disp_b,d_a2 = theta
            fct = self.model_Ha_2(theta,wave-dw/2,wave+dw/2,disp_lsf)[1]
        if line == 'OIII':
            disp_n,z,a1,a2,m,d,d_disp,d_a1 = theta
            fct = self.model_OIII_2(theta,wave-dw/2,wave+dw/2,disp_lsf)[1]
        if line == 'P':
            disp_n,z,a1,a2,a3,m,d,disp_b,d_a1,d_a2,d_a3 = theta
            fct = self.model_P_2(theta,wave-dw/2,wave+dw/2,disp_lsf)[1]
        if line == 'Pb':
            disp_n,z,a1,m,d,disp_b,d_a1 = theta
            fct = self.model_Pb_2(theta, wave-dw/2,wave+dw/2,disp_lsf)[1]
        return fct

    # 2-component AGN fit function , only broad
    def fitfct_2_b(self,theta,wave,disp_lsf,line):
        dw = wave[1] - wave[0]
        if line == 'Ha':
            disp_n,z,a1,a2,a4,a5,m,d,disp_b,d_a2 = theta
            fct = self.model_Ha_2(theta,wave-dw/2,wave+dw/2,disp_lsf)[2]
        if line == 'OIII':
            disp_n,z,a1,a2,m,d,d_disp,d_a1 = theta
            fct = self.model_OIII_2(theta,wave-dw/2,wave+dw/2,disp_lsf)[2]
        if line == 'P':
            disp_n,z,a1,a2,a3,m,d,disp_b,d_a1,d_a2,d_a3 = theta
            fct = self.model_P_2(theta,wave-dw/2,wave+dw/2,disp_lsf)[2]
        if line == 'Pb':
            disp_n,z,a1,m,d,disp_b,d_a1 = theta
            fct = self.model_Pb_2(theta, wave-dw/2,wave+dw/2,disp_lsf)[2]
        return fct

    # 2-component gas-flow fit function  
    def fitfct_3(self,theta,wave,disp_lsf,line):
        dw = wave[1] - wave[0]
        if line == 'Ha':
            disp_n,z,a1,a2,a4,a5,m,d,disp_b,d_a1,d_a2,d_a4,d_a5 = theta
            fct = self.model_Ha_3(theta,wave-dw/2,wave+dw/2,disp_lsf)[0]
        if line == 'OIII':
            disp_n,z,a1,a2,m,d,disp_b,d_a1,d_a2 = theta
            fct = self.model_OIII_3(theta,wave-dw/2,wave+dw/2,disp_lsf)[0]
        return fct

    # 2-component gas-flow fit function, only narrow
    def fitfct_3_n(self,theta,wave,disp_lsf,line):
        dw = wave[1] - wave[0]
        if line == 'Ha':
            disp_n,z,a1,a2,a4,a5,m,d,disp_b,d_a1,d_a2,d_a4,d_a5 = theta
            fct = self.model_Ha_3(theta,wave-dw/2,wave+dw/2,disp_lsf)[1]
        if line == 'OIII':
            disp_n,z,a1,a2,m,d,disp_b,d_a1,d_a2 = theta
            fct = self.model_OIII_3(theta,wave-dw/2,wave+dw/2,disp_lsf)[1]
        return fct

    # 2-component gas-flow fit function, only broad
    def fitfct_3_b(self,theta,wave,disp_lsf,line):
        dw = wave[1] - wave[0]
        if line == 'Ha':
            disp_n,z,a1,a2,a4,a5,m,d,disp_b,d_a1,d_a2,d_a4,d_a5 = theta
            fct = self.model_Ha_3(theta,wave-dw/2,wave+dw/2,disp_lsf)[2]
        if line == 'OIII':
            disp_n,z,a1,a2,m,d,disp_b,d_a1,d_a2 = theta
            fct = self.model_OIII_3(theta,wave-dw/2,wave+dw/2,disp_lsf)[2]
        return fct