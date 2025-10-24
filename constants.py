from scipy import constants
import numpy as np

# define constants

# -----------------------------
# Emission lines
# -----------------------------
# Ha region
l_Ha = 656.4614 * 1e-3  # um
l_NII_1 = 654.986 * 1e-3  # um
l_NII_2 = 658.527 * 1e-3  # um
l_SII_1 = 671.829*1e-3 # um
l_SII_2 = 673.268*1e-3 # um
lines_Ha = np.array([l_NII_1,l_Ha,l_NII_2,l_SII_1,l_SII_2]) 
labels_Ha = ['[NII]','H-aplha','[NII]','[SII]','[SII]']
colors_Ha = ['g','r','g','black','black']
wr_Ha = 0.002 # microns, usefull for plots

# OIII region
l_Hb = 486.268 * 1e-3  # um
l_OIII_1 = 496.0295 * 1e-3  # um
l_OIII_2 = 500.8240 * 1e-3  # um
lines_OIII = np.array([l_Hb,l_OIII_1,l_OIII_2])
labels_OIII = ['H-beta','[OIII]','[OIII]']
colors_OIII = ['r','g','g']
wr_OIII = 0.002 # microns, usefull for plots

# P region
l_Pg = 1.09411 # um Paschen-gamma   https://www.gemini.edu/observing/resources/near-ir-resources/spectroscopy/hydrogen-recombination-lines
l_Pd = 1.00521 # um  Paschen-delta
l_P = (l_Pg+l_Pd)/2 # um, wl in the middle of fitted Paschen region, could be usefull
l_HeI = 1.0832 # um
lines_P = np.array([l_Pd,l_HeI,l_Pg])
labels_P = ['P-delta','HeI','P-gamma']
colors_P = ['r','g','black']
wr_P = 0.002 # microns, usefull for plots

# Pb region
l_Pb = 1.28216 # um   https://www.gemini.edu/observing/resources/near-ir-resources/spectroscopy/hydrogen-recombination-lines
lines_Pb = np.array([l_Pb])
labels_Pb = ['P-beta']
colors_Pb = ['g']
wr_Pb = 0.009 # microns, usefull for plots

# speed of light
c = constants.c * 1e-3  # km/s
# Wavelength limits for dispersers, index 0 = g235h, index 1 = g395h
l_u = {'g235h': 3.1697029, 'g395h': 5.2396514}   # um
l_l = {'g235h': 1.66, 'g395h': 2.83}   # um

# -----------------------------
# Parameter labels for fitting routines
# -----------------------------
# Ha
# 1-component fit
par_Ha_1 = ['disp_Ha','z_Ha','a1_Ha','a2_Ha','a4_Ha','a5_Ha', 'm_Ha','d_Ha']
dpar_l_Ha_1 = ['disp_l_Ha','z_l_Ha','a1_l_Ha','a2_l_Ha','a4_l_Ha','a5_l_Ha','m_l_Ha','d_l_Ha',]
dpar_u_Ha_1 = ['disp_u_Ha','z_u_Ha','a1_u_Ha','a2_u_Ha','a4_u_Ha','a5_u_Ha','m_u_Ha','d_u_Ha']
# 2-component AGN fit
par_Ha_2 = ['disp_2_n_Ha','z_2_Ha','a1_2_n_Ha','a2_2_n_Ha','a4_2_n_Ha','a5_2_n_Ha', 'm_2_Ha','d_2_Ha','disp_2_b_Ha','a2_2_b_Ha']
dpar_l_Ha_2 = ['disp_2_n_l_Ha','z_2_l_Ha','a1_n_l_Ha','a2_n_l_Ha','a4_n_l_Ha','a5_n_l_Ha','m_2_l_Ha','d_2_l_Ha','disp_2_b_l_Ha','a2_b_l_Ha']
dpar_u_Ha_2 = ['disp_2_n_u_Ha','z_2_u_Ha','a1_n_u_Ha','a2_n_u_Ha','a4_n_u_Ha','a5_n_u_Ha','m_2_u_Ha','d_2_u_Ha','disp_2_b_u_Ha','a2_b_u_Ha']
# 2-component gas-flow fit
par_Ha_3 = ['disp_3_n_Ha','z_3_Ha','a1_3_n_Ha','a2_3_n_Ha','a4_3_n_Ha','a5_3_n_Ha', 'm_3_Ha','d_3_Ha','disp_3_b_Ha','a1_3_b_Ha',\
            'a2_3b_Ha','a4_3_b_Ha','a5_3_b_Ha']
dpar_l_Ha_3 = ['disp_3_n_l_Ha','z_3_l_Ha','a1_3_n_l_Ha','a2_3_n_l_Ha','a4_3_n_l_Ha','a5_3_n_l_Ha', 'm_3_l_Ha','d_3_l_Ha','disp_3_b_l_Ha','a1_3_b_l_Ha',\
            'a2_3_b_l_Ha','a4_3_b_l_Ha','a5_3_b_l_Ha']
dpar_u_Ha_3 = ['disp_3_n_u_Ha','z_3_u_Ha','a1_3_n_u_Ha','a2_3_n_u_Ha','a4_3_n_u_Ha','a5_3_n_u_Ha', 'm_3_u_Ha','d_3_u_Ha','disp_3_b_u_Ha','a1_3_b_u_Ha',\
            'a2_3_b_u_Ha','a4_3_b_u_Ha','a5_3_b_u_Ha']

# OIII
# 1-component fit
par_OIII_1 = ['disp_OIII','z_OIII','a1_OIII','a2_OIII', 'm_OIII','d_OIII']
dpar_l_OIII_1 = ['disp_l_OIII','z_l_OIII','a1_l_OIII','a2_l_OIII','m_l_OIII','d_l_OIII',]
dpar_u_OIII_1 = ['disp_u_OIII','z_u_OIII','a1_u_OIII','a2_u_OIII','m_u_OIII','d_u_OIII']
# 2-component AGN fit
par_OIII_2 = ['disp_2_n_OIII','z_2_OIII','a1_2_n_OIII','a2_2_n_OIII', 'm_2_OIII','d_2_OIII','disp_2_b_OIII','a1_2_b_OIII']
dpar_l_OIII_2 = ['disp_2_n_l_OIII','z_2_l_OIII','a1_2_n_l_OIII','a2_2_n_l_OIII','m_2_l_OIII','d_2_l_OIII','disp_2_b_l_OIII','a1_2_b_l_OIII']
dpar_u_OIII_2 = ['disp_2_n_u_OIII','z_2_u_OIII','a1_2_n_u_OIII','a2_2_n_u_OIII','m_2_u_OIII','d_2_u_OIII','disp_2_b_u_OIII','a1_2_b_u_OIII']
# 2-component gas-flow fit
par_OIII_3 = ['disp_3_n_OIII','z_3_OIII','a1_3_n_OIII','a2_3_n_OIII', 'm_3_OIII','d_3_OIII','disp_3_b_OIII','a1_3_b_OIII','a2_3_b_OIII']
dpar_l_OIII_3 = ['disp_3_n_l_OIII','z_3_l_OIII','a1_3_n_l_OIII','a2_3_n_l_OIII','m_3_l_OIII','d_3_l_OIII','disp_3_b_l_OIII','a1_3_b_l_OIII','a2_3_b_l_OIII']
dpar_u_OIII_3 = ['disp_3_n_u_OIII','z_3_u_OIII','a1_3_n_u_OIII','a2_3_n_u_OIII','m_3_u_OIII','d_3_u_OIII','disp_3_b_u_OIII','a1_3_b_u_OIII','a2_3_b_u_OIII']

# P
# 1-component fit
par_P_1 = ['disp_P','z_P','a1_P','a2_P','a3_P','m_P','d_P']
dpar_l_P_1 = ['disp_l_P','z_l_P','a1_l_P','a2_l_P','a3_l_P','m_l_P','d_l_P']
dpar_u_P_1 = ['disp_u_P','z_u_P','a1_u_P','a2_u_P','a3_u_P','m_u_P','d_u_P']
# 2-component fit
par_P_2 = ['disp_2_n_P','z_2_P','a1_n_P','a2_n_P','a3_n_P','m_2_P','d_2_P','disp_2_b_P','a1_b_P','a2_b_P','a3_b_P']
dpar_l_P_2 = ['disp_2_n_l_P','z_2_l_P','a1_n_l_P','a2_n_l_P','a3_n_l_P','m_2_l_P','d_2_l_P','disp_2_b_l_P','a1_b_l_P','a2_b_l_P','a3_b_l_P']
dpar_u_P_2 = ['disp_2_n_u_P','z_2_u_P','a1_n_u_P','a2_n_u_P','a3_n_u_P','m_2_u_P','d_2_u_P','disp_2_b_u_P','a1_b_u_P','a2_b_u_P','a3_b_u_P']

# Pb
# 1-component fit
par_Pb_1 = ['disp_Pb','z_Pb','a1_Pb','m_Pb','d_Pb']
dpar_l_Pb_1 = ['disp_l_Pb','z_l_Pb','a1_l_Pb','m_l_Pb','d_l_Pb']
dpar_u_Pb_1 = ['disp_u_Pb','z_u_Pb','a1_u_Pb','m_u_Pb','d_u_Pb']
# 2-component fit
par_Pb_2 = ['disp_2_n_Pb','z_2_Pb','a1_n_Pb','m_2_Pb','d_2_Pb','disp_2_b_Pb','a1_b_Pb']
dpar_l_Pb_2 = ['disp_2_n_l_Pb','z_2_l_Pb','a1_n_l_Pb','m_2_l_Pb','d_2_l_Pb','disp_2_b_l_Pb','a1_b_l_Pb']
dpar_u_Pb_2 = ['disp_2_n_u_Pb','z_2_u_Pb','a1_n_u_Pb','m_2_u_Pb','d_2_u_Pb','disp_2_b_u_Pb','a1_b_u_Pb']


# create dictionaries with output depending on line/wl-region

# arrays containing wavelengths
ll = {'Ha': l_NII_1, 'OIII' : l_Hb, 'P' : l_Pd, 'Pb' : l_Pb}
lu = {'Ha': l_NII_2, 'OIII' : l_OIII_2, 'P' : l_Pg, 'Pb' : l_Pb}
lines = {'Ha': lines_Ha, 'OIII' : lines_OIII, 'P' : lines_P, 'Pb' : lines_Pb}
l_central = {'Ha': l_Ha, 'OIII': l_OIII_1, 'P': l_P, 'Pb': l_Pb}
wr = {'Ha': wr_Ha, 'OIII': wr_OIII, 'P': wr_P, 'Pb': wr_Pb}
# labels
par_1 = {'Ha': par_Ha_1, 'OIII' : par_OIII_1, 'P' : par_P_1, 'Pb' : par_Pb_1}
dpar_l_1 = {'Ha': dpar_l_Ha_1, 'OIII' : dpar_l_OIII_1, 'P' : dpar_l_P_1, 'Pb' : dpar_l_Pb_1}
dpar_u_1 ={'Ha': dpar_u_Ha_1, 'OIII' : dpar_u_OIII_1, 'P' : dpar_u_P_1, 'Pb' : dpar_u_Pb_1}
par_2 = {'Ha': par_Ha_2, 'OIII' : par_OIII_2, 'P' : par_P_2, 'Pb' : par_Pb_2}
dpar_l_2 = {'Ha': dpar_l_Ha_2, 'OIII' : dpar_l_OIII_2, 'P' : dpar_l_P_2, 'Pb' : dpar_l_Pb_2}
dpar_u_2 ={'Ha': dpar_u_Ha_2, 'OIII' : dpar_u_OIII_2, 'P' : dpar_u_P_2, 'Pb' : dpar_u_Pb_2}
par_3 = {'Ha': par_Ha_3, 'OIII' : par_OIII_3}
dpar_l_3 = {'Ha': dpar_l_Ha_3, 'OIII' : dpar_l_OIII_3}
dpar_u_3 ={'Ha': dpar_u_Ha_3, 'OIII' : dpar_u_OIII_3}
labels = {'Ha': labels_Ha, 'OIII': labels_OIII, 'P': labels_P, 'Pb': labels_Pb}
colors = {'Ha': colors_Ha, 'OIII': colors_OIII, 'P': colors_P, 'Pb': colors_Pb}

# units:
# [disp] = km/s
# [z] = 
# [a] = 1e-19 * erg/s/cm^2
# [m] = 1e-14 * erg/s/cm^2/A^2
# [d] = 1e-19 * erg/s/cm^2/A

# photmetric filters of cutouts, morph. fits and LSF
all_filters = ['f090w-clear','f115w-clear','f150w-clear','f200w-clear','f444w-clear','f356w-clear','f277w-clear','f182m-clear','f210m-clear',\
           'f335m-clear','f410m-clear','f430m-clear','f460m-clear','f480m-clear']

test_filt = all_filters[0]

i_amp = {'OIII':4,'Ha':6,'P':5,'Pb':3}