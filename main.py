import time
import  matplotlib.pyplot as plt
from constants import *
from data_handling import DataHandler
from model import FittingModel
from mcmc import MCMC
from plots import Plot
import pandas as pd
import os

import multiprocessing as mp
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)          # fix seed

# Create instances of each class
dh = DataHandler('../highRes_JADES_WIDEv3_flag_morph_new.csv')      # data handling
fm = FittingModel()         # fit model
mcmc = MCMC()           # mcmc handler
ph = Plot()         # plot handler

# timer, just out of curiosity
start_time = time.time()

# save edited file under 'filename.csv'
def save(filename):
    dh.table.to_csv('table_fitresults/'+str(filename)+'.csv', index=False)

# fits emission line region "line" for spectrum with index "i"
def fit(i,filename,line):
    np.random.seed(42) # fix seed
    # i - index/spectrum to be fitted
    # line - line to be fitted ['Ha','OIII','P','Pb']
    dh.read(i)  # Read data for object i
    dh.download()  # Download spectrum
    dh.param(dh.z_in)     # initial guesses for fitting parameters
    # create some more dictionaries where the output is depending on z
    # 1 component param
    par0_1 = {'Ha': dh.par0_Ha_1, 'OIII' : dh.par0_OIII_1, 'P' : dh.par0_P_1, 'Pb' : dh.par0_Pb_1}
    par_l_1 = {'Ha': dh.par_l_Ha_1, 'OIII' : dh.par_l_OIII_1, 'P' : dh.par_l_P_1, 'Pb' : dh.par_l_Pb_1}
    par_u_1 ={'Ha': dh.par_u_Ha_1, 'OIII' : dh.par_u_OIII_1, 'P' : dh.par_u_P_1, 'Pb' : dh.par_u_Pb_1}
    # AGN 2 component param
    par0_2 = {'Ha': dh.par0_Ha_2, 'OIII' : dh.par0_OIII_2, 'P' : dh.par0_P_2, 'Pb' : dh.par0_Pb_2}
    par_l_2 = {'Ha': dh.par_l_Ha_2, 'OIII' : dh.par_l_OIII_2, 'P' : dh.par_l_P_2, 'Pb' : dh.par_l_Pb_2}
    par_u_2 ={'Ha': dh.par_u_Ha_2, 'OIII' : dh.par_u_OIII_2, 'P' : dh.par_u_P_2, 'Pb' : dh.par_u_Pb_2}
    # outflow 2 component param
    par0_3 = {'Ha': dh.par0_Ha_3, 'OIII' : dh.par0_OIII_3}
    par_l_3 = {'Ha': dh.par_l_Ha_3, 'OIII' : dh.par_l_OIII_3}
    par_u_3 ={'Ha': dh.par_u_Ha_3, 'OIII' : dh.par_u_OIII_3}

    # test if redshift allows any lines to be observable in spectrum
    dh.test_z(lines[line][0],lines[line][-1],line=line)
    if dh.table[line][dh.a] == 1:
        pass
    else:
        wave,spec,err = dh.wrange(ll[line],lu[line])   #specify wl range that is going to be fitted
        # delete nans and check if there are some peak left and if nans are close to peak => appropriate flag
        dh.nans(wave,spec,err,dh.ind1(lines[line],wave,dh.z_in),line= line)
        dh.table.at[dh.a,line] = dh.flag
        # if there is at least one emission line covered by flux values (flag - [0,2]) => fit
        if dh.flag in [0,2]:
            print('fitting :',line)
            disp_lsf = dh.int_lsf(lines[line])
            wave2 = dh.wave_nan
            spec2 = dh.spec_nan
            err2 = dh.err_nan
            # scale up errors and remove 'bad' pixels
            v_step_1 = 1100 # km/s
            print(dh.z_in)
            step_1 = int((1+dh.z_in)*l_central[line]*v_step_1/(c*dh.dw))
            print(f'step: {step_1}')
            dh.rm_badpix(lines[line],wave2,spec2,err2,step=step_1)
            wave3, spec3, err3 = dh.wave_good, dh.spec_good, dh.err_good
            err4, su = dh.scale_up(lines[line],wave2,spec2,err2,step=step_1)
            print(f'scale up: {su}')
            dh.table.at[dh.a,'scale_up_'+line] = su

            # initial guess of fitparameters using scipy.optimize
            th0_1 = mcmc.optimize(par0_1[line],par_l_1[line],par_u_1[line],wave3,disp_lsf,spec3,err3,line=line,comp='1') # 1 comp
            th0_2 = mcmc.optimize(par0_2[line],par_l_2[line],par_u_2[line],wave3,disp_lsf,spec3,err3,line=line,comp='2') # AGN 2 comp
            # mcmc fit and remove burnin
            # flat samples
            par_l_1_new = np.append(par_l_1[line],0.8)
            par_u_1_new = np.append(par_u_1[line],1.5)
            par_l_2_new = np.append(par_l_2[line],0.8)
            par_u_2_new = np.append(par_u_2[line],1.5)
            th0_1_new = np.append(th0_1,1.1)
            th0_2_new = np.append(th0_2,1.1)


            fs_1 = mcmc.mcmc(th0_1_new,wave3,disp_lsf,spec3,err4,par_l_1_new,par_u_1_new,line=line,comp='1') # 1 comp
            fs_2 = mcmc.mcmc(th0_2_new,wave3,disp_lsf,spec3,err4,par_l_2_new,par_u_2_new,line=line,comp='2') # AGN 2 comp
            # store fitparameters
            th1_1, dth1_l_1, dth1_u_1 = mcmc.det_param(fs_1) # 1 comp
            th1_2, dth1_l_2, dth1_u_2 = mcmc.det_param(fs_2) # AGN 2 comp
            # give a bad fit flag depending on flux S/N
            # condition completely random, just looked at a couple spectra
            # maximum flux S/N of any of the peaks
            sn_flux_1 = np.max([th1_1[i]/np.max((dth1_l_1[i],dth1_u_1[i])) for i in range(2,i_amp[line])])
            print(f'testing flux condition 1 component: {sn_flux_1}')
            dh.table.at[dh.a,'sn_flux_1_'+line] = sn_flux_1
            if sn_flux_1 < 6:
                dh.table.at[dh.a,line] = 5
            bic_1 = mcmc.BIC(th1_1,wave3, disp_lsf,spec3,err4,line=line,comp='1')
            bic_2 = mcmc.BIC(th1_2,wave3, disp_lsf,spec3,err4,line=line,comp='2')
            # disp = th1_2[i-2], d_disp = th1_2[i_amp[line]+3], d_a[j] = th1_2[i+i_amp[line]+4], a[j] = a[i],  disp_lsf[j] = disp_lsf[i-2]
            sn_flux_2 = np.max([th1_2[i]/np.max((dth1_l_2[i],dth1_u_2[i])) for i in range(i_amp[line]+3,len(th1_2)-1)])
            print(f'testing flux condition AGN 2 component: {sn_flux_2}')

            dh.table.at[dh.a,'sn_flux_2_'+line] = sn_flux_2
            dh.table.at[dh.a,'bic_1_'+line] = bic_1
            dh.table.at[dh.a,'bic_2_'+line] = bic_2

            # outflow 2 comp fitting for Ha and OIII:
            sn_flux_3, bic_3 = 0, 0 # placeholders
            if line in ['Ha','OIII']:
                th0_3 = mcmc.optimize(par0_3[line],par_l_3[line],par_u_3[line],wave3,disp_lsf,spec3,err3,line=line,comp='3')
                par_l_3_new = np.append(par_l_3[line],0.8)
                par_u_3_new = np.append(par_u_3[line],1.5)
                th0_3_new = np.append(th0_3,1.1)
                fs_3 = mcmc.mcmc(th0_3_new,wave3,disp_lsf,spec3,err4,par_l_3_new,par_u_3_new,line=line,comp='3')
                th1_3, dth1_l_3, dth1_u_3 = mcmc.det_param(fs_3)
                sn_flux_3 = np.max([th1_3[i]/np.max((dth1_l_3[i],dth1_u_3[i])) for i in range(i_amp[line]+3,len(th1_3)-1)])
                print(f'testing flux condition outflow 2 component: {sn_flux_3}')
                bic_3 = mcmc.BIC(th1_3,wave3, disp_lsf,spec3,err4,line=line,comp='3')
                dh.table.at[dh.a,'sn_flux_3_'+line] = sn_flux_3
                dh.table.at[dh.a,'bic_3_'+line] = bic_3

            # look for the lowest value between BIC_1-10, BIC_2 and BIC_3
            # BIC_1 - one-component fit, BIC_2 - AGN two-component fit, BIC_3 - gas-flow two-component fit 
            sn_Ha_OIII = {'1':sn_flux_1,'2':sn_flux_2,'3':sn_flux_3}
            sn_P_Pb = {'1':sn_flux_1,'2':sn_flux_2}
            sn_l = {'Ha':sn_Ha_OIII,'OIII':sn_Ha_OIII,'P':sn_P_Pb,'Pb':sn_P_Pb}

            bic_Ha_OIII = {1:bic_1-10,2:bic_2,3:bic_3}
            bic_P_Pb = {1:bic_1-10,2:bic_2}
            bic_l = {'Ha':bic_Ha_OIII,'OIII':bic_Ha_OIII,'P':bic_P_Pb,'Pb':bic_P_Pb}

            key =  min(bic_l[line], key=bic_l[line].get) 
            flag_2c = {'Ha':6,'OIII':6,'P':8,'Pb':8}
            if key == 2 and sn_flux_2 > 3:
                dh.table.at[dh.a,line] = flag_2c[line]
            if key == 3 and sn_flux_3 > 3:
                dh.table.at[dh.a,line] = 7
            if dh.table[line][dh.a] in [0,2] and th1_1[0]>700:
                dh.table.at[dh.a,line] = 9

            # append nu_par to fitparameters
            par_1_new = np.append(par_1[line],f'nu_par1_{line}')
            dpar_l_1_new = np.append(dpar_l_1[line],f'nu_par1_l_{line}')
            dpar_u_1_new = np.append(dpar_u_1[line],f'nu_par1_u_{line}')
            par_2_new = np.append(par_2[line],f'nu_par2_{line}')
            dpar_l_2_new = np.append(dpar_l_2[line],f'nu_par2_l_{line}')
            dpar_u_2_new = np.append(dpar_u_2[line],f'nu_par2_u_{line}')
            # store fitparameters
            for i in range(len(par_1_new)): # for 1 component fit
                dh.table.at[dh.a,par_1_new[i]] = th1_1[i]
                dh.table.at[dh.a,dpar_l_1_new[i]] = dth1_l_1[i]
                dh.table.at[dh.a,dpar_u_1_new[i]] = dth1_u_1[i]
            for i in range(len(par_2_new)): # for 2 component fit
                dh.table.at[dh.a,par_2_new[i]] = th1_2[i]
                dh.table.at[dh.a,dpar_l_2_new[i]] = dth1_l_2[i]
                dh.table.at[dh.a,dpar_u_2_new[i]] = dth1_u_2[i]
            # calculate and store reduced chi squared
            chi2_r_1 = mcmc.red_chi2(th1_1,wave3,disp_lsf,spec3,err4,line=line,comp='1')
            chi2_r_2 = mcmc.red_chi2(th1_2,wave3,disp_lsf,spec3,err4,line=line,comp='2')
            dh.table.at[dh.a,'chi2_r_1_'+line] = chi2_r_1
            dh.table.at[dh.a,'chi2_r_2_'+line] = chi2_r_2
            # plot fit result and spectrum
            ph.plot_fit_1(wave3,disp_lsf*th1_1[-1],spec3,th1_1[:-1],str(dh.a)+'_fit_1_'+line,dh.name,dh.table[line][dh.a],line=line)
            ph.plot_corner(par_1[line],fs_1,str(dh.a)+'_corner_1_'+line,line=line)
            #ph.plot_fit_2(wave3,disp_lsf*1.1,spec3,th1_2[:-1],str(dh.a)+'_fit_2_'+line,dh.name,dh.table[line][dh.a],line=line)
            ph.plot_fit_2(wave3,disp_lsf*th1_2[-1],spec3,th1_2[:-1],str(dh.a)+'_fit_2_'+line,dh.name,dh.table[line][dh.a],line=line)
            ph.plot_corner(par_2[line],fs_2,str(dh.a)+'_corner_2_'+line,line=line)

            # outflow 2 comp for Ha and OIII:
            if line in ['Ha','OIII']:
                par_3_new = np.append(par_3[line],f'nu_par3_{line}')
                dpar_l_3_new = np.append(dpar_l_3[line],f'nu_par3_l_{line}')
                dpar_u_3_new = np.append(dpar_u_3[line],f'nu_par3_u_{line}')  
                for i in range(len(par_3_new)): # for 2 component fit
                    dh.table.at[dh.a,par_3_new[i]] = th1_3[i]
                    dh.table.at[dh.a,dpar_l_3_new[i]] = dth1_l_3[i]
                    dh.table.at[dh.a,dpar_u_3_new[i]] = dth1_u_3[i]  
                chi2_r_3 = mcmc.red_chi2(th1_3,wave3,disp_lsf,spec3,err4,line=line,comp='3')
                dh.table.at[dh.a,'chi2_r_3_'+line] = chi2_r_3
                ph.plot_fit_3(wave3,disp_lsf*th1_3[-1],spec3,th1_3[:-1],str(dh.a)+'_fit_3_'+line,dh.name,dh.table[line][dh.a],line=line)
                ph.plot_corner(par_3[line],fs_3,str(dh.a)+'_corner_3_'+line,line=line)


            # for AGN and out-/inflow contaminants the fits are additionally stored in a dedicated folder
            if dh.table[line][dh.a] in [6,7,8,9]:
                path_broad = f'png_broad/{int(dh.table[line][dh.a])}'
                ph.plot_fit_1(wave3,disp_lsf*th1_1[-1],spec3,th1_1[:-1],str(dh.a)+'_fit_1_'+line,dh.name,dh.table[line][dh.a],line=line,path=path_broad)
                ph.plot_corner(par_1[line],fs_1,str(dh.a)+'_corner_1_'+line,line=line,path=path_broad)
                ph.plot_fit_2(wave3,disp_lsf*th1_2[-1],spec3,th1_2[:-1],str(dh.a)+'_fit_2_'+line,dh.name,dh.table[line][dh.a],line=line,path=path_broad)
                ph.plot_corner(par_2[line],fs_2,str(dh.a)+'_corner_2_'+line,line=line,path=path_broad)
                if line in ['Ha','OIII']:
                    ph.plot_fit_3(wave3,disp_lsf*th1_3[-1],spec3,th1_3[:-1],str(dh.a)+'_fit_3_'+line,dh.name,dh.table[line][dh.a],line=line,path=path_broad)
                    ph.plot_corner(par_3[line],fs_3,str(dh.a)+'_corner_3_'+line,line=line,path=path_broad)


    print(f'flag: {dh.table[line][dh.a]}')
    # save fitresults and flags etc. in table
    save(filename)

# extend to multiple indices and specified emissoin line regions
def loop_fit(array,fn,flines):
    # array - array containing indices/spectra to be fitted
    # which lines to fit ['Ha','OIII','P','Pb']
    for i in array:
        print(i)
        for j in range(len(flines)):
            fit(i,fn,line=flines[j])
        #print('OIII-flag:',dh.table['OIII'][i],' Ha-flag:',dh.table['Ha'][i],' P-flag:',dh.table['P'][i],' Pb-flag:',dh.table['Pb'][i])
        # print time since code was started
        curr_time = time.time()
        time_passed = curr_time-start_time
        print(f'total time running: {time_passed:.1f} seconds')

# fits spectra using LSF determined for morphology in a specific photometric band: filt_lsf
def loop_fit_filt(ind, filt_lsf, fn, flines):
    print(filt_lsf)
    for i in ind:
        print(i)
        if os.path.exists(f'../LSF/LSF_result/{i}_{filt_lsf}_LSF.csv'):
            for j in range(len(flines)):
                fit(i,fn,line=flines[j])

file = pd.read_csv('../highRes_JADES_WIDEv3_flag_morph_new.csv')
# define indices of analyzed spectra and # cpu for multiprocessing 
file2 = file[file['flag_morph']==1]
list_ind = file2.index
n_cpu = 50


# use multiprocessing for emission line fitting
def fit_mp(ind,n_cpu):
    # split list of spectra up into chunks
    chunk_size = len(ind) // n_cpu
    chunks = [ind[i:i + chunk_size] for i in range(0, len(ind), max(1, chunk_size))]
    print(f'Chunks: {chunks}')
    #mp.set_start_method('spawn', force=True)  # Set start method to 'spawn'
    with Pool(n_cpu) as p:
        p.starmap(loop_fit, [(chunk,f'el_fit_{chunk[0]}-{chunk[-1]}',['Ha','OIII','P','Pb']) for chunk in chunks])

    # combine results from different workers/tables:
    file_comb = pd.read_csv('../highRes_JADES_WIDEv3_flag_morph.csv')
    for chunk in chunks:
        fn = f'table_fitresults/el_fit_{chunk[0]}-{chunk[-1]}.csv'
        f_chunk = pd.read_csv(fn)
        for i in chunk:
            for par in f_chunk.columns:
                file_comb.at[i, par] = f_chunk[par][i]
    file_comb.to_csv(f'table_fitresults/highRes_JADES_WIDEv3.7_fit_spec.csv', index=False)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    fit_mp(list_ind,n_cpu)

#fit(2417,'test2','Ha')
