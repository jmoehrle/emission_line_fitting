import pandas as pd
from astropy.io import fits
import astropy.units as u
from constants import *
from scipy import interpolate
import scipy.stats as stats

class DataHandler:
    def __init__(self, file_path):
        self.table = pd.read_csv(file_path)

    # read table for parameters of spectrum
    def read(self,a):
        self.a = a
        self.oid = self.table['srcid'][a]
        self.ra = self.table['ra'][a]
        self.dec = self.table['dec'][a]
        self.name = self.table['file'][a]
        self.root = self.table['root'][a]
        self.z_in = self.table['z'][a]
        self.disp = self.table['disp'][a]

    # download spectrum
    def download(self):
        base_url = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/'
        url = base_url+self.root+'/'+self.name
        hdu = fits.open(url)
        wave_0 = hdu[1].data['wave'] * u.um
        spec_0 = hdu[1].data['flux'] * u.uJy
        err_0 = hdu[1].data['err']  * u.uJy
        self.spec = spec_0.to(u.erg/u.s/u.cm**2/u.AA,equivalencies=u.spectral_density(wave_0)).value * 1e19 # [erg/s/cm^2/A * 1e-19]
        self.err = err_0.to(u.erg/u.s/u.cm**2/u.AA,equivalencies=u.spectral_density(wave_0)).value * 1e19   # [erg/s/cm^2/A * 1e-19]
        self.wave = wave_0.value
        self.dw = self.wave[1]-self.wave[0]  # wl values equidistant

    # define parameters, initial guesses for their values and their boundaries
    def param(self,z):
        # 1 component parameters
        # disp, z, a1, a2, ... , m, d, n_par
        self.par0_Ha_1 = 50,z,1,15,1,1,0,0
        self.par_l_Ha_1 = np.array([0,z-0.01,0,0,0,0,-100,-100])
        self.par_u_Ha_1 = np.array([2*1e+3,z+0.01,400,800,400,400,100,100])
        self.par0_OIII_1 = 50,z,25,15,0,0
        self.par_l_OIII_1 = np.array([0, z-0.01, 0, 0, -100, -100])
        self.par_u_OIII_1 = np.array([2*1e+3, z+0.01, 800, 800, 100, 100])
        self.par0_P_1 = 50,z,1,10,2,0,0
        self.par_l_P_1 = np.array([0,z-0.01,0,0,0,-100,-100])
        self.par_u_P_1 = np.array([2*1e+3,z+0.01,400,800,400,100,100])
        self.par0_Pb_1 = 50,z,10,0,0
        self.par_l_Pb_1 = np.array([0,z-0.01,0,-100,-100])
        self.par_u_Pb_1 = np.array([2*1e+3,z+0.01,800,100,100])
        # AGN 2 component parameters
        # disp_n, z, a1, a2, ... , m, d, disp_b, d_a1, d_a2, ..., nu_par
        self.par0_Ha_2 = 50,z,1,15,1,1,0,0,200,8
        self.par_l_Ha_2 = np.array([0,z-0.01,0,0,0,0,-100,-100,40,0])
        self.par_u_Ha_2 = np.array([2*1e+3,z+0.01,400,800,400,400,100,100,2*1e+3,800])
        self.par0_OIII_2 = 50,z,25,15,0,0,200,13
        self.par_l_OIII_2 = np.array([0,z-0.01,0,0,-100,-100,40,0])
        self.par_u_OIII_2 = np.array([2*1e+3, z+0.01, 800, 800, 100, 100,2*1e+3,800])
        self.par0_P_2 = 50,z,1,10,2,0,0,200,0.2,0.5,5
        self.par_l_P_2 = np.array([0,z-0.01,0,0,0,-100,-100,40,0,0,0])
        self.par_u_P_2 = np.array([2*1e+3,z+0.01,400,800,400,100,100,2*1e+3,400,800,400])
        self.par0_Pb_2 = 50,z,10,0,0,200,5
        self.par_l_Pb_2 = np.array([0,z-0.01,0,-100,-100,40,0])
        self.par_u_Pb_2 = np.array([2*1e+3,z+0.01,800,100,100,2*1e+3,800])
        # outflow 2 component parameters
        # disp_n, z, a1, a2, ... , m, d, disp_b, d_a1, d_a2, ..., nu_par
        self.par0_Ha_3 = 50,z,1,15,1,1,0,0,200,0.5,8,0.5,0.5
        self.par_l_Ha_3 = np.array([0,z-0.01,0,0,0,0,-100,-100,40,0,0,0,0])
        self.par_u_Ha_3 = np.array([2*1e+3,z+0.01,400,800,400,400,100,100,500,400,800,400,400])
        self.par0_OIII_3 = 50,z,25,15,0,0,200,13,8
        self.par_l_OIII_3 = np.array([0,z-0.01,0,0,-100,-100,40,0,0])
        self.par_u_OIII_3 = np.array([2*1e+3, z+0.01, 800, 800, 100, 100,500,800,800])

    # test z if at least 1 of the lines could be in spectrum	
    def test_z(self,l1,l2,line):
        if l_l[self.disp] <= l1*(self.z_in+1) <= l_u[self.disp] or l_l[self.disp] <= l1*(self.z_in+1) <= l_u[self.disp]:
            tz = 0
        else:
            tz = 1
        self.table.at[self.a,line] = tz

    # determine index of certain RF wl 'l'
    def ind(self,l,wave,z):
        dw = wave[1]-wave[0]
        return int((l*(1+z)-wave[0])/dw)+1

    def ind1(self,l_array,wave,z):
        ind = []
        for i in range(len(l_array)):
            ind.append(self.ind(l_array[i],wave,z))
        return ind

    # once nans are deleted ind1 won't work consistently
    def ind2(self,l,wave2,z):
        try:
        # Find the index where the wavelength in self.wave matches the wavelength in wave2
            result = np.where(wave2 == self.wave[self.ind(l, self.wave, z)])[0][0]
            return result
        except IndexError:
            return []

    # function that determines the indices for an array of RF wavelengths
    def ind3(self,l_array,wave2,z):
        indices = []
        for i in range(len(l_array)):
            indices.append(self.ind2(l_array[i],wave2,z))
        indices2 = [n for n in indices if isinstance(n, np.int64)]
        return indices2
    
    # function that determines indices in range 'step' from l_array 
    def ind4(self,l_array,wave2,z,step):
        indices = self.ind3(l_array,wave2,z)
        indices2 = np.concatenate([[n-i,n+i] for n in indices for i in range(step)])  # all indices within stepsize of peak
        indices3 = np.sort(np.unique(indices2))  # remove duplicates and sort
        indices4 = list(set(indices3) & set(list(range(len(wave2)))))    # make sure index exists
        return indices4

    # restrict wl range: -0.03 um of low wl peak up to +0.03 um of high wl peak in RF
    def bound(self,l1,l2):
        bound1_th = self.ind(l1-0.03,self.wave,self.z_in)
        bound2_th = self.ind(l2+0.03,self.wave,self.z_in)
        # make sure index exists
        bound1 = np.max([bound1_th,0])
        bound2 = np.min([bound2_th,len(self.wave)])
        return bound1,bound2

    # define wl range that is going to get fitted
    def wrange(self,l1,l2):
        b1,b2 = self.bound(l1,l2)
        return self.wave[b1:b2], self.spec[b1:b2], self.err[b1:b2]

    # nans ...
    def nans(self,wave,spec,err,ind,line):
        # find indices of all nans in spec
        nan_ind1 = []
        for i in range(len(wave)):
            if np.isnan(spec[i]) == True:
                nan_ind1.append(i)
        # if there are no nans => arrays unchanged and flag = 0 
        if len(nan_ind1) == 0:
            self.flag = 0
            self.wave_nan = wave
            self.spec_nan = spec
            self.err_nan = err
        # if spec contains only nans => flag = 3
        elif len(nan_ind1) == len(wave):
            self.flag = 3
        else:
            # delete pixels adjacent to nans, they can be bugged
            adj = [np.array([n-1, n, n+1]) for n in nan_ind1]
            nan_ind2 = np.unique(np.concatenate(adj))
            nan_ind3 = [x for x in nan_ind2 if x >= 0 and x<=len(wave)-1]  # delete invalid indices
            self.wave_nan = np.delete(wave,nan_ind3)
            self.spec_nan = np.delete(spec,nan_ind3)
            self.err_nan = np.delete(err,nan_ind3)
            # if remaining spectrum contains no peaks => flag = 4
            if len(self.ind3(lines[line],self.wave_nan,self.z_in))==0:
                self.flag = 4
            else:
                dwl = []
                # check if there are nans within 1e+3 km/s around one of the peaks
                for i in range(len(ind)):
                    dwl.append(min(abs(ind[i] - value) for value in nan_ind3)) 
                v_max = 2000 # km/s
                if min(dwl)*self.dw <= (self.z_in+1)*l_central[line]/c * v_max:
                    self.flag = 2
                else:
                    self.flag = 0


    # scale up errors if necessary

    # removes peaks +/- step from wave, spec and err
    def rm_peaks(self,lines,wave,spec,err,step):
        peaks = self.ind4(lines,wave,self.z_in,step)
        wave_rm = np.delete(wave,peaks) 
        spec_rm = np.delete(spec,peaks) 
        err_rm = np.delete(err,peaks) 
        return wave_rm, spec_rm, err_rm

    # mask 'bad' pixels, continuum values that deviate strongly
    def rm_badpix(self,lines,wave,spec,err,step): 
        wave_rm, spec_rm, err_rm = self.rm_peaks(lines,wave,spec,err,step) # spectrum without peak areas
        if len(wave_rm) == 0:
            wave_gp, spec_gp, err_gp = wave_rm, spec_rm, err_rm
            self.wave_good = self.wave_nan                            # define new arrays where bad pixels are removed
            self.spec_good = self.spec_nan
            self.err_good = self.err_nan
        else:
            std = np.std(spec_rm)                                              # std of continuum flux
            med = np.median(spec_rm)
            self.med_spec = med                                           # median of errors in continuum area
            mask = np.abs(spec_rm-med) > 4 * std                              # mask values, where the flux deviates too much
            wave_bp = wave_rm[mask]                                            # bad wavelength values
            #if spec_rm[mask]>0:
                #print(f'sigma deviation bp: {np.abs(spec_rm[mask][0]-med)/std}')
            #print(f'bad wavelengths: {wave_bp}')
            ind_bp = [np.where(self.wave_nan == w_bp)[0][0] for w_bp in wave_bp]        # indeces of bad wl values in _nan arrays
            self.wave_good = np.delete(self.wave_nan,ind_bp)                            # define new arrays where bad pixels are removed
            self.spec_good = np.delete(self.spec_nan,ind_bp)
            self.err_good = np.delete(self.err_nan,ind_bp)
            wave_gp, spec_gp, err_gp = wave_rm[~mask], spec_rm[~mask], err_rm[~mask]    # arrays without peaks and wthout bad pixels
        return wave_gp, spec_gp, err_gp

    # calculates median of err without peak regions
    def error_med(self,lines,wave,spec,err,step): 
        return np.median(self.rm_badpix(lines,wave,spec,err,step)[2])
        #return np.median(self.rm_peaks(lines,wave,spec,err,step)[2])

    # calculates std of spec without peak regions
    def std(self,lines,wave,spec,err,step):
        return np.std(self.rm_badpix(lines,wave,spec,err,step)[1])
        #return np.std(self.rm_peaks(lines,wave,spec,err,step)[1])

    # calculates upscaling of err
    def scale_up(self,lines,wave,spec,err,step):
        if len(self.rm_badpix(lines,wave,spec,err,step)[2]) > 0 :
            ratio = self.std(lines,wave,spec,err,step)/self.error_med(lines,wave,spec,err,step)
            self.std_spec = self.std(lines,wave,spec,err,step)
        else:
            ratio = 1
            self.std_spec = np.median(err)
        upscaling = np.min((np.max((1,ratio)),5)) # capped at 5 just to make sure
        err_new = self.err_good * upscaling
        return err_new, upscaling

    # read lsf and interpolate it for relevant wl
    def int_lsf(self,lines):
        lsf_root = '../LSF/LSF_result3/'
        filt_lsf = self.table['filt_LSF'][self.a]
        lsf_name = f'{self.a}_{filt_lsf}_LSF.csv'
        #lsf_name = {'g235h':'LSF_files/LSF_disp_g235h.csv', 'g395h':'LSF_files/LSF_disp_g395h.csv'}
        #lsf = pd.read_csv(lsf_name[self.disp])
        lsf = pd.read_csv(lsf_root+lsf_name)
        wave_lsf = lsf['wave']
        disp_lsf = lsf['disp_kms']
        f = interpolate.interp1d(wave_lsf, disp_lsf, kind='linear', bounds_error=False, fill_value="extrapolate")
        disp_lsf_lines = np.array(f(lines*(1+self.z_in)))
        self.disp_lsf_lines = disp_lsf_lines
        return  disp_lsf_lines