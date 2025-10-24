import matplotlib.pyplot as plt
from model import FittingModel
from data_handling import DataHandler
from constants import *
import corner

fm = FittingModel()
dh = DataHandler('../highRes_JADES_WIDEv3.csv')
path = 'png'  # path for plots


class Plot:
    def __init__(self):
        pass

    # plot fitresult and fitted spectrum for 1 component fit

    # Ha, OIII, Pb regions
    def plot_fit1_1(self, wave, disp_lsf, spec, th, fn, name, flag, line, path=path):
        # line in ['Ha','OIII','Pb']
        ind1 = dh.ind(lines[line][0],wave,th[1])
        ind2 = dh.ind(lines[line][-1],wave,th[1])
        bt = np.max((np.min(spec),-0.5))
        xlim = [[wave[0], wave[-1]], [np.max((wave[0],(lines[line][0]-wr[line])*(1+th[1]))), np.min((wave[-1],(lines[line][-1]+wr[line])*(1+th[1])))]]
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.2)
        name_clear = name.replace('.spec.fits', '')
        title = [f'{name_clear} (z = {th[1]:.3}; flag = {flag})','zoom in']
        for i in range(2):
            ax = axes[i]
            ax.set_title(title[i])
            ax.step(wave, spec, label='data')
            ax.step(wave, fm.fitfct_1(th,wave,disp_lsf,line), label='fit')
            for j in range(len(lines[line])):
                ax.axvline(lines[line][j]*(th[1]+1),lw=1,ls='--',color = colors[line][j],label=labels[line][j])
            ax.set_xlabel(r'observed wavelength [$\mu$m]', size=10)
            ax.set_ylabel(r'$\rm f_\lambda \ \rm [10^{-19}\, erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]$', size=10)
            ax.set_xlim(xlim[i])
            ax.set_ylim(bottom=bt)
            ax.legend()
        plt.savefig(f'{path}/{fn}.png')
        plt.close()

    # P region
    def plot_fit1_2(self, wave, disp_lsf, spec, th, fn, name, flag, line, path=path):
        if line == 'P':
            lines = lines_P
            lines2 = [lines,[lines_P[0]],[lines_P[1],lines_P[2]]]
            labels = labels_P
            labels2 = [labels,[labels[0]],[labels[1],labels[2]]]
            colors = colors_P
            colors2 = [colors,[colors[0]],[colors[1],colors[2]]]
        ind1 = dh.ind(lines[0],wave,th[1])
        ind2 = dh.ind(lines[1],wave,th[1])
        ind3 = dh.ind(lines[-1],wave,th[1])
        bt = np.max((np.min(spec),-0.5))
        xlim = [[wave[0],wave[-1]],[np.max((wave[0],(lines[0]-0.009)*(1+th[1]))), np.min((wave[-1],(lines[0]+0.009)*(1+th[1])))],\
        [np.max((wave[0],(lines[1]-0.003)*(1+th[1]))), np.min((wave[-1],(lines[2]+0.003)*(1+th[1])))]]
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        fig.subplots_adjust(hspace=0.25)
        name_clear = name.replace('.spec.fits', '')
        title = [f'{name_clear} (z = {th[1]:.3}; flag = {flag})','zoom in P-delta', 'zoom in [HeI] and P-gamma']
        for i in range(3):
            ax = axes[i]
            ax.set_title(title[i])
            ax.step(wave, spec, label='data')
            ax.step(wave, fm.fitfct_1(th,wave,disp_lsf,line), label='fit')
            for j in range(len(lines2[i])):
                ax.axvline(lines2[i][j]*(th[1]+1),lw=1,ls='--',color = colors2[i][j],label=labels2[i][j])   
            ax.set_xlabel(r'observed wavelength [$\mu$m]', size=10)
            ax.set_ylabel(r'$\rm f_\lambda \ \rm [10^{-19}\, erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]$', size=10)
            ax.set_xlim(xlim[i])
            ax.set_ylim(bottom=bt)
            ax.legend()
        #plt.savefig(f'png/{fn}.png')
        plt.savefig(f'{path}/{fn}.png')
        plt.close()

    def plot_fit_1(self,wave,disp_lsf,spec,th,fn,name,flag,line, path=path):
        if line == 'P':
            self.plot_fit1_2(wave,disp_lsf,spec,th,fn,name,flag,line, path=path)
        else:
            self.plot_fit1_1(wave,disp_lsf,spec,th,fn,name,flag,line, path=path)

#--------------------------------------------------------------------------------------------------------------------------------------------#

    # plot fitresult and fitted spectrum for AGN 2 component fit

    # Ha, OIII, Pb regions
    def plot_fit2_1(self, wave, disp_lsf, spec, th, fn, name, flag, line, path=path):
        # line in ['Ha','OIII','Pb']
        ind1 = dh.ind(lines[line][0],wave,th[1])
        ind2 = dh.ind(lines[line][-1],wave,th[1])
        bt = np.max((np.min(spec),-0.5))
        xlim = [[wave[0], wave[-1]], [np.max((wave[0],(lines[line][0]-wr[line])*(1+th[1]))), np.min((wave[-1],(lines[line][-1]+wr[line])*(1+th[1])))]]
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.2)
        name_clear = name.replace('.spec.fits', '')
        title = [f'{name_clear} (z = {th[1]:.3}; flag = {flag})','zoom in']
        for i in range(2):
            ax = axes[i]
            ax.set_title(title[i])
            ax.step(wave, spec, label='data')
            ax.step(wave, fm.fitfct_2_n(th,wave,disp_lsf,line), color='C2', label='narrow')
            ax.step(wave, fm.fitfct_2_b(th,wave,disp_lsf,line), color='black', label='broad')
            ax.step(wave, fm.fitfct_2(th,wave,disp_lsf,line), color='C1', label='narrow+broad')
            for j in range(len(lines[line])):
                ax.axvline(lines[line][j]*(th[1]+1),lw=1,ls='--',color = colors[line][j],label=labels[line][j])
            ax.set_xlabel(r'observed wavelength [$\mu$m]', size=10)
            ax.set_ylabel(r'$\rm f_\lambda \ \rm [10^{-19}\, erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]$', size=10)
            ax.set_xlim(xlim[i])
            ax.set_ylim(bottom=bt)
            ax.legend()
        plt.savefig(f'{path}/{fn}.png')
        plt.close()

    # P region
    def plot_fit2_2(self, wave, disp_lsf, spec, th, fn, name, flag, line, path=path):
        if line == 'P':
            lines = lines_P
            lines2 = [lines,[lines_P[0]],[lines_P[1],lines_P[2]]]
            labels = labels_P
            labels2 = [labels,[labels[0]],[labels[1],labels[2]]]
            colors = colors_P
            colors2 = [colors,[colors[0]],[colors[1],colors[2]]]
        ind1 = dh.ind(lines[0],wave,th[1])
        ind2 = dh.ind(lines[1],wave,th[1])
        ind3 = dh.ind(lines[-1],wave,th[1])
        bt = np.max((np.min(spec),-0.5))
        xlim = [[wave[0],wave[-1]],[np.max((wave[0],(lines[0]-0.009)*(1+th[1]))), np.min((wave[-1],(lines[0]+0.009)*(1+th[1])))],\
        [np.max((wave[0],(lines[1]-0.003)*(1+th[1]))), np.min((wave[-1],(lines[2]+0.003)*(1+th[1])))]]
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        fig.subplots_adjust(hspace=0.25)
        name_clear = name.replace('.spec.fits', '')
        title = [f'{name_clear} (z = {th[1]:.3}; flag = {flag})','zoom in P-delta', 'zoom in [HeI] and P-gamma']
        for i in range(3):
            ax = axes[i]
            ax.set_title(title[i])
            ax.step(wave, spec, label='data')
            ax.step(wave, fm.fitfct_2_n(th,wave,disp_lsf,line), color='C2', label='narrow')
            ax.step(wave, fm.fitfct_2_b(th,wave,disp_lsf,line), color='black', label='broad')
            ax.step(wave, fm.fitfct_2(th,wave,disp_lsf,line), color='C1', label='narrow+broad')
            for j in range(len(lines2[i])):
                ax.axvline(lines2[i][j]*(th[1]+1),lw=1,ls='--',color = colors2[i][j],label=labels2[i][j])   
            ax.set_xlabel(r'observed wavelength [$\mu$m]', size=10)
            ax.set_ylabel(r'$\rm f_\lambda \ \rm [10^{-19}\, erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]$', size=10)
            ax.set_xlim(xlim[i])
            ax.set_ylim(bottom=bt)
            ax.legend()
        plt.savefig(f'{path}/{fn}.png')
        plt.close()

    def plot_fit_2(self,wave,disp_lsf,spec,th,fn,name,flag,line, path=path):
        if line == 'P':
            self.plot_fit2_2(wave,disp_lsf,spec,th,fn,name,flag,line, path=path)
        else:
            self.plot_fit2_1(wave,disp_lsf,spec,th,fn,name,flag,line, path=path)

#-----------------------------------------------------------------------------------------------------------------------------------#

    # plot fitresult and fitted spectrum for AGN 2 component fit

    # Ha and OIII regions
    def plot_fit_3(self, wave, disp_lsf, spec, th, fn, name, flag, line, path=path):
        # line in ['Ha','OIII']
        ind1 = dh.ind(lines[line][0],wave,th[1])
        ind2 = dh.ind(lines[line][-1],wave,th[1])
        bt = np.max((np.min(spec),-0.5))
        xlim = [[wave[0], wave[-1]], [np.max((wave[0],(lines[line][0]-wr[line])*(1+th[1]))), np.min((wave[-1],(lines[line][-1]+wr[line])*(1+th[1])))]]
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.2)
        name_clear = name.replace('.spec.fits', '')
        title = [f'{name_clear} (z = {th[1]:.3}; flag = {flag})','zoom in']
        for i in range(2):
            ax = axes[i]
            ax.set_title(title[i])
            ax.step(wave, spec, label='data')
            ax.step(wave, fm.fitfct_3_n(th,wave,disp_lsf,line), color='C2', label='narrow')
            ax.step(wave, fm.fitfct_3_b(th,wave,disp_lsf,line), color='black', label='broad')
            ax.step(wave, fm.fitfct_3(th,wave,disp_lsf,line), color='C1', label='narrow+broad')
            for j in range(len(lines[line])):
                ax.axvline(lines[line][j]*(th[1]+1),lw=1,ls='--',color = colors[line][j],label=labels[line][j])
            ax.set_xlabel(r'observed wavelength [$\mu$m]', size=10)
            ax.set_ylabel(r'$\rm f_\lambda \ \rm [10^{-19}\, erg\,s^{-1}\,cm^{-2}\,\AA^{-1}]$', size=10)
            ax.set_xlim(xlim[i])
            ax.set_ylim(bottom=bt)
            ax.legend()
        #plt.savefig(f'png/{fn}.png')
        plt.savefig(f'{path}/{fn}.png')
        plt.close()

#-----------------------------------------------------------------------------------------------------------------------------#

    # plot samples
    def plot_samples(self):
        pass

    # corner plot fitparams
    def plot_corner(self,lab,flat_samples,fn,line, path=path):
        labels = lab + ['nuissance par']
        fig = corner.corner(flat_samples, labels=labels)
        #plt.savefig(f'png/{fn}.png')
        plt.savefig(f'{path}/{fn}.png')
        plt.close()
