import emcee
import numpy as np
from scipy.optimize import minimize
from model import FittingModel
import scipy.stats as stats

fm = FittingModel()
np.random.seed(42)

fitfct = {'1': fm.fitfct_1, '2': fm.fitfct_2, '3': fm.fitfct_3}

class MCMC:
    def __init__(self, nwalkers=32, nsteps=2500, burnin=1000):
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burnin = burnin

    # Maximum likelihood
    def lnlike(self, theta, x, disp_lsf, y, yerr, line, comp):
        nu_par = theta[-1]
        model_theta = theta[:-1]
        disp_lsf_n = disp_lsf * nu_par
        model = fitfct[comp](model_theta, x, disp_lsf_n, line)
        return -0.5 * np.nansum(((y - model) / yerr)**2 + np.log(2 * np.pi * yerr**2))

    # Initial optimization
    def optimize(self, par_0, par_l, par_u, wave, disp_lsf, spec, err, line, comp):
        bnds = [(par_l[i]+0.001, par_u[i]-0.001) for i in range(len(par_0))]
        def nll(theta): return -self.lnlike(np.append(theta, 1.1), wave, disp_lsf, spec, err, line, comp)
        soln = minimize(nll, par_0, method='L-BFGS-B', bounds=bnds)
        return soln.x

    # Prior with Gaussian prior on nu_par
    def lnprior(self, theta, par_l, par_u):
        model_theta = theta[:-1]
        nu_par = theta[-1]
        if all(par_l[i] <= theta[i] <= par_u[i] for i in range(len(theta))):
            return stats.norm.logpdf(nu_par, loc=1.1, scale=0.2)
        return -np.inf

    # Posterior probability function
    def lnprob(self, theta, x, disp_lsf, y, yerr, line, par_l, par_u, comp):
        lp = self.lnprior(theta, par_l, par_u)
        if not np.isfinite(lp):
            return -np.inf, None
        lnlike = self.lnlike(theta, x, disp_lsf, y, yerr, line, comp)
        return lp + lnlike, theta[-1]

    # Initial ball of walkers
    def make_ini_ball(self, th, par_l, par_u, e=0.1):
        ball = np.zeros((len(th), self.nwalkers))
        for i, ival in enumerate(th):
            scale = min(abs(ival - par_l[i]), abs(ival - par_u[i])) * 0.1 #* 0.1
            ball[i] = np.random.normal(loc=ival, scale=abs(e * scale), size=self.nwalkers)
        return ball.T

    # Run MCMC
    def mcmc(self, th0, wave, disp_lsf, spec, err, par_l, par_u, line, comp):
        
        pos = self.make_ini_ball(th0, par_l, par_u, e=0.1)
        self.nwalkers, self.ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.lnprob,
            args=(wave, disp_lsf, spec, err, line, par_l, par_u, comp))
        sampler.run_mcmc(pos, self.nsteps)

        flat_samples = sampler.get_chain(discard=self.burnin, thin=1, flat=True)
        return flat_samples

    # Extract fit results
    def det_param(self, flat_samples):
        th1 = []
        dth1_l = []
        dth1_u = []
        for i in range(flat_samples.shape[1]):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            th1.append(mcmc[1])
            dth1_l.append(q[0])
            dth1_u.append(q[1])
        return th1, dth1_l, dth1_u

    # Fit quality
    def chi2(self, theta, x, disp_lsf, y, yerr, line, comp):
        nu_par = theta[-1]
        model_theta = theta[:-1]
        disp_lsf_n = disp_lsf * nu_par
        return np.nansum(((y - fitfct[comp](model_theta, x, disp_lsf_n, line)) / yerr)**2)

    def red_chi2(self, theta, x, disp_lsf, y, yerr, line, comp):
        return self.chi2(theta, x, disp_lsf, y, yerr, line, comp) / (len(x) - len(theta) + 1)

    def BIC(self, theta, x, disp_lsf, y, yerr, line, comp):
        x2 = self.chi2(theta, x, disp_lsf, y, yerr, line, comp)
        k = len(theta)
        n = len(x)
        return k * np.log(n) + x2