# Emission Line Fitting Pipeline for JWST NIRSpec High-Resolution Spectra  
    
This Python pipeline performs emission line fitting for high-resolution galactic spectra from JWST NIRSpec. It provides multiple fitting strategies, each including a linear continuum as well as Gaussian emission line profiles. Distinct two-component Gaussian fits enable the identification of AGN and out-/inflows.

---

#### Emission Line Regions Supported:   

OIII: Hβ λ 4863, [OIII] λ 4960, [OIII] λ 5008  
Ha: [NII] λ 6550, Hα λ 6565, [NII] λ 6585, [SII] λ 6718, [SII] λ 6733   
P: Pδ λ 10052, HeI λ 10832, Pγ λ 10941  
Pb: Pβ λ 12822  
  
#### Features:

- Single-component Gaussian fits (narrow lines) for all regions
- AGN 2-component fits (narrow + broad components) for all regions
- Outflow 2-component fits (narrow + broad components) for Ha and OIII region
- MCMC-based parameter estimation for uncertainties
- Plotting of fits and corner plots
- Automatic data preprocessing, including LSF handling

---
  
## File Structure:
```
emission_line_fits/
├── constants.py        # Physical constants, line wavelengths, labels and plotting info
├── data_handling.py    # Read and preprocess spectra, interpolate LSF
├── model.py            # Gaussian line models for fitting
├── mcmc.py             # MCMC routines for fit parameter estimation
├── plots.py            # Functions to plot spectra, fits, and corner plots
├── main.py             # Main script for batch fitting using multiprocessing
├── png/                # Output folder for plots
├── png_broad/          # Output folder for plots of AGN and gas-flow contaminants
├── table_fitresults/   # Output CSV tables of fit results
├── LICENSE             # License for project usage
└── README.md           # Project documentation
```

---  

## Dependencies:
  
Python 3.8+ is required. Install the following packages:
  
```bash
pip install numpy scipy matplotlib pandas emcee corner astropy



