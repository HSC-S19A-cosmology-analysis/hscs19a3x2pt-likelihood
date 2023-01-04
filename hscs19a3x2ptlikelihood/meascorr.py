import numpy as np
import astropy.cosmology
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RegularGridInterpolator
import time

from astropy import constants as const
c = const.c.to('Mpc/s').value
G = const.G.to('Mpc^3/s^2kg^1').value
M_sun = const.M_sun.to('g').value
sigcrit_prefactor = c**2/(4.*np.pi*G)*10**3/M_sun/10**12

class dSigma_meascorr_class:
    def __init__(self, config):
        """
        Args:
          config (dict) : Following keys must be included
            - fname_sumwlssigcritinvPz: a file name saving sumwlssigcritinvPz.
            - fname_zsbin: a file name saving the photometric redshift bin of source galaxies
            - Om   : Omega_m used for measurement
            - wde  : Equation of state of dark energy used for measurement
        
        https://arxiv.org/abs/2111.10966
        """
        self.config = config
        self.load_sumwlssigcritinvPz()
        if self.config['use_lensbin']:
            self.prepare_lensbin()
        self.compute_at_meas()
        
    def load_sumwlssigcritinvPz(self):
        data = np.loadtxt(self.config['fname_sumwlssigcritinvPz'])
        if data.ndim == 1: # In case lens redshift bin is single redshift.
            data = np.reshape(data, (1, -1))
        self.zl = data[:,1]
        self.sumwlssigcritinvPz = data[:,2:]
        self.zs = np.loadtxt(self.config['fname_zsbin'])
        assert self.sumwlssigcritinvPz.shape[1] == self.zs.size, 'shapes of the sumwlssigcritinvPz and zs do not match.'
        
    def compute_at_meas(self):
        """
        This method computes all the neccesary quantities for the computetion of measurement correction factor.
        - cosmo_meas : a cosmology instance at measurement cosmology.
        - dSigma_corr_numerator : the numerator of the dSigma correction factor.
        """
        self.cosmo_meas = self.get_cosmo(self.config['Om'], self.config['wde'])
        self.dSigma_corr_numerator_meas = self._get_dSigma_corr_numerator(0.0, self.config['Om'], self.config['wde'])
        
    def get_cosmo(self, Om, wde):
        return astropy.cosmology.FlatwCDM(100, Om, w0=wde)
    
    def prepare_lensbin(self):
        nbin = self.config['n_lensbin']
        zmin = self.zl.min()
        zmax = self.zl.max()
        zlbin_lowedge = np.linspace(zmin, zmax, nbin+1)[:-1]
        dz = zlbin_lowedge[1]-zlbin_lowedge[0]
        self.zl_bin = zlbin_lowedge + dz/2.0
        idx = np.array((self.zl-zmin)/dz, dtype=int)
        
        self.hist_lensbin = np.empty(nbin)
        self.sumwlssigcritinvPz_lensbin_ave = np.empty((nbin, self.zs.size))
        for i in range(nbin):
            sel = idx == i
            self.hist_lensbin[i] = np.sum(sel)
            self.sumwlssigcritinvPz_lensbin_ave[i,:] = np.mean(self.sumwlssigcritinvPz[sel, :], axis=0)
        
        # delete
        del self.sumwlssigcritinvPz
    
    def _get_dSigma_corr_numerator_direct(self, dpz, Om, wde):
        zl = self.zl
        zs = self.zs - dpz
        sumwlssigcritinvPz = self.sumwlssigcritinvPz
        
        cosmo = self.get_cosmo(Om, wde)

        zl_rough = np.linspace(zl.min(), zl.max()+0.001,100) # Max has padding
        chi_zl = ius(zl_rough, cosmo.comoving_distance(zl_rough).value)(zl)
        chi_zs = cosmo.comoving_distance(zs).value

        dSigma_corr_numerator = 0.0
        for j in range(len(zs)):
            if chi_zs[j] <= 0.0:
                continue
            Sigma_cr_inv_array = 1./sigcrit_prefactor * (1.+zl)*chi_zl*(1.-chi_zl/chi_zs[j])
            Sigma_cr_inv_array[Sigma_cr_inv_array <0.0] = 0.0
            dSigma_corr_numerator += np.sum(Sigma_cr_inv_array*sumwlssigcritinvPz[:,j])

        return dSigma_corr_numerator
    
    def _get_dSigma_corr_numerator_lensbin(self, dpz, Om, wde):
        zl = self.zl_bin
        zs = self.zs - dpz
        sumwlssigcritinvPz = self.sumwlssigcritinvPz_lensbin_ave
        H  = self.hist_lensbin
        
        cosmo = self.get_cosmo(Om, wde)

        zl_rough = np.linspace(zl.min(), zl.max()+0.001,100) # Max has padding
        chi_zl = ius(zl_rough, cosmo.comoving_distance(zl_rough).value)(zl)
        chi_zs = cosmo.comoving_distance(zs).value

        dSigma_corr_numerator = 0.0
        for j in range(len(zs)):
            if chi_zs[j] <= 0.0:
                continue
            Sigma_cr_inv_array = 1./sigcrit_prefactor * (1.+zl)*chi_zl*(1.-chi_zl/chi_zs[j])
            Sigma_cr_inv_array[Sigma_cr_inv_array <0.0] = 0.0
            dSigma_corr_numerator += np.sum(H*Sigma_cr_inv_array*sumwlssigcritinvPz[:,j])

        return dSigma_corr_numerator
    
    def _get_dSigma_corr_numerator(self, dpz, Om, wde):
        if self.config.get('use_lensbin', False):
            dSigma_corr_numerator = self._get_dSigma_corr_numerator_lensbin(dpz, Om, wde)
        else:
            dSigma_corr_numerator = self._get_dSigma_corr_numerator_direct(dpz, Om, wde)
        return dSigma_corr_numerator
    
    def get_dSigma_corr(self, dpz, Om, wde):
        dSigma_corr = self._get_dSigma_corr_numerator(dpz, Om, wde)/self.dSigma_corr_numerator_meas
        return dSigma_corr
        
    def get_dSigma_corr_r_corr(self, zl_rep, dpz, Om, wde):
        """
        Args:
          zl_rep (float) : the representative redshift of lens samples
          dpz    (float) : the residual photoz bias parameter
          Om     (float) : Omega_m at the required cosmology
          wde    (float) : w_de at the required cosmology
        Returns:
          dSigma_corr (float): the multiplicative correction factor for dSigma
          r_corr      (float): the multiplicative correction factor for comoving radial separation
        """
        cosmo = self.get_cosmo(Om, wde)
        dSigma_corr = self.get_dSigma_corr(dpz, Om, wde)
        r_corr = cosmo.comoving_distance(zl_rep).value/self.cosmo_meas.comoving_distance(zl_rep).value
        return dSigma_corr, r_corr
    
    # usefull for likelihood
    def set_param(self, param):
        self.current_param = param
        
    def get_corrs(self, zl_rep):
        param = self.current_param
        dpz, Om, wde = param['dpz'], param['Om'], param['wde']
        return self.get_dSigma_corr_r_corr(zl_rep, dpz, Om, wde)
        
        
class wp_meascorr_class:
    def __init__(self, config):
        """
        Args:
          config (dict) : Following keys must be included
            - Om   : Omega_m used for measurement
            - wde  : Equation of state of dark energy used for measurement
        
        https://arxiv.org/abs/2111.10966
        """
        self.config = config
        self.compute_at_meas()
        
    def compute_at_meas(self):
        """
        This method computes all the neccesary quantities for the computetion of measurement correction factor.
        - cosmo_meas : a cosmology instance at measurement cosmology.
        - dSigma_corr_numerator : the numerator of the dSigma correction factor.
        """
        self.cosmo_meas = self.get_cosmo(self.config['Om'], self.config['wde'])
        
    def get_cosmo(self, Om, wde):
        return astropy.cosmology.FlatwCDM(100, Om, w0=wde)
    
    def get_pimax_corr_r_corr(self, zl_rep, Om, wde):
        """
        Args:
          zl_rep (float) : the representative redshift of lens samples
          Om     (float) : Omega_m at the required cosmology
          wde    (float) : w_de at the required cosmology
        Returns:
          wp_corr (float): the multiplicative correction factor for wp
          r_corr  (float): the multiplicative correction factor for comoving radial separation
        """
        cosmo   = self.get_cosmo(Om, wde)
        pimax_corr = cosmo.inv_efunc(zl_rep)/self.cosmo_meas.inv_efunc(zl_rep) # pi = c*Delta z/ H_0*E(z), pimax is proportional to **inverse** of E(z).
        r_corr  = cosmo.comoving_distance(zl_rep).value/self.cosmo_meas.comoving_distance(zl_rep).value        
        return pimax_corr, r_corr
    
    # useful for likelihood
    def set_param(self, param):
        self.current_param = param
        
    def get_corrs(self, zl_rep):
        param = self.current_param
        Om, wde = param['Om'], param['wde']
        return self.get_pimax_corr_r_corr(zl_rep, Om, wde)
    