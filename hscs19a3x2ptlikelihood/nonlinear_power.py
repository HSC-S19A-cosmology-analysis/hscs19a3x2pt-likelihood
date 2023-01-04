"""
by Sunao Sugiyama

This defined the interfaces (python class) to various packages computing non-linear matter power spectrum to communicate with hscs19a3x2pt likelihood.
"""
import numpy as np
from . import pyhalofit
from .cosmology import cosmology_class
from .utils import setdefault_config, empty_dict

try:
    import camb
except:
    print('import fail: camb')

try:
    import pyhmcode
except:
    print('import fail: pyhmcode')


# base of nonlinear class
class nonlinear_base:
    def __init__(self):
        pass
    def set_pklin(self, k, z, pklin):
        pass
    def get_pknonlin(self):
        pass

# pyhalofit based nonlinear class
config_default_nonlinear_pyhalofit = {'verbose':True}
class nonlinear_pyhalofit_class(nonlinear_base):
    def __init__(self, config=empty_dict()):
        self.config = config
        setdefault_config(self.config, config_default_nonlinear_pyhalofit)
        self.cosmology = cosmology_class()
        self.set_cosmology(self.cosmology, init=True)
        self.set_nuisance({})
    
    def set_cosmology(self, cosmology, init=False):
        if (not self.cosmology == cosmology) or init:
            self.cosmology = cosmology.copy()
            self.init_flags()
        elif self.config['verbose']:
            print('Got same cosmological parameters. Keep quantities already computed')
            
    def set_nuisance(self, nuisance):
        self.nuisance = {'f1h':1.0}
        self.nuisance.update(nuisance)
        self.init_flags()
    
    def init_flags(self):
        self.pknonlin_table_args = [0,0,0]
        self.computed_pknonlin_table = False
        
    def set_pklin(self, k, z, pklin):
        self.kzpklin = k, z, pklin
        
    def get_pknonlin(self):
        Omm0 = self.cosmology.get_Omm0()
        Omde0 = self.cosmology.get_Omde0()
        h = self.cosmology.get_h()
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cosmology.cparam
        f1h = self.nuisance['f1h']
        k, z, pklin = self.kzpklin
        pkhalo = pyhalofit.pklin2pkhalofit(k, z, pklin, Omm0, Omde0, w0, wa, h, mnu, f1h=f1h)
        return pkhalo
    
    def compute_pknonlin_table(self, k_arr, z_arr, pklin_table):
        Omm0 = self.cosmology.get_Omm0()
        Omde0 = self.cosmology.get_Omde0()
        h = self.cosmology.get_h()
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cosmology.cparam
        f1h = self.nuisance['f1h']
        self.pknonlin_table=pyhalofit.pklintable2pkhalotable(k_arr, z_arr, pklin_table, Omm0, Omde0, w0, wa, h, mnu, f1h=f1h)
        self.pknonlin_table_args = [k_arr, z_arr, pklin_table]
        self.computed_pknonlin_table = True
        
    def get_pknonlin_table(self, k_arr, z_arr, pklin_table):
        kz = np.all(k_arr==self.pknonlin_table_args[0]) 
        kz&= np.all(z_arr==self.pknonlin_table_args[1])
        kz&= np.all(pklin_table==self.pknonlin_table_args[2])
        if (not self.computed_pknonlin_table) or (not kz):
            self.compute_pknonlin_table(k_arr, z_arr, pklin_table)
        return self.pknonlin_table.copy()

class nonlinear_pyhmcode_class:
    def __init__(self, config=empty_dict()):
        self.config = config
        self.cosmology = cosmology_class()
        self.set_cosmology(self.cosmology, init=True)
        self.pknonlin_table_args = [0,0,0]
    
    def set_cosmology(self, cosmology, init=False):
        if (not self.cosmology == cosmology) or init:
            self.cosmology = cosmology.copy()
            Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = cosmology.cparam
            h = cosmology.get_h()
            Om= cosmology.get_Om()
            
            self.hmcode_cosmo = pyhmcode.Cosmology()
            self.hmcode_cosmo.om_m = Om
            self.hmcode_cosmo.om_b = Ombh2/h**2
            self.hmcode_cosmo.om_v = Omde# 
            self.hmcode_cosmo.h = h
            self.hmcode_cosmo.ns = ns
            self.hmcode_cosmo.m_nu = mnu  
            
            if self.config['version'] == 'HMcode2015':
                self.hmcode_model = pyhmcode.Halomodel(pyhmcode.HMcode2015, verbose=False)
            elif self.config['version'] == 'HMcode2016':
                self.hmcode_model = pyhmcode.Halomodel(pyhmcode.HMcode2016, verbose=False)
            elif self.config['version'] == 'HMcode2020_feedback':
                self.hmcode_model = pyhmcode.Halomodel(pyhmcode.HMcode2020_feedback, verbose=False)
            
            self.init_flags()
        else:
            print('Got same cosmological parameters. Keep quantities already computed')
        
    def init_flags(self):
        self.computed_pknonlin_table = False
        
    def set_pklin(self, k, z, pklin):
        if isinstance(z, (int, float)):
            z = np.array([z])
            pklin = np.reshape(pklin, (z.size, k.size))
        self.hmcode_cosmo.set_linear_power_spectrum(k, z, pklin)
        
    def set_sigma8(self, sigma8):
        self.hmcode_cosmo.sig8 = sigma8
        self.init_flags()
        
    def set_nuisance(self, nuisance):
        if self.config['version'] == 'HMcode2015':
            self.hmcode_model.As = nuisance['As']
        elif self.config['version'] == 'HMcode2016':
            self.hmcode_model.As = nuisance['As']
        elif self.config['version'] == 'HMcode2020_feedback':
            # theat os included in hmcode cosmology instance.
            self.hmcode_cosmo.theat = 10**nuisance['logTAGN']
        self.init_flags()
        
    def get_pknonlin(self):
        pknonlin = pyhmcode.calculate_nonlinear_power_spectrum(self.hmcode_cosmo, self.hmcode_model, verbose=False)
        return pknonlin
        
    def compute_pknonlin_table(self, k_arr, z_arr, pklin_table):
        self.set_pklin(k_arr, z_arr, pklin_table.T)
        self.pknonlin_table = pyhmcode.calculate_nonlinear_power_spectrum(self.hmcode_cosmo, self.hmcode_model, verbose=False).T
        self.pknonlin_table_args = [k_arr, z_arr, pklin_table]
        self.computed_pknonlin_table = True
        
    def get_pknonlin_table(self, k_arr, z_arr, pklin_table):
        kz = np.all(k_arr==self.pknonlin_table_args[0]) 
        kz&= np.all(z_arr==self.pknonlin_table_args[1])
        kz&= np.all(pklin_table==self.pknonlin_table_args[2])
        if (not self.computed_pknonlin_table) or (not kz):
            self.compute_pknonlin_table(k_arr, z_arr, pklin_table)
        return self.pknonlin_table.copy()
    
    
# Do not use. Not tested.
def camb_pars_results_to_halofit(pars, results, halofit_version='takahashi', HMCode_A_baryon=3.13):
    """
    see https://camb.readthedocs.io/en/latest/nonlinear.html .
    """
    pars.NonLinearModel.set_params(halofit_version=halofit_version, HMCode_A_baryon=HMCode_A_baryon)
    results.get_nonlinear_matter_power_spectrum(params=pars)
    pknl_interp = results.get_matter_power_interpolator(nonlinear=True)
    return pknl_interp
