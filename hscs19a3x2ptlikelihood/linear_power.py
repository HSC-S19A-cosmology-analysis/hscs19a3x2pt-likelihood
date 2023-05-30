"""
by Sunao Sugiyama

This defined the interfaces (python class) to various packages computing linear matter power spectrum to communicate with hscs19a3x2pt likelihood.
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import integrate
from .cosmology import cosmology_class
from .utils import empty_dict, setdefault_config

# base of linear class
class linear_base:
    def __init__(self):
        pass
    def get_pklin(self, k, z):
        pass
    def get_sigma8(self, z):
        pass
    def get_pklin_table(self, k_arr, z_arr):
        pass
    
# dark emulator based linear class
from dark_emulator.darkemu.pklin import pklin_gp
from dark_emulator.darkemu import cosmo_util
print('Loading dark emulator for linear module at', cosmo_util.__file__)
config_default_linear_darkemu = {'verbose':True}
class linear_darkemu_class(linear_base):
    def __init__(self, config=empty_dict()):
        self.config = config
        setdefault_config(self.config, config_default_linear_darkemu)
        self.cosmo = cosmo_util.cosmo_class()
        self.pkL = pklin_gp()
        self.cosmology = cosmology_class()
        self.set_cosmology(self.cosmology, init=True)
        
    def set_cosmology(self, cosmology, init=False):
        """
        Args:
          cosmology (cosmology.cosmology_class) : cosmology
        """
        if (not self.cosmology == cosmology) or init:
            try:
                cosmology.test_darkemu_support()
                emucparam = cosmology.get_darkemu_cparam()
                cosmo_util.test_cosm_range_linear(emucparam)
                self.cosmology = cosmology.copy()
                self.cosmo.cparam = np.reshape(emucparam, (1,6))
                self.pkL.set_cosmology(self.cosmo)
                self.init_flags()
            except Exception as excpt:
                print(excpt)
                print('cosmological parameter out of supported range!')
        elif self.config['verbose']:
            print("Got same cosmological parameters. Keep quantities already computed")
            
    def init_flags(self):
        self.computed_pklin_table = False
        self.pklin_table_kz = [0,0]
    
    def Dgrowth_from_z(self, z):
        if isinstance(z, (int, float)):
            return self.cosmo.Dgrowth_from_z(z)
        elif z.size > 100:
            z_arr = np.linspace(z.min(), z.max(), 100)
            Dp = np.array([self.cosmo.Dgrowth_from_z(_z) for _z in z_arr])
            return ius(z_arr, Dp)(z)
        else:
            return np.array([self.cosmo.Dgrowth_from_z(_z) for _z in z])
        
    def get_z2D(self, zmin=0, zmax=15, nbin=200):
        z = np.linspace(zmin, zmax, nbin)
        D = self.Dgrowth_from_z(z)
        z2D = ius(z,D, ext=3)
        return z2D
        
    def fgrowth_from_z(self, z, dz=0.001):
        zarr= np.array([z-dz, z+dz])
        lna = -np.log(1+zarr)
        lnD = np.log(self.Dgrowth_from_z(zarr))
        return np.diff(lnD)/np.diff(lna)
    
    def get_z2fz(self, zmin=0, zmax=15, nbin=200, dz=0.001):
        z2D = self.get_z2D(zmin=zmin-dz, zmax=zmax+dz)
        z  = np.linspace(zmin, zmax, nbin)
        zp = z-0.0001
        zm = z+0.0001
        lnDp = np.log(z2D(zp))
        lnDm = np.log(z2D(zm))
        lnap = -np.log(1+zp)
        lnam = -np.log(1+zm)
        fz = (lnDp-lnDm)/(lnap-lnam)
        z2fz = ius(z, fz, ext=3)
        return z2fz
        
    def get_pklin(self, k, z):
        Dp = self.Dgrowth_from_z(z)
        return Dp**2 * self.pkL.get(k)
    
    def get_sigma8(self, logkmin=-4, logkmax=1, nint=100):
        R = 8.
        ks = np.logspace(logkmin, logkmax, nint)
        logks = np.log(ks)
        kR = ks * R
        integrant = ks**3*self.get_pklin(ks, 0.0)*self._window_tophat(kR)**2
        return np.sqrt(integrate.trapz(integrant, logks)/(2.*np.pi**2))
    
    def _window_tophat(self, kR):
        return 3.*(np.sin(kR)-kR*np.cos(kR))/kR**3
    
    def compute_pklin_table(self, k_arr, z_arr):
        Dp = self.Dgrowth_from_z(z_arr)
        pklin0 = self.pkL.get(k_arr)
        pklin2d, Dp2d = np.meshgrid(pklin0, Dp)
        self.pklin_table_kz = [k_arr, z_arr]
        self.pklin_table = (pklin2d*Dp2d**2).T
        self.computed_pklin_table = True
    
    def get_pklin_table(self, k_arr, z_arr):
        kz = np.all(k_arr==self.pklin_table_kz[0])
        kz&= np.all(z_arr==self.pklin_table_kz[1])
        if (not self.computed_pklin_table) or (not kz):
            self.compute_pklin_table(k_arr, z_arr)
        return self.pklin_table.copy()

    def get_hubble(self):
        return np.sqrt((self.cosmo.cparam[0, 0] + self.cosmo.cparam[0, 1] + 0.00064) / (1 - self.cosmo.cparam[0, 2]))

# camb based linear class
try:
    import camb
except:
    print('import fail: camb')
class linear_camb_class(linear_base):
    def __init__(self, config=empty_dict()):
        self.config = config
        self.cosmology = cosmology_class()
        self.set_cosmology(self.cosmology, init=True)
        
    def set_cosmology(self, cosmology, init=False):
        if (not self.cosmology == cosmology) or init:
            self.camb_pars = camb.CAMBparams(WantTransfer=True, 
                                             WantCls=False, 
                                             Want_CMB_lensing=False, 
                                             DoLensing=False)
            h = cosmology.get_h()
            As= cosmology.get_As()
            Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = cosmology.get_cosmology()
            self.camb_pars.set_cosmology(H0=h*100, ombh2=Ombh2, omch2=Omch2, omk=Omk, num_massive_neutrinos=1, mnu=mnu)
            self.camb_pars.set_initial_power(camb.InitialPowerLaw(As=As, ns=ns))
            self.camb_pars.set_dark_energy(w=w0, cs2=1.0, wa=wa, dark_energy_model='fluid')
            self.init_flags()
        else:
            print("Got same cosmological parameters. Keep quantities already computed")
        
    def init_flags(self):
        self.computed_pklin_table = False
        self.pklin_table_args = [0,0]
        self.pklin_initialized = False
    
    def init_pklin_array(self, extrap_kmax=True):
        redshifts = np.linspace(0.0, 10, 80)
        self.camb_pars.set_matter_power(redshifts=redshifts, kmax=1e3, nonlinear=False)
        
        results = camb.get_results(self.camb_pars)
        self.camb_pklin_interp = results.get_matter_power_interpolator(nonlinear=False,
                                                                      extrap_kmax=extrap_kmax)
        
        self.camb_sigma8  = results.get_sigma8()[-1]
        self.camb_results = results
        self.pklin_initialized = True
        
    def get_sigma8(self):
        if not self.pklin_initialized:
            self.init_pklin_array()
        return self.camb_sigma8
    
    def get_pklin(self, k, z):
        if not self.pklin_initialized:
            self.init_pklin_array()
        return self.camb_pklin_interp.P(z, k)
    
    def compute_pklin_table(self, k_arr, z_arr):
        if not self.pklin_initialized:
            self.init_pklin_array()
        self.pklin_table = self.camb_pklin_interp.P(z_arr, k_arr).T
        self.pklin_table_args = [k_arr, z_arr]
        self.computed_pklin_table = True
        
    def get_pklin_table(self, k_arr, z_arr):
        kz = np.all(k_arr==self.pklin_table_args[0]) 
        kz&= np.all(z_arr==self.pklin_table_args[1])
        if (not self.computed_pklin_table) or (not kz):
            self.compute_pklin_table(k_arr, z_arr)
        return self.pklin_table.copy()
    
    def Dgrowth_from_z(self, z):
        z2D = self.get_z2D(z.min(), z.max(), z.size)
        return z2D(z)
        
    def get_z2D(self, zmin=0, zmax=15, nbin=200, eval_k=1e-5):
        z = np.linspace(zmin, zmax, nbin)
        D = self.camb_pklin_interp.P(z, eval_k).T
        D/= D[0]
        z2D = ius(z,D, ext=3)
        return z2D
        
    def fgrowth_from_z(self, z, dz=0.001):
        zarr= np.array([z-dz, z+dz])
        lna = -np.log(1+zarr)
        lnD = np.log(self.Dgrowth_from_z(zarr))
        return np.diff(lnD)/np.diff(lna)
    
    def get_z2fz(self, zmin=0, zmax=15, nbin=200, dz=0.001):
        z2D = self.get_z2D(zmin=zmin-dz, zmax=zmax+dz)
        z  = np.linspace(zmin, zmax, nbin)
        zp = z-0.0001
        zm = z+0.0001
        lnDp = np.log(z2D(zp))
        lnDm = np.log(z2D(zm))
        lnap = -np.log(1+zp)
        lnam = -np.log(1+zm)
        fz = (lnDp-lnDm)/(lnap-lnam)
        z2fz = ius(z, fz, ext=3)
        return z2fz
    
