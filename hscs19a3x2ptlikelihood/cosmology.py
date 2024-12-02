"""
by Sunao Sugiyama

defined cosmology class used in this package.
"""
import astropy
import numpy as np
import copy
from scipy.interpolate import InterpolatedUnivariateSpline as ius

class cosmology_class:
    """
    cosmology class

    Attributes:
        cparam (np.array): cosmological parameters
    """
    def __init__(self):
        self.cparam = np.array([0.02225,  
                                0.1198 ,  
                                0.6844 ,  
                                3.094  ,  
                                0.9645 , 
                                -1.    , 
                                0.06   , 
                                0.0    , 
                                0.0    ])
    
    def get_cparam_name(self):
        """
        return the name of cosmological parameters

        Returns:
            list: name of cosmological parameters
        """
        return ['Ombh2', 'Omch2', 'Omde', 'ln10p10As', 'ns', 'w0', 'mnu', 'wa', 'Omk']
    
    def get_cparam_latex(self):
        """
        return the latex name of cosmological parameters

        Returns:
            list: latex name of cosmological parameters
        """
        return [r'$\Omega_{\rm b}h^2$', 
                r'$\Omega_{\rm c}h^2$', 
                r'$\Omega_{\rm de}$', 
                r'$\ln 10^{10}A_{\rm s}$', 
                r'$n_{\rm s}$',
                r'$w_0$',
                r'$m_{\nu}$',
                r'$w_a$',
                r'$\Omega_K$']
    
    def set_cosmology(self, cparam):
        """
        set cosmological parameters

        Args:
            cparam (np.array): cosmological parameters

        Returns:
            bool: if the cosmological parameters are changed
        
        Note:
            the order of cosmological parameters are
            [Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk]
        """
        changed = np.any(self.cparam != cparam)
        self.cparam = cparam
        return changed
    
    def get_h(self):
        """return h = H0/100"""
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cparam
        h = ((Ombh2+Omch2+0.00064*(mnu/0.06))/(1.0-Omde-Omk))**0.5
        return h

    def get_As(self):
        """return As"""
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cparam
        As= np.exp(ln10p10As)*1e-10
        return As
    
    def get_Om(self):
        """return Omega_m"""
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cparam
        return 1.0-Omde-Omk
    
    def get_Omm0(self):
        """return Omega_m"""
        return self.get_Om()
    
    def get_Omde0(self):
        """return Omega_de"""
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cparam
        return Omde
    
    def get_fnu(self):
        """
        nuetrino fraction at z=0
        """
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cparam
        h = self.get_h()
        Omnuh2   = 0.00064*(mnu/0.06)
        Omm0 = self.get_Omm0()
        fnu = Omnuh2/h**2 / Omm0
        return fnu

    def get_Ommz(self, z):
        """return Omega_m(z)"""
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cparam
        Omm0 = self.get_Omm0()
        Omde0 = self.get_Omde0()
        a = 1.0/(1+z)
        Qa2 = a**(-1.0-3.0*(w0+wa))*np.exp(-3.0*(1-a)*wa)
        Omt =1.0+(Omm0+Omde0-1.0)/(1-Omm0-Omde0+Omde0*Qa2+Omm0/a)
        Ommz=Omt*Omm0/(Omm0+Omde0*a*Qa2)
        return Ommz

    def get_Omdez(self, z):
        """return Omega_de(z)"""
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cparam
        Omm0 = self.get_Omm0()
        Omde0 = self.get_Omde0()
        a = 1.0/(1+z)
        Qa2 = a**(-1.0-3.0*(w0+wa))*np.exp(-3.0*(1-a)*wa)
        Omt =1.0+(Omm0+Omde0-1.0)/(1-Omm0-Omde0+Omde0*Qa2+Omm0/a)
        Omde=Omt*Omde0*Qa2/(Omde0*Qa2+Omm0/a)
        return Omde
    
    def get_cosmology(self):
        """return cosmological parameters"""
        return self.cparam.copy()
        
    def get_astropycosmo(self):
        """return astropy cosmology object"""
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cparam
        H0 = self.get_h()*100
        Om = self.get_Om()
        h  = self.get_h()
        Omb= Ombh2/h**2
        cosmo = astropy.cosmology.w0waCDM(H0, Om, Omde, Ob0=Omb, w0=w0, wa=wa)
        return cosmo

    def get_chi2z_z2chi(self, zmin=0, zmax=15, nbin=200):
        """
        z -> chi [Mpc/h]
        chi [Mpc/h] -> z
        """
        cosmo = self.get_astropycosmo()
        h = self.get_h()
        z   = np.linspace(zmin, zmax, nbin)
        chi = cosmo.comoving_distance(z).value # Mpc
        chi*= h
        chi2z = ius(chi, z, ext=3)
        z2chi = ius(z, chi, ext=3)
        return chi2z, z2chi
        
    def get_darkemu_cparam(self):
        """return cosmological parameters in the format of dark emulator"""
        return self.cparam[:6].copy()
    
    def test_darkemu_support(self):
        """
        test if the cosmology is supported by dark emulator

        Returns:
            bool: if the cosmology is supported by dark emulator
        """
        Ombh2, Omch2, Omde, ln10p10As, ns, w0, mnu, wa, Omk = self.cparam
        assert mnu == 0.06, 'dark emulator cannot have mnu!=0.06, but mnu=%f is given.'%mnu
        assert wa  == 0.0 , 'dark emulator cannot have w1 !=0.00, but wa =%f is given.'%wa
        assert Omk == 0.0 , 'dark emulator cannot have omk!=0.00, but omk=%f is given.'%Omk
    
    def __eq__(self, cosmology_in):
        """
        check if the cosmology is the same

        Args:
            cosmology_in (cosmology_class): cosmology to compare
            
        Returns:
            bool: if the cosmology is the same
        """
        if not isinstance(cosmology_in, cosmology_class):
            return NotImplemented
        return np.all(self.cparam == cosmology_in.cparam)
    
    def copy(self):
        """
        return a copy of the cosmology

        Returns:
            cosmology_class: copy of the cosmology
        """
        return copy.deepcopy(self)
    
