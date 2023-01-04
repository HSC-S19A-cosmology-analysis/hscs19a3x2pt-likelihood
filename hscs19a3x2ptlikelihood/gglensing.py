"""
by Sunao Sugiyama

"""
from scipy import integrate
from .cosmology import cosmology_class
from .linear_power import linear_darkemu_class
from .nonlinear_power import nonlinear_pyhalofit_class
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import numpy as np
from .fftlog import hankel, fftlog
from scipy.special import eval_legendre
from .utils import setdefault_config, empty_dict

config_default_mb = {'do_Kaiser_corr':True, 
                     'pimax_wp'      :100.0, 
                     'binave'        :True, 
                     'verbose':True}

class minimal_bias_class:
    def __init__(self, config=empty_dict()):
        self.config = config
        setdefault_config(self.config, config_default_mb)
        # update instances with consistent cosmology 
        self.cosmology = cosmology_class()
        self.set_cosmology(self.cosmology, init=True)
        self.pk_list = [0,0,0,0]
        self.ds_list = [0,0]
        self.wp_list = [0,0]

        ## init nuiscane
        self.nuisance = {'b1':1.0, 'Mpm':0.0}
        
    def init_flags(self):
        self.computed_wp = False
        self.computed_ds = False
        
    def set_cosmology(self, cosmology, init=False):
        if (not self.cosmology == cosmology) or init:
            self.cosmology = cosmology.copy()
            self.chi2z, self.z2chi = self.cosmology.get_chi2z_z2chi(0, 15, 200)
            self.init_flags()
        elif self.config['verbose']:
            print('Got same cosmological parameters. Keep quantities already computed')
        
    def set_galaxy(self, nuisance):
        """
        set nuisance parameter for the model. at least must include
        - b1 : linear galaxy bias
        - Mpm: point mass in Msun/h, contributing as +M/R^2 from small scales.
        """
        self.nuisance.update(nuisance)
        self.init_flags()
        
    def set_pk(self, z, k, pklin, pknonlin):
        """
        set power spectra
        """
        self.pk_list = [z, k, pklin, pknonlin]
        self.init_flags()
        
    def set_z2fz(self, z2fz):
        """
        set mapping function from redshift z to logarithmic graowth rate f(z).
        """
        self.z2fz = z2fz
        self.init_flags()
    
    def _get_ds(self, k, pk, dlnR, N_extrap_low=None, N_extrap_high=None):
        if N_extrap_low is None:
            N_extrap_low = k.size
        if N_extrap_high is None:
            N_extrap_high = k.size
        b1      = self.nuisance['b1']
        rhocrit = 2.77536627e11 # h^2 M_sun Mpc^-3
        rhom0   = rhocrit*self.cosmology.get_Om()
        myhankel = hankel(k, k**2*pk, 1.01, 
                          N_extrap_low=N_extrap_low, 
                          N_extrap_high=N_extrap_high)
        if dlnR > 0:
            D = 2 # dimension
            R, ds = myhankel.hankel_binave(2, dlnR, D)
        else:
            R, ds = myhankel.hankel(2)
        ds *= b1*rhom0/(2.0*np.pi)
        ds += self.nuisance['Mpm']/R**2
        return R, ds/1e12 # [h M_sun/pc^2]
    
    def _compute_ds(self, dlnR):
        z, k, pklin, pknonlin = self.pk_list
        R, ds = self._get_ds(k, pknonlin, dlnR)
        self.ds_list = [R, ds, dlnR]
        self.computed_ds = True
        
    def _get_xigg_n(self, k, pk, n, N_extrap_low=None, N_extrap_high=None):
        if N_extrap_low is None:
            N_extrap_low = k.size
        if N_extrap_high is None:
            N_extrap_high = k.size
        b1      = self.nuisance['b1']
        myfftlog = fftlog(k, k**3*pk, 1.01, 
                          N_extrap_low=N_extrap_low, 
                          N_extrap_high=N_extrap_high)
        r, xi = myfftlog.fftlog(n)
        xi *= b1**2/(2*np.pi**2)
        return r, xi
        
    def _get_wp_pimax(self, k, pk, pimax, dlnR):
        r, xi = self._get_xigg_n(k, pk, 0)
        
        rpi = np.logspace(-3, np.log10(pimax), 300)
        rp2d, rpi2d = np.meshgrid(r, rpi)
        s = (rp2d**2+rpi2d**2)**0.5
        xi2d = ius(r, xi, ext=3)(s)
        
        wp = 2*integrate.simps(xi2d, rpi, axis=0)
        
        if dlnR>0.0:
            wp = binave_array(r, wp, dlnR)
        
        return r, wp
        
    def _get_wp_pimaxinf(self, k, pk, dlnR, N_extrap_low=None, N_extrap_high=None):
        if N_extrap_low is None:
            N_extrap_low = k.size
        if N_extrap_high is None:
            N_extrap_high = k.size
        b1      = self.nuisance['b1']
        myhankel = hankel(k, pk*k**2, 1.01, 
                          N_extrap_low, 
                          N_extrap_high)
        if dlnR == 0.0:
            R, wp = myhankel.hankel(0)
        else:
            R, wp = myhankel.hankel_binave(0, dlnR, 2)
        wp *= b1**2/(2.0*np.pi)
        return R, wp
    
    def _get_wp_Kaiser(self, k, pk, fz, pimax):
        r, xi0 = self._get_xigg_n(k, pk, 0)
        r, xi2 = self._get_xigg_n(k, pk, 2)
        r, xi4 = self._get_xigg_n(k, pk, 4)
        xi2 = -xi2 # fftlog does not include (-i)^n factor

        # calculate beta
        b1= self.nuisance['b1']
        beta = fz/b1

        wp_rsd = _get_wp_aniso(r, xi0, xi2, xi4, beta, r, pimax)
        
        return r, wp_rsd
    
    def _compute_wp(self, dlnR, pimax=None, do_Kaiser_corr=None):
        z, k, pklin, pknonlin = self.pk_list
        if do_Kaiser_corr is None:
            do_Kaiser_corr = self.config['do_Kaiser_corr']
        if pimax is None:
            pimax          = self.config['pimax_wp']
        if not do_Kaiser_corr:
            if pimax == 'inf':
                r, wp = self._get_wp_pimaxinf(k, pknonlin, dlnR)
            elif isinstance(pimax, (int, float)):
                r, wp = self._get_wp_pimax(k, pknonlin, pimax, dlnR)
            else:
                raise ValueError('pimax must be int or "inf".')
        else:
            if pimax == 'inf':
                raise ValueError('Cannot apply Kaiser with pimax="inf".')
            elif isinstance(pimax, (int, float)):
                fz = self.z2fz(z)
                r, wp = self._get_wp_pimax(k, pknonlin, pimax, 0.0)
                r_aniso, wp_aniso = self._get_wp_Kaiser(k, pklin, fz, pimax)
                r_iso  , wp_iso   = self._get_wp_pimax(k, pklin, pimax, 0.0)
                assert np.all(r_aniso==r) and np.all(r_iso==r), 'r binning do not match.'
                wp *= wp_aniso/wp_iso
            else:
                raise ValueError('pimax must be int or "inf".')
            if dlnR > 0.0:
                wp = binave_array(r, wp, dlnR)
        
        self.wp_list = [r, wp, dlnR]
        self.computed_wp = True
        
    def get_ds(self, R, dlnR=None):
        if self.config['binave'] and dlnR is None:
            dlnR = np.log(R[1]/R[0])
        elif not self.config['binave']:
            dlnR = 0.0
        if (not self.computed_ds) or (dlnR != self.ds_list[2]):
            self._compute_ds(dlnR)
        return ius(self.ds_list[0], self.ds_list[1])(R)
    
    def get_wp(self, R, dlnR=None, pimax=None, do_Kaiser_corr=None):
        """
        if dlnR, pimax or do_Kaiser_corr is given as input, this will once take precedence over 
        the setting in config.
        """
        if self.config['binave'] and dlnR is None:
            dlnR = np.log(R[1]/R[0])
        elif not self.config['binave']:
            dlnR = 0.0
        if (not self.computed_wp) or (dlnR != self.wp_list[2]):
            self._compute_wp(dlnR, pimax=pimax, do_Kaiser_corr=do_Kaiser_corr)
        return ius(self.wp_list[0], self.wp_list[1])(R)
    
config_default_mag = {'binave':True, 
                      'verbose':True}

class magnificationbias_class:
    def __init__(self, config=empty_dict()):
        self.config = config
        setdefault_config(self.config, config_default_mag)
        self.cosmology = cosmology_class()
        self.set_cosmology(self.cosmology, init=True)
        self.load_nzs()
        self.load_nzl()
        self.init_flags()
        
    def init_flags(self):
        self.computed_ds_mag = False
        
    def set_cosmology(self, cosmology, init=False):
        if (not self.cosmology == cosmology) or init:
            self.cosmology = cosmology.copy()
            self.chi2z, self.z2chi = self.cosmology.get_chi2z_z2chi(0, 15, 200)
            self.init_flags()
        elif self.config['verbose']:
            print('Got same cosmological parameters. Keep quantities already computed')
        
    def set_nuisance(self, nuisance):
        """
        set nuisance parameter for the model. at least must include
        - alphamag: magnification bias parameter 
        """
        self.nuisance = nuisance
        self.init_flags()
    
    def set_kzpktable(self, k, z, pktable):
        """
        For ds_mag, wp_mag(not implemented)
        """
        self.kzpktable = [k,z,pktable]
        self.init_flags()
        
    def load_nzs(self, fname=None):
        if fname is None:
            fname = self.config['nzs']
        self.zs_nzs = np.loadtxt(fname, unpack=True)
        
    def load_nzl(self, fname=None):
        if fname is None:
            fname = self.config['nzl']
        self.zl_nzl = np.loadtxt(fname, unpack=True)
        
    def get_kz_request(self, zl_min=1e-2, zl_max=10, zl_switch=1e-1, 
                       zlbin_low=50, zlbin_high=300):
        k = np.logspace(-4, 3, 500)
        z = np.hstack([np.logspace(np.log10(zl_min), 
                                   np.log10(zl_switch)-0.01, 
                                   zlbin_low),
                       np.linspace(zl_switch,
                                   zl_max,
                                   zlbin_high)])
        return k,z
    
    def _kzpktable2pltable(self, k, z, pktable, l):
        """
        Convert P(k_i, z_j) -> P(l_i/chi_j, z_j)
        """
        pltable = np.empty((l.size, z.size))
        chi = self.z2chi(z) # [Mpc/h]
        for i,_z in enumerate(z):
            pltable[:,i] = log_extrap_func(k, pktable[:,i])((l+0.5)/chi[i])
        return pltable
    
    def _get_clSigmacrit_mag(self, k, z, pktable):
        zs, nzs = self.zs_nzs
        zl, nzl = self.zl_nzl
        dpz     = self.nuisance['dpz']
        window = get_Sigmacr_Cl_window(zs, nzs, zl, nzl, self.z2chi, dpz) # [h/Mpc]
        
        H0 = 100/299792.4580 # [h/Mpc]
        Om= self.cosmology.get_Om()
        rhocrit = 2.77536627e11 # h^2 M_\odot/Mpc^3
        prefactor = (2.0/3.0*rhocrit/H0**2)*(3.0/2.0*H0**2*Om)**2 # [h^4Msun/Mpc^5]
        
        l = np.logspace(0, 5, 1000)
        pltable = self._kzpktable2pltable(k, z, pktable, l)
        
        chi= self.z2chi(z) # [Mpc/h]
        integrand   = window(chi)*pltable
        clSigmacrit = integrate.simps(integrand.T, chi, axis=0)
        clSigmacrit*= prefactor/1e12 # hMsun/pc^2
        return l, clSigmacrit
    
    def _get_ds_mag(self, zl_rep, R, k, z, pktable, dlnR, N_extrap_low=None, N_extrap_high=None):
        
        l, clSigmacrit = self._get_clSigmacrit_mag(k, z, pktable)
        if N_extrap_low is None:
            N_extrap_low = l.size
        if N_extrap_high is None:
            N_extrap_high = l.size
        if np.any(clSigmacrit < 0):
            print('here!')
        myhankel = hankel(l, l**2*clSigmacrit, 1.01, 
                          N_extrap_low=N_extrap_low,
                          N_extrap_high=N_extrap_high,
                          c_window_width=0.25)
    
        if dlnR > 0:
            D = 2 # dimension
            t, ds_mag = myhankel.hankel_binave(2, dlnR, D)
        else:
            t, ds_mag = myhankel.hankel(2)
        ds_mag /= 2.0*np.pi

        chil_rep = self.z2chi(zl_rep) # [Mpc/h]
        ds_mag = ius(t, ds_mag, ext=3)(R/chil_rep)
        alpha  = self.nuisance['alphamag']
        ds_mag*= 2*(alpha-1)
        
        return ds_mag
        
    def get_ds_mag(self, zl_rep, R, k, z, pktable, dlnR=None):
        if self.config['binave'] and dlnR is None:
            dlnR = np.log(R[1]/R[0])
        elif not self.config['binave']:
            dlnR = 0.0
        ds_mag = self._get_ds_mag(zl_rep, R, k, z, pktable, dlnR)
        return ds_mag
    
    
def log_extrap_func(x,y):
    def func(x_new):
        if isinstance(x_new, (int, float)):
            x_new = np.atleast_1d(x_new)
        ans = np.zeros(x_new.size)
        sel = x_new < x.min()
        if np.sum(sel):
            ans[sel] = np.exp( np.log(y[1]/y[0])/np.log(x[1]/x[0]) * np.log(x_new[sel]/x[0]) + np.log(y[0]) )
        sel = x.max() < x_new
        if np.sum(sel):
            ans[sel] = np.exp( np.log(y[-2]/y[-1])/np.log(x[-2]/x[-1]) * np.log(x_new[sel]/x[-1]) + np.log(y[-1]) )
        sel = np.logical_and(x.min()<= x_new, x_new <= x.max())
        ans[sel] = 10**ius(np.log10(x),np.log10(y))(np.log10(x_new[sel]))
        return ans
    return func
    
def binave_array(x, y, dlnx, D=2, nbin=100):
    """
    Assumes dlnx << np.diff(np.log(x)).
    Performs the forward bin average in dimension D.
    ::math::
        \\bar{y} = \\frac{1}{d\\ln x} \\int_{\\ln x}^{\\ln x+d\\ln x} x^D y(x)
    """
    X = np.linspace(0.0, dlnx, nbin)
    x2d, X2d = np.meshgrid(x,X)
    arg = x2d*np.exp(X2d)
    y2d = ius(x, y)(arg)
    
    nom = integrate.simps(arg**D, X, axis=0)
    
    ybar = integrate.simps(y2d*arg**D, X, axis=0)/nom
    
    return ybar
        
def _get_wp_aniso(r, xi0, xi2, xi4, beta, rp_in, pimax):
    # Numerator of Eq. (48) of arxiv: 1206.6890 using mutipole expansion of anisotropic xi including Kaiser effect in Eq. (51) of the same paper.
    dlnrp_min = 0.01 # bin size of dlnrp enough to obtain 0.01 %
    if np.log10(rp_in[1]/rp_in[0]) < dlnrp_min:
        print('Input rp is dense. Using more sparse rp to calculate wp_aniso.')
        # calcurate wp_aniso on more sparse rp and then interpolate it to obtain wp_aniso on rp_in.
        rp = 10**np.arange(np.log10(rp_in.min()), np.log10(rp_in.max()), dlnrp_min)
        interpolate = True
    else:
        rp = rp_in
        interpolate = False

    rpi = np.logspace(-3, np.log10(pimax), 300) # Ok binning for 1% accuracy.

    rp2, rpi2 = np.meshgrid(rp, rpi)

    s = (rp2**2+rpi2**2)**0.5
    mu= rpi2/s

    xi0s = (1+2.0/3.0*beta+1.0/5.0*beta**2)*ius(r, xi0)(s)
    xi2s = (4.0/3.0*beta+4.0/7.0*beta**2)*ius(r, xi2)(s)
    xi4s = 8.0/35.0*beta**2*ius(r, xi4)(s)

    p0 = 1
    p2 = eval_legendre(2, mu)
    p4 = eval_legendre(4, mu)

    xi_aniso = 2*(xi0s*p0+xi2s*p2+xi4s*p4)

    wp_aniso = integrate.simps(xi_aniso, rpi, axis=0)

    if interpolate:
        wp_aniso = ius(rp, wp_aniso)(rp_in)
    
    return wp_aniso
    
def get_Sigmacr_Cl_window(zs, nzs, zl, nzl, z2chi, dpz):
    if isinstance(zl, (int, float)) and isinstance(nzl, (int, float)):
        zl = np.array([zl])
        nzl = np.array([1])
        dzl = 1
    else:
        dzl = np.diff(zl)[0]
        assert np.all(np.isclose(np.diff(zl), dzl))
    if isinstance(zs, (int, float)) and isinstance(nzs, (int, float)):
        zs = np.array([zs-dpz])
        nzs = np.array([1])
        dzs = 1
    else:
        zs = zs - dpz
        dzs = np.diff(zs)[0]
        assert np.all(np.isclose(np.diff(zs), dzs, rtol=0.01)) # photoz bin must be linear within 1% 

    nzs_normed = nzs/(np.sum(nzs)*dzs)
    nzl_normed = nzl/(np.sum(nzl)*dzl)
    
    chil = z2chi(zl) # Mpc/h
    chis = z2chi(zs) # Mpc/h
    
    c0, c1, c2 = [], [], []
    for _zs, _nzs, _chis in zip(zs-dpz, nzs_normed, chis):
        _c0, _c1, _c2 = [], [], []
        for _zl, _nzl, _chil in zip(zl, nzl_normed, chil):
            if _zl >= _zs:
                _c0.append(0.0)
                _c1.append(0.0)
                _c2.append(0.0)
            else:
                _c0.append(_nzs*_nzl/(1+_zl)/_chil**2/(_chis-_chil) * _chil*_chis )
                _c1.append(_nzs*_nzl/(1+_zl)/_chil**2/(_chis-_chil) * (_chil+_chis) )
                _c2.append(_nzs*_nzl/(1+_zl)/_chil**2/(_chis-_chil) )
        c0.append(_c0)
        c1.append(_c1)
        c2.append(_c2)
        
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    
    zlMat, zsMat = np.meshgrid(zl, zs-dpz)
    assert np.all(c0.shape == zlMat.shape)
    assert np.all(c0.shape == zsMat.shape)
    
    z = np.linspace(0.0, zl[zl>0].max()*1.01, 100)
    chi = z2chi(z) # Mpc/h
    c0_chi, c1_chi, c2_chi = [], [], []
    for _z, _chi in zip(z, chi):
        mask = np.logical_and(zlMat>_z, zsMat>_z, zsMat>zlMat)
        c0_chi.append( np.sum(c0[mask])*dzl*dzs )
        c1_chi.append( np.sum(c1[mask])*dzl*dzs )
        c2_chi.append( np.sum(c2[mask])*dzl*dzs )
    c0_chi = np.array(c0_chi)
    c1_chi = np.array(c1_chi)
    c2_chi = np.array(c2_chi)
    
    window = (c0_chi - c1_chi*chi + c2_chi*chi**2)*(1+z)**2
    window = ius(chi, window, ext=3) # [h/Mpc]
    return window
    
