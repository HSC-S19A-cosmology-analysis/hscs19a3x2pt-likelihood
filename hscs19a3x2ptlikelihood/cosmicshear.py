"""
by Sunao Sugiyama

"""
from scipy import integrate
from .cosmology import cosmology_class
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import numpy as np
from .fftlog import hankel
from .utils import setdefault_config, empty_dict

config_default = {'compute_xipm_each_term':False, 
                  'binave':True, 
                  'verbose':True}
class cosmicshear_class:
    def __init__(self,config=empty_dict()):
        self.config = config
        setdefault_config(self.config, config_default)
        self.config.update(config)
        self.cosmology = cosmology_class()
        self.set_cosmology(self.cosmology, init=True)
        self.load_nzs()
        self.set_nuisance({})
        
    def init_flags(self):
        self.computed_lens_window = False
        self.computed_IA_window   = False
        self.computed_cl          = False
        self.computed_xipm   = False
        self.dlnt = -1.0
    
    def set_cosmology(self, cosmology, init=False):
        if (not self.cosmology == cosmology) or init:
            self.cosmology = cosmology.copy()
            self.chi2z, self.z2chi = self.cosmology.get_chi2z_z2chi(0.0, 15, 200)
            self.init_flags()
        elif self.config['verbose']:
            print('Got same cosmological parameters. Keep quantities already computed')
        
    def set_nuisance(self, nuisance):
        """
        set nuisance parameter for the model. at least must include
        - dpz   : residual photo-z
        - dm    : residual multiplicative bias
        - AIA   : amplitude of IA
        - etaIA : index of redshift dependence of IA
        - z0    : pivot redshift of IA
        """
        self.nuisance = {'dpz':0.0, 'dm':0.0, 'AIA':0.0, 'etaIA':0.0, 'z0IA':0.62, 'zs_dump':0.1}
        self.nuisance.update(nuisance)
        self.init_flags()
    
    def set_z2D(self, z2D):
        """
        set a mapping function from redshift z to the linear growth function D(z)
        """
        self.z2D = z2D
        self.init_flags()
        
    def set_kzpktable(self, k, z, pktable):
        self.kzpktable = [k,z,pktable]
        self.init_flags()
        
    def load_nzs(self, fnames=None):
        """
        load n(zs) or stacked pof zs of source galaxies
        """
        if fnames is None:
            fnames = self.config['nzs']
        if fnames is None:
            return 0
        if isinstance(fnames, str):
            fnames = [fnames]
        self.nzs_list = []
        zs_list = []
        for fname in fnames:
            zs, nzs = np.loadtxt(fname, unpack=True, usecols=(0,1,))
            self.nzs_list.append( nzs )
            zs_list.append(zs)
        for zs in zs_list:
            assert np.all(zs_list[0] == zs), 'source bin mismatch.'
        self.zsbin= zs_list[0]
        self.n_source = len(fnames)
        self.init_flags()
        
    def set_nzs(self, zsbin, nzs_list):
        self.zsbin = zsbin
        self.nzs_list = nzs_list
        self.n_source = len(nzs_list)
        self.init_flags()
    
    def _q_over_chil(self, zs, nzs, dpz, zl_max=None, zl_bin=100, return_func=True):
        """
        returns lensing efficiency over comoving distance, q(chi)/chi
        
        Args:
          zs (array) : source redshift
          nzs (array): source redshift distribution
          dpz (float): residual photoz uncertainty. dpz<0 means source 
                       galxies are further away than estimated.
          cosmo (astropy.cosmology): astropy cosmology instance
          zl_max (float) : maximum redshift to which lensing window is calculated.
          zl_bin (int) : size of lens redshift bin.
          return_func (bool) : return func or array
        Returns:
          q_over_chil: lensing efficiency divided by comoving distance at 
                       lens redshift. unit is [h/Mpc]
                       
        See for photo-z bias, https://www.overleaf.com/read/kkkjqvcwwfhr
        """
        
        if zl_max is None:
            zl_max = zs.max()
        zl = np.linspace(1e-4, zl_max, zl_bin)
        chil = self.z2chi(zl) # [Mpc/h]

        H0 = 100 # [h km/sec/Mpc]
        H0/=299792.4580 # [h/Mpc]
        Om0= self.cosmology.get_Om()
        prefactor = 3.0/2.0*Om0*H0**2*(1+zl)

        nzs_norm = integrate.trapz(nzs, zs)
        chis = self.z2chi(zs-dpz) # [Mpc/h]

        q_over_chil = []
        # avoid zero-division and the negative contribution.
        sel = zs> max(0, dpz)
        for _chil in chil:
            integrand = np.zeros(zs.shape)
            integrand[sel] = (chis[sel]-_chil)/chis[sel]*nzs[sel]/nzs_norm
            integrand[integrand<0] = 0.0
            _q = integrate.trapz(integrand, zs)
            q_over_chil.append(_q)
        q_over_chil = prefactor*np.array(q_over_chil)

        if return_func:
            return ius(chil, q_over_chil, ext=3)
        else:
            return zl, chil, q_over_chil
        
    def _compute_q_over_chil(self):
        self.q_over_chi_list = []
        zsbin = self.zsbin
        for i in range(self.n_source):
            nzs = self.nzs_list[i]
            dpz = self.nuisance['dpz'][i]
            self.q_over_chi_list.append( self._q_over_chil(zsbin, nzs, dpz) )
        self.computed_lens_window = True
            
    def _FIA_over_chi(self, z, AIA, etaIA, z0IA):
        # coefficients
        C1 = 5e-14 # h^-2 Msun^-1 Mpc^3
        rhocrit = 2.77536627e11 # h^2 M_sun Mpc^-3
        Om0= self.cosmology.get_Om()
        z0 = 0.62
        Dp = self.z2D(z)
        F = -AIA*C1*rhocrit*Om0/Dp*((1+z)/(1+z0IA))**etaIA
        return F
    
    def _FIAp_over_chi(self, zs, nzs, dpz, AIA, etaIA, z0IA, zs_dump):
        """
        """
        z = np.linspace(1e-4, self.zsbin.max(), 100)
        chi=self.z2chi(z) # [Mpc/h]
        
        pzs_est = ius(zs, nzs/integrate.trapz(nzs, zs), ext=1)
        pzs_true= pzs_est(z+dpz)
        
        cosmo = self.cosmology.get_astropycosmo()
        Hz    = cosmo.H(z).value # [km/sec/Mpc]
        h     = self.cosmology.get_h()
        Hz   /= h # [h km/sec/Mpc]
        Hz   /= 299792.4580 # [h/Mpc]
        
        F = self._FIA_over_chi(z, AIA, etaIA, z0IA)
        Fp_ov_chi = F*pzs_true*Hz/chi * (1-np.exp(-(z/zs_dump)**2))
        return ius(chi,Fp_ov_chi,ext=3)
        
    def _compute_FIAp_over_chi(self):
        AIA   = self.nuisance['AIA']
        etaIA = self.nuisance['etaIA']
        z0IA  = self.nuisance['z0IA']
        zs_dump = self.nuisance['zs_dump']
        self.FIAp_over_chi_list = []
        for i in range(self.n_source):
            nzs = self.nzs_list[i]
            dpz = self.nuisance['dpz'][i]
            Fp = self._FIAp_over_chi(self.zsbin, nzs, dpz, AIA, etaIA, z0IA, zs_dump)
            self.FIAp_over_chi_list.append( Fp )
        self.computed_IA_window = True
    
    def get_kz_request(self, zl_min=1e-2, zl_max=10, zl_switch=1e-1, 
                       zlbin_low=50, zlbin_high=300):
        k = np.logspace(-4, 3, 500)
        z = np.hstack([np.logspace(np.log10(zl_min), 
                                   np.log10(zl_switch)-0.0001, 
                                   zlbin_low),
                       np.linspace(zl_switch,
                                   zl_max,
                                   zlbin_high)])
        return k,z
    
    def __get_finite_shell_chieff(self, chi):
        # arxiv: 1901.09488 Eq. (B5)
        dchi = np.diff(chi)[0]
        r1 = chi-dchi/2.0
        r2 = chi+dchi/2.0
        chieff = 3.0/4.0*(r2**4-r1**4)/(r2**3-r1**3)
        return chieff
    
    def _kzpktable2pltable(self, k, z, pktable, l):
        """
        Convert P(k_i, z_j) -> P(l_i/chi_j, z_j)
        """
        pltable = np.empty((l.size, z.size))
        chi = self.z2chi(z) # [Mpc/h]
        if self.config.get('finite_shell_model',False):
            chi = self.__get_finite_shell_chieff(chi)
        for i,_z in enumerate(z):
            pltable[:,i] = log_extrap_func(k, pktable[:,i])((l+0.5)/chi[i])
        return pltable
    
    def _compute_cl_win1win2(self, l, z, pltable, win1, win2):
        chi= self.z2chi(z) # [Mpc/h]
        integrand = win1(chi)*win2(chi)*pltable # [h/Mpc]
        if self.config.get('finite_shell_model', False):
            # finite shell model
            # arxiv: 1901.09488 Eq. (B4)
            chieff = self.__get_finite_shell_chieff(chi)
            dchi = np.diff(chi)[0]
            integrand *= dchi*(chi/chieff)**2
        cl = integrate.simps(integrand.T, chi, axis=0)
        if self.config.get('simulation_resolution_model', False):
            # resolution effect
            # arxiv: 1901.09488 Eq. (26)
            N_SIDE = 8192
            l_sim = 1.6*N_SIDE
            cl = cl/(1+(l/l_sim)**2)
            sel = 3*N_SIDE < l
            cl[sel] = 0.0
        return cl
    
    def _compute_all_cl(self, k, z, pktable, lmin=1, lmax=1e5, lbin=1000):
        self.l = np.logspace(np.log10(lmin), np.log10(lmax), lbin)
        pltable = self._kzpktable2pltable(k, z, pktable, self.l)
        n = self.n_source
        
        self.cl_lenslens_dict = dict()
        for i in range(n):
            for j in range(n):
                key = '%d,%d'%(i,j)
                if i>j:
                    self.cl_lenslens_dict[key] = self.cl_lenslens_dict['%d,%d'%(j,i)]
                q1 = self.q_over_chi_list[i]
                q2 = self.q_over_chi_list[j]
                cl = self._compute_cl_win1win2(self.l, z, pltable, q1, q2)
                self.cl_lenslens_dict[key] = cl
        
        self.cl_lensIA_dict = dict()
        for i in range(n):
            for j in range(n):
                key = '%d,%d'%(i,j)
                if i>j:
                    self.cl_lensIA_dict[key] = self.cl_lensIA_dict['%d,%d'%(j,i)]
                q1  = self.q_over_chi_list[i]
                q2  = self.q_over_chi_list[j]
                Fp1 = self.FIAp_over_chi_list[i]
                Fp2 = self.FIAp_over_chi_list[j]
                cl12= self._compute_cl_win1win2(self.l, z, pltable, q1, Fp2)
                cl21= self._compute_cl_win1win2(self.l, z, pltable, q2, Fp1)
                self.cl_lensIA_dict[key] = cl12+cl21
                
        self.cl_IAIA_dict = dict()
        for i in range(n):
            for j in range(n):
                key = '%d,%d'%(i,j)
                if i>j:
                    self.cl_IAIA_dict[key] = self.cl_IAIA_dict['%d,%d'%(j,i)]
                Fp1 = self.FIAp_over_chi_list[i]
                Fp2 = self.FIAp_over_chi_list[j]
                cl  = self._compute_cl_win1win2(self.l, z, pltable, Fp1, Fp2)
                self.cl_IAIA_dict[key] = cl
                
        self.computed_cl = True

    def _compute_all_xipm(self, dlnt=0, N_extrap_low=None, N_extrap_high=None, method_cl2xi='fftlog'):
        n = self.n_source
        self.xip_tot_dict = dict()
        self.xim_tot_dict = dict()
        for i in range(n):
            for j in range(n):
                key = '%d,%d'%(i,j)
                if i>j:
                    self.xip_tot_dict[key] = self.xip_tot_dict['%d,%d'%(j,i)]
                    self.xim_tot_dict[key] = self.xim_tot_dict['%d,%d'%(j,i)]
                cl_tot = self.cl_lenslens_dict[key].copy()
                cl_tot+= self.cl_lensIA_dict[key]
                cl_tot+= self.cl_IAIA_dict[key]
                if method_cl2xi == 'fftlog':
                    tp, xip, tm, xim = cl2xipm(self.l, cl_tot, dlnt, 
                                               N_extrap_low=N_extrap_low,
                                               N_extrap_high=N_extrap_high)
                else:
                    tp, xip, tm, xim = cl2xipm_direct(self.l, cl_tot, dlnt, 
                                                      skip=1, x_dump=50)
                self.xip_tot_dict[key] = xip
                self.xim_tot_dict[key] = xim
                self.tp = tp
                self.tm = tm
        
        if not self.config['compute_xipm_each_term']:
            # False by default to save computational time.
            self.dlnt = dlnt
            self.computed_xipm = True
            return None
        
        self.xip_lenslens_dict = dict()
        self.xim_lenslens_dict = dict()
        for i in range(n):
            for j in range(n):
                key = '%d,%d'%(i,j)
                if i>j:
                    self.xip_tot_dict[key] = self.xip_tot_dict['%d,%d'%(j,i)]
                    self.xim_tot_dict[key] = self.xim_tot_dict['%d,%d'%(j,i)]
                cl = self.cl_lenslens_dict[key]
                tp, xip, tm, xim = cl2xipm(self.l, cl, dlnt, 
                                           N_extrap_low=N_extrap_low,
                                           N_extrap_high=N_extrap_high)
                #tp, xip, tm, xim = cl2xipm_direct(self.l, cl_tot, 
                #                                  skip=1, x_dump=50)
                self.xip_lenslens_dict[key] = xip
                self.xim_lenslens_dict[key] = xim
                self.tp = tp
                self.tm = tm
                
        self.xip_lensIA_dict = dict()
        self.xim_lensIA_dict = dict()
        for i in range(n):
            for j in range(n):
                key = '%d,%d'%(i,j)
                if i>j:
                    self.xip_tot_dict[key] = self.xip_tot_dict['%d,%d'%(j,i)]
                    self.xim_tot_dict[key] = self.xim_tot_dict['%d,%d'%(j,i)]
                cl = self.cl_lensIA_dict[key]
                tp, xip, tm, xim = cl2xipm(self.l, cl, dlnt, 
                                           N_extrap_low=N_extrap_low,
                                           N_extrap_high=N_extrap_high)
                #tp, xip, tm, xim = cl2xipm_direct(self.l, cl_tot, 
                #                                  skip=1, x_dump=50)
                self.xip_lensIA_dict[key] = xip
                self.xim_lensIA_dict[key] = xim
                self.tp = tp
                self.tm = tm
                
        self.xip_IAIA_dict = dict()
        self.xim_IAIA_dict = dict()
        for i in range(n):
            for j in range(n):
                key = '%d,%d'%(i,j)
                if i>j:
                    self.xip_tot_dict[key] = self.xip_tot_dict['%d,%d'%(j,i)]
                    self.xim_tot_dict[key] = self.xim_tot_dict['%d,%d'%(j,i)]
                cl = self.cl_IAIA_dict[key]
                tp, xip, tm, xim = cl2xipm(self.l, cl, dlnt, 
                                           N_extrap_low=N_extrap_low,
                                           N_extrap_high=N_extrap_high)
                #tp, xip, tm, xim = cl2xipm_direct(self.l, cl_tot, 
                #                                  skip=1, x_dump=50)
                dmi = self.nuisance['dm'][i]
                dmj = self.nuisance['dm'][j]
                
                self.xip_IAIA_dict[key] = (1+dmi)*(1+dmj)*xip
                self.xim_IAIA_dict[key] = (1+dmi)*(1+dmj)*xim
                self.tp = tp
                self.tm = tm
                
        self.dlnt = dlnt
        self.computed_xipm = True
        
    def get_xip(self, theta, i, j, dlnt=None, method_cl2xi='fftlog', **args):
        if not self.computed_lens_window:
            self._compute_q_over_chil()
        if not self.computed_IA_window:
            self._compute_FIAp_over_chi()
        if not self.computed_cl:
            k, z, pktable = self.kzpktable
            self._compute_all_cl(k,z, pktable)
        if self.config['binave'] and dlnt is None:
            dlnt = np.log(theta[1]/theta[0])            
        elif not self.config['binave']:
            dlnt = 0.0
        if (not self.computed_xipm) or (dlnt != self.dlnt):
            self._compute_all_xipm(dlnt=dlnt, method_cl2xi=method_cl2xi, N_extrap_low=args.get('N_extrap_low', None), N_extrap_high=args.get('N_extrap_high', None))
            
        key = '%d,%d'%(min([i,j]), max([i,j]))
        xip = ius(self.tp, self.xip_tot_dict[key])(theta)
        return xip
        
    def get_xim(self, theta, i, j, dlnt=None, method_cl2xi='fftlog', **args):
        if not self.computed_lens_window:
            self._compute_q_over_chil()
        if not self.computed_IA_window:
            self._compute_FIAp_over_chi()
        if not self.computed_cl:
            k, z, pktable = self.kzpktable
            self._compute_all_cl(k,z, pktable)
        if self.config['binave'] and dlnt is None:
            dlnt = np.log(theta[1]/theta[0])            
        elif not self.config['binave']:
            dlnt = 0.0
        if (not self.computed_xipm) or (dlnt != self.dlnt):
            self._compute_all_xipm(dlnt=dlnt, method_cl2xi=method_cl2xi, N_extrap_low=args.get('N_extrap_low', None), N_extrap_high=args.get('N_extrap_high', None))
            
        key = '%d,%d'%(min([i,j]), max([i,j]))
        xim = ius(self.tm, self.xim_tot_dict[key])(theta)
        return xim
                
    def get_tomographic_xipm(self, theta, dlnt=None, method_cl2xi='fftlog', **args):
        if not self.computed_lens_window:
            self._compute_q_over_chil()
        if not self.computed_IA_window:
            self._compute_FIAp_over_chi()
        if not self.computed_cl:
            k, z, pktable = self.kzpktable
            self._compute_all_cl(k,z, pktable)
        if self.config['binave'] and dlnt is None:
            dlnt = np.log(theta[1]/theta[0])            
        elif not self.config['binave']:
            dlnt = 0.0
        if (not self.computed_xipm) or (dlnt != self.dlnt):
            self._compute_all_xipm(dlnt=dlnt, method_cl2xi=method_cl2xi, N_extrap_low=args.get('N_extrap_low', None), N_extrap_high=args.get('N_extrap_hihg', None))
        
        xip_dict = dict()
        xim_dict = dict()
        n = self.n_source
        for i in range(n):
            for j in range(n):
                key = '%d,%d'%(i,j)
                if i>j:
                    xip_dict[key] = xip_dict['%d,%d'%(j,i)]
                    xim_dict[key] = xim_dict['%d,%d'%(j,i)]
                xip_dict[key] = ius(self.tp, self.xip_tot_dict[key])(theta)
                xim_dict[key] = ius(self.tm, self.xim_tot_dict[key])(theta)
        
        return xip_dict, xim_dict
    
        
def log_extrap_func(x,y):
    def func(x_new):
        if isinstance(x_new, (int, float)):
            x_new = np.atleast_1d(x_new)
        ans = np.zeros(x_new.size)
        sel = x_new < x.min()
        if np.sum(sel) and y[0]!=0.0 and y[1] != 0.0:
            ans[sel] = np.exp( np.log(y[1]/y[0])/np.log(x[1]/x[0]) * np.log(x_new[sel]/x[0]) + np.log(y[0]) )
        sel = x.max() < x_new
        if np.sum(sel) and y[-1]!=0.0 and y[-2] != 0.0:
            ans[sel] = np.exp( np.log(y[-2]/y[-1])/np.log(x[-2]/x[-1]) * np.log(x_new[sel]/x[-1]) + np.log(y[-1]) )
        sel = np.logical_and(x.min()<= x_new, x_new <= x.max())
        ans[sel] = 10**ius(np.log10(x[y>0]),np.log10(y[y>0]))(np.log10(x_new[sel]))
        return ans
    return func

def cl2xipm(l, cl, dlnt=0, N_extrap_low=None, N_extrap_high=None):
    """
    convert cl to xi_{+/-} using fftlog.
    Args:
      l    : must be equally spaced in logarithmic.
      cl   : angular power specturm on l.
      dlnt : bin width in logarithmic space. 
             If dlnt>0, bin-averaged xi is computed, 
             and bare xi is computed otherwise. 
    Returns
      tp   : theta for xi_+, [arcmin]
      xip  : xi_+
      tm   : theta for xi_-, [arcmin]
      xim  : xi_-
    Note
      If bin average is performed, theta is lower edge of each bin. 
      The upper edge of the bin can be computed as theta*exp(dlnt)
    """
    
    # this is our default choice for S19A 3x2pt analysis.
    # See validate_cl2xi_fftlog.ipynb for validation.
    if N_extrap_low is None:
        N_extrap_low = l.size
    if N_extrap_high is None:
        N_extrap_high = l.size
    
    myhankel = hankel(l, l**2*cl, 1.01, 
                      N_extrap_low=N_extrap_low,
                      N_extrap_high=N_extrap_high, c_window_width=0.25)
    if dlnt > 0:
        D = 2 # dimension
        tp, xip = myhankel.hankel_binave(0, dlnt, D)
        tm, xim = myhankel.hankel_binave(4, dlnt, D)
    else:
        tp, xip = myhankel.hankel(0)
        tm, xim = myhankel.hankel(4)
    xip /= 2*np.pi
    xim /= 2*np.pi
    
    # change unit
    tp = np.rad2deg(tp)*60. # arcmin
    tm = np.rad2deg(tm)*60. # arcmin
    
    return tp, xip, tm, xim

def cl2xipm_direct(l, cl, dlnt=0.0, skip=1, x_dump=50):
    from scipy.special import jn
    from scipy.integrate import quad
    
    cl_spl = ius(l, cl, ext=1)
    t = 1/l[::-1]
    
    xip = []
    for _t in t[::skip]:
        def integrand(logl):
            _l = 10**logl
            return _l**2*cl_spl(_l)*jn(0, _t*_l)*np.exp(-(_t*_l/x_dump)**2)
        a = quad(integrand, -2, 5)[0]
        a/= 2*np.pi*np.log10(np.e)
        xip.append(a)
    xip = ius(t[::skip], xip, ext=1)(t)
    
    xim = []
    for _t in t[::skip]:
        def integrand(logl):
            _l = 10**logl
            return _l**2*cl_spl(_l)*jn(4, _t*_l)*np.exp(-(_t*_l/x_dump)**2)
        a = quad(integrand, -2, 5)[0]
        a/= 2*np.pi*np.log10(np.e)
        xim.append(a)
    xim = ius(t[::skip], xim, ext=1)(t)
        
    tp = np.rad2deg(t)*60. # arcmin
    tm = np.rad2deg(t)*60. # arcmin
    xip, xim = np.array(xip), np.array(xim)
    
    if dlnt > 0:
        xip = binave_array(tp, xip, dlnt, D=2)
        xim = binave_array(tp, xim, dlnt, D=2)

    return tp, xip, tm, xim

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
