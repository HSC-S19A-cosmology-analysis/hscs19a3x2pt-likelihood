import os
import numpy as np
import copy
import pandas
from .utils import setdefault_config
from scipy.special import erfinv, erf
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import UnivariateSpline
pandas.set_option('display.max_columns', 200)

config_default = {'param':{}, 'verbose':True}

class likelihood_class:
    def __init__(self, config, dataset, verbose=True):
        """
        Args:
            config (dict): configure dict, must includes
            - model (dict): dict of model configs
            - param (dict): dict of parameters
        """
        self.config = config
        config_default['verbose'] = verbose
        setdefault_config(self.config, config_default)
        self.dataset= dataset
        
        if self.dataset is not None:
            self.probes_data  = self.dataset.get_probes()
            self.probes_model = self.dataset.get_probes().copy()
            self.covariance  = self.dataset.get_covariance()
            self.probe_names = self.probes_data.probe_names

        self.init_param()
        self.reset_nlikecall()
        
        # instantiate others needed for model or likelihood evaluation
    
    def reset_nlikecall(self):
        self.nlikecall = 0
    
    def init_param(self):
        pconf = self.config['param']
        self.param_name_full = np.array([name for name in pconf.keys()])
        
        # identify sampling parameter
        self.param_sample_sel  = np.zeros(self.param_name_full.size, dtype=bool)
        for i, name in enumerate(self.param_name_full):
            if pconf[name]['sample']:
                self.param_sample_sel[i] = True
        
        # init param value
        self.current_param_value_full = np.array([pconf[name]['init'] for name in pconf.keys()])
    
    def update_param(self, sampling_param):
        if self.config['verbose']:
            prev = self.current_param_value_full[self.param_sample_sel]
            new  = sampling_param
            names= self.param_name_full[self.param_sample_sel]
            if np.all(prev==new):
                print('Same params are supplied.')
            else:
                print('Updated the param.')
                df = pandas.DataFrame([prev[ prev!=new], new[  prev!=new]], 
                                      columns=names[prev!=new], 
                                      index=['prev', 'new'])
                print(df)
                print('Full input param.')
                print(new)
        self.current_param_value_full[self.param_sample_sel] = sampling_param
        
    def get_current_param(self, which='full', dtype=dict):
        p = self.current_param_value_full
        # choose which parameter set you need
        if which == 'full':
            sel = np.ones(p.size, dtype=bool)
        elif which == 'sampling':
            sel = self.param_sample_sel
        # return
        if dtype == dict:
            names = self.param_name_full
            return dict(zip(names[sel], p[sel]))
        if dtype == 'array':
            return p[sel].copy()
        
    def compute_model(self):
        """
        update self.probe_model at a given model parameter set
        """
        pass
    
    def get_chi2(self):
        icov = self.covariance.get_icov()
        sig_data = self.probes_data.get_signal()
        sig_model= self.probes_model.get_signal()
        diff = sig_data-sig_model
        chi2 = np.dot(diff, np.dot(icov, diff))
        return chi2
    
    def get_lnlike(self):
        self.nlikecall += 1
        chi2 = self.get_chi2()
        lnlike = -0.5*chi2
        if self.config['verbose']:
            print('lnlike = ', lnlike)
        return lnlike
    
    def get_lnprior(self):
        vals = self.current_param_value_full[self.param_sample_sel]
        names= self.param_name_full[self.param_sample_sel]
        lnprior = 0
        for v, name in zip(vals, names):
            pconf = self.config['param'][name]
            if self.config['verbose']:
                print("prior for %s: %s" % (name,pconf))
            P = eval_lnP_v(v, pconf)
            lnprior += P
        return lnprior

    def get_boundary_for_metropolis(self):
        names = self.get_param_names_sampling()
        boundary = []
        for name in names:
            pconf = self.config['param'][name]
            vmin = pconf.get('min', None)
            vmax = pconf.get('max', None)
            boundary.append([vmin, vmax])
        return boundary
    
    def get_lnpost(self):
        lnl = self.get_lnlike()
        lnp = self.get_lnprior()
        return lnl+lnp
    
    def map_u_v(self, u, force_uniform=False):
        pconf = self.config['param']
        sel = self.param_sample_sel
        names = self.param_name_full
        for i, name in enumerate(names[sel]):
            p = pconf[name].copy()
            if force_uniform:
                p['type'] = 'U'
            u[i] = map_u_v(u[i], p)
    
    def get_param_names_sampling(self):
        sel = self.param_sample_sel
        names = self.param_name_full
        return names[sel]
    
    def get_param_names_derived(self):
        return []
    
    def get_param_names(self):
        n1 = self.get_param_names_sampling()
        n2 = self.get_param_names_derived()
        return list(n1)+list(n2)
    
    def compute_fisher_matrix(self, param_fid=None, dlnp=0.01, change_variables_dict=None):
        if param_fid is None:
            param_fid = self.get_current_param(which='sampling', dtype='array')
        if change_variables_dict is None:
            change_variables_dict = {'ln10p10As':'sigma8', 'Omde':'Omm'}
            
        # compute fiducial model
        self.update_param(param_fid)
        self.compute_model()
        model_fid = self.probes_model.get_signal()
        derived_fid = self.get_derived()
        
        # param names
        p_names       = self.get_param_names_sampling()
        derived_names = self.get_param_names_derived()
        t_names       = p_names.copy()
        for i,n in enumerate(t_names):
            t_names[i] = change_variables_dict.get(n, n)
        
        # theta
        cands = dict(zip( np.hstack([p_names, derived_names]), np.hstack([param_fid, derived_fid]) ))
        theta_fid = np.array([ cands[n] for n in t_names ])
        
        # compute model at slightly shifted param
        dmdp = []
        dtdp = []
        for i,p in enumerate(param_fid):
            param = param_fid.copy()
            name = p_names[i]
            if p != 0.0:
                param[i] = p*(1+dlnp)
            else:
                pconf = self.config['param'][name]
                if pconf['type'] == 'N':
                    # assume gaussian prior and set 0.1sigma value
                    param[i] = 0.1*pconf['sigma']
                else:
                    # else set 0.1*0.5*(max-min)
                    param[i] = 0.1*0.5*(pconf['max']-pconf['min'])
            dp = param[i] - p
            self.update_param(param)
            self.compute_model()
            model = self.probes_model.get_signal()
            dmdp.append( (model-model_fid)/dp )
            
            # derived param coeff
            derived = self.get_derived()
            cands = dict(zip( np.hstack([p_names, derived_names]), np.hstack([param, derived]) ))
            theta = np.array([ cands[n] for n in t_names ])
            dt    = theta-theta_fid
            dtdp.append(dt/dp)
            
        self.dmdp = np.array(dmdp)
            
        dpdt = np.linalg.inv(dtdp)
        
        dmdt = []
        n = len(param_fid)
        for i in range(n):
            dmdt.append(np.dot(dpdt[i,:], dmdp))
        
        # compute fisher matrix
        icov = self.covariance.get_icov()
        f = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                f[i,j] = np.dot(dmdt[i], np.dot(icov, dmdt[j]))
        self.fisher_matrix = f
        self.model_fid = model_fid
        self.param_fid = param_fid
        self.param_fid_fisher = theta_fid
        self.param_names_fisher = t_names
        
    def get_fisher_matrix(self):
        return self.fisher_matrix.copy()
    
    def get_fisher_matrix_prior(self):
        names = self.get_param_names_sampling()
        iv = []
        for i, name in enumerate(names):
            pconf = self.config['param'][name]
            if pconf['type'] == 'N':
                s = pconf['sigma']
            else:
                # 0.5*(max-min)
                s = 0.5*(pconf['max']-pconf['min'])
            iv.append(1/s**2)
        fisher_matrix_prior = np.diag(iv)
        return fisher_matrix_prior
    
    def get_fisher_MCSamples(self, label='Fisher', include_fisher_prior=True, seed=1, size=2000):
        f = self.get_fisher_matrix()
        if include_fisher_prior:
            f+= self.get_fisher_matrix_prior()
        inv_fisher_matrix = np.linalg.inv(f)
        p = self.param_fid_fisher
        np.random.seed(seed)
        sample_fisher = np.random.multivariate_normal(p, inv_fisher_matrix, size=size)
        names=self.param_names_fisher
        
        # S8
        if 'sigma8' in names and 'Omm' in names:
            names = np.append(names, 'S8')
            i_sigma8 = np.argwhere(names == 'sigma8')[0][0]
            i_Omm    = np.argwhere(names == 'Omm'   )[0][0]
            sigma8   = sample_fisher[:,i_sigma8]
            Omm      = sample_fisher[:,i_Omm]
            sel      = Omm>0.0
            S8       = -1*np.empty(sigma8.size)
            S8[sel]  = sigma8[sel]*(Omm[sel]/0.3)**0.5
            sample_fisher = np.hstack([sample_fisher, np.atleast_2d(S8).T])
        
        # ranges
        ranges = dict()
        for name in names:
            pconf = self.config['param']
            if name in pconf:
                ranges[name] = [pconf[name]['min'], pconf[name]['max']]
        
        from .chainutils import names_label_dict
        labels = [names_label_dict[name] for name in names]
        from getdist import MCSamples
        samples_fisher = MCSamples(samples=sample_fisher, names=names, labels=labels, ranges=ranges, label=label)
        return samples_fisher
    
    def prepare_gaussian_linear_model(self, param_fid=None, dlnp=0.01, x=None):
        if not hasattr(self, 'dmdp'):
            self.compute_fisher_matrix(param_fid=param_fid)
        iS= self.covariance.get_icov()
        M = self.dmdp.T # Eq. (7) of Raveri & Hu
        iM= np.dot( np.linalg.inv(np.dot(M.T, np.dot(iS, M))) , np.dot(M.T, iS)) # Eq. (8)
        
        if x is None:
            x = self.probes_data.get_signal()
        m_hat = self.model_fid
        t_hat = self.param_fid
        
        t_ML = t_hat + np.dot(iM, x-m_hat)
        iC   = np.dot(M.T, np.dot(iS, M))
        C    = np.linalg.inv(iC) # Eq. (12)
        
        # chi2 at maximum likelihood by GLM model: Eq. (11) of Raveri & Hu
        P = np.dot(M, iM)
        I = np.eye(P.shape[0])
        _ = np.dot(I-P, x-m_hat)
        chi2_ML = np.dot(_, np.dot(iS, _))
        
        # chi2 at MAP by GLM
        ## Set prior as a Gaussian
        iC_Pi = []
        t_Pi  = []
        for name in self.get_param_names_sampling():
            pconf = self.config['param'][name]
            if pconf['type'] in ['U', 'uniform', 'flat']:
                iC_Pi.append(0)
                t_Pi.append(0)
            elif pconf['type'] in ['N', 'normal', 'gaussian', 'Gaussian']:
                s = pconf['sigma']
                m = pconf['mean']
                iC_Pi.append(1/s**2)
                t_Pi.append(m)
            else:
                print('Cannot set prior in GLM for %s'%name)
                iC_Pi.append(0)
                t_Pi.append(0)
        iC_Pi = np.diag(iC_Pi)
        t_Pi = np.array(t_Pi)
        ## Eq. (13)
        C_p = np.linalg.inv(iC_Pi + iC)
        t_p = np.dot(C_p, np.dot(iC_Pi,t_Pi) + np.dot(iC, t_ML))
        ## Add the second term in the exponential in Eq. (10) 
        _ = t_p - t_ML
        chi2_MAP = chi2_ML + np.dot(_, np.dot(iC, _))
        
        self.icov_GLM_like = iC
        self.icov_GLM_pri  = iC_Pi
        self.icov_GLM_post = iC + iC_Pi
        
        return chi2_ML, chi2_MAP
    
    def estimate_effective_dof_by_GLM_from_noiseless_mock(self, dlnp=0.01, size=1000, return_samples=False):
        """
        This function can be used only when the data vector is noiseless mock.
        Never use for noisy mock or real data.
        """
        
        # First 
        param_fid = self.get_current_param(which='sampling', dtype='array')
        names = self.get_param_names_sampling()
        if 'AIA' in names:
            param_fid[np.where(names=='AIA')[0][0]] = 0.01
        chi2_ML, chi2_MAP = self.prepare_gaussian_linear_model(param_fid, dlnp=dlnp)
        
        # Noiseless
        x_mean = self.probes_data.get_signal()
        cov = self.covariance.get_cov()
        
        # Noisy datasets
        x_list = np.random.multivariate_normal(x_mean, cov, size=size)
        
        # chi2 dist from noisy datasets
        chi2_ML, chi2_MAP = [], []
        for x in x_list:
            a, b = self.prepare_gaussian_linear_model(param_fid, x=x)
            chi2_ML.append(a)
            chi2_MAP.append(b)
            
        dof_eff_ML  = np.mean(chi2_ML)
        dof_eff_MAP = np.mean(chi2_MAP)
        
        if return_samples:
            return chi2_ML, chi2_MAP, dof_eff_ML, dof_eff_MAP
        else:
            return dof_eff_ML, dof_eff_MAP
    
    
from .meascorr import dSigma_meascorr_class, wp_meascorr_class
from .gglensing import magnificationbias_class
from .gglensing import minimal_bias_class
from .cosmicshear import cosmicshear_class
from .cosmology import cosmology_class
from .linear_power import linear_darkemu_class, linear_camb_class
from .nonlinear_power import nonlinear_pyhalofit_class
    
class minimalbias_likelihood_class(likelihood_class):
    """
    This is the likelihood using minimal bias model and halofit for HSC 3x2pt.
    """
    def __init__(self, config, dataset, verbose=True):
        super().__init__(config, dataset, verbose=verbose)
        
        # instantiate basics
        mconf = self.config['model']
        # cosmology
        self.cosmology = cosmology_class()
        # linear model
        if mconf['linear_model_name'] == 'darkemu':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_darkemu_class(linconf)
            del linconf
        elif mconf['linear_model_name'] == 'camb':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_camb_class(linconf)
        else:
            raise NotImplemented
        # nonlinear model
        if mconf['nonlinear_model_name'] == 'pyhalofit':
            nlconf = {'verbose':self.config['verbose']}
            self.nonlinear_model = nonlinear_pyhalofit_class(nlconf)
            del nlconf
        else:
            raise NotImplemented
        
        # 2x2pt
        self.gal2x2ptmodel = dict()
        self.magnificationbias = dict()
        self.dSigma_meascorr = dict()
        self.wp_meascorr = dict()
        for name in self.probe_names:
            pconf = self.dataset.get_probe_config(name)
            if pconf['type'] == 'dSigma':
                # add 2x2pt and meascorr model if not added yet
                lensid   = self.dataset.get_lensid(name)
                sourceid = self.dataset.get_sourceid(name)
                if not lensid in self.gal2x2ptmodel:
                    mconf['minimalbias'].update({'verbose':self.config['verbose']})
                    mb = minimal_bias_class(mconf['minimalbias'])
                    self.gal2x2ptmodel[lensid] = mb
                # add meascorr model
                if mconf['do_meascorr']:
                    meconf = self.dataset.get_dSigma_meascorr_config(name, base=mconf['meascorr'])
                    self.dSigma_meascorr[name] = dSigma_meascorr_class(meconf)
                # add magnification bias model
                mconf['magnificationbias'].update({'verbose':self.config['verbose']})
                magconf = self.dataset.get_magnificationbias_class_config(name, base=mconf['magnificationbias'])
                self.magnificationbias[name] = magnificationbias_class(magconf)
            if pconf['type'] == 'wp':
                # add 2x2pt and meascorr model if not added yet
                lensid   = self.dataset.get_lensid(name)
                if not lensid in self.gal2x2ptmodel:
                    mconf['minimalbias'].update({'verbose':self.config['verbose']})
                    mb = minimal_bias_class(mconf['minimalbias'])
                    self.gal2x2ptmodel[lensid] = mb
                # add meascorr model
                if mconf['do_meascorr']:
                    meconf = self.dataset.get_wp_meascorr_config(name)
                    self.wp_meascorr[name] = wp_meascorr_class(meconf)
            
        # cosmic shear
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            mconf['cosmicshear'].update({'verbose':self.config['verbose']})
            csconf = self.dataset.get_cosmicshear_class_config(base=mconf['cosmicshear'])
            self.cosmicshear = cosmicshear_class(csconf)
        
        # update_param
        p = self.get_current_param(which='sampling', dtype='array')
        self.update_param(p)
        
    def update_param(self, sampling_param):
        super().update_param(sampling_param)
        pdict = self.get_current_param(which='full', dtype=dict)
        
        # set cosmology
        cparam = [pdict[name] for name in self.cosmology.get_cparam_name()]
        cparam = np.array(cparam)
        self.cosmology.set_cosmology(cparam)
        self.linear_model.set_cosmology(self.cosmology)
        self.nonlinear_model.set_cosmology(self.cosmology)
        if isinstance(self.nonlinear_model, nonlinear_pyhalofit_class):
            self.nonlinear_model.set_nuisance({'f1h':pdict.get('f1h',1)})
        for i in self.gal2x2ptmodel.keys():
            self.gal2x2ptmodel[i].set_cosmology(self.cosmology)
        for name in self.magnificationbias.keys():
            self.magnificationbias[name].set_cosmology(self.cosmology)
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            self.cosmicshear.set_cosmology(self.cosmology)
        
        # set nuisance
        # linear bias
        for i in self.gal2x2ptmodel.keys():
            b1 = pdict['b1_%d'%i]
            # Point mass in unit of Mun/h. Set 0 by default.
            if 'logMpm_%d'%i in pdict:
                Mpm = 10**pdict['logMpm_%d'%i]
            else:
                Mpm = 0.0
            self.gal2x2ptmodel[i].set_galaxy({'b1':b1, 'Mpm':Mpm})
        # alphamag
        for name in self.magnificationbias.keys():
            lensid   = self.dataset.get_lensid(name)
            alphamag = pdict['alphamag_%s'%lensid]
            sourceid   = self.dataset.get_sourceid(name)
            if self.config['model']['magnificationbias'].get('nodpz', False):
                dpz = 0.0
            else:
                dpz = pdict['dpz_%d'%sourceid]
            self.magnificationbias[name].set_nuisance({'alphamag':alphamag, 'dpz':dpz})
        # dpz, Om, wde for dSigma meascorr
        mconf = self.config['model']
        if mconf['do_meascorr']:
            for name in self.dSigma_meascorr.keys():
                sourceid   = self.dataset.get_sourceid(name)
                dpz = pdict['dpz_%d'%sourceid]
                Om = self.cosmology.get_Om()
                w0 = self.cosmology.cparam[5]
                self.dSigma_meascorr[name].set_param({'dpz':dpz, 'Om':Om, 'wde':w0})
        # Om, wde for wp meascorr
        if mconf['do_meascorr']:
            for name in self.wp_meascorr.keys():
                Om = self.cosmology.get_Om()
                w0 = self.cosmology.cparam[5]
                self.wp_meascorr[name].set_param({'Om':Om, 'wde':w0})
        del mconf
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            # dpz, AIA, etaIA for cosmic shear
            ns    = self.cosmicshear.n_source
            dpz   = np.array([pdict['dpz_%d'%i] for i in range(ns)])
            dm    = np.array([pdict['dm_%d'%i]  for i in range(ns)])
            AIA   = pdict['AIA']; etaIA = pdict['etaIA']
            self.cosmicshear.set_nuisance({'dpz':dpz, 'dm':dm, 'AIA':AIA, 'etaIA':etaIA})
        
        
    def compute_model(self):
        pdict = self.get_current_param(which='full', dtype=dict)
        # compute pktable
        ktab = np.logspace(-4, 3, 500)
        zl_min=1e-2; zl_max=10; zl_switch=1e-1; zlbin_low=50; zlbin_high=300
        ztab = np.hstack([np.logspace(np.log10(zl_min), np.log10(zl_switch)-0.0001, zlbin_low),
                          np.linspace(zl_switch,zl_max,zlbin_high)])
        pktab   = self.linear_model.get_pklin_table(ktab, ztab)
        pktabnl = self.nonlinear_model.get_pknonlin_table(ktab, ztab, pktab)
        
        # set linear growth rate to cosmic shear model
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            z2D = self.linear_model.get_z2D()
            self.cosmicshear.set_z2D(z2D)
        
        # set logarithmic growth rate to the gglensing models
        z2fz = self.linear_model.get_z2fz()
        for i in self.gal2x2ptmodel.keys():
            self.gal2x2ptmodel[i].set_z2fz(z2fz)
            
        # set power spectrum to each of gglensing models
        k = np.logspace(-4, 2, 500)
        pkdone = []
        for name in self.probe_names:
            pconf = self.dataset.get_probe_config(name)
            if pconf['type'] in ['dSigma', 'wp']:
                lensid   = self.dataset.get_lensid(name)
                if lensid not in pkdone:
                    zl_rep = pconf['zl_rep']
                    pklin = self.linear_model.get_pklin(k, zl_rep)
                    self.nonlinear_model.set_pklin(k, zl_rep, pklin)
                    pknonlin = self.nonlinear_model.get_pknonlin()
                    self.gal2x2ptmodel[lensid].set_pk(zl_rep, k, pklin, pknonlin)
                    pkdone.append(lensid)
        del pkdone
        
        # set pktable to cosmic shear model
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            self.cosmicshear.set_kzpktable(ktab, ztab, pktabnl)
        
        mconf = self.config['model']
        for name in self.probe_names:
            pconf = self.dataset.get_probe_config(name)
            probe = self.probes_model.probes[name]
            t0 = time.time()
            if pconf['type'] == 'dSigma':
                zl_rep = pconf['zl_rep']
                # photo-z, meas corr
                if mconf['do_meascorr']:
                    f_ds, f_rp = self.dSigma_meascorr[name].get_corrs(zl_rep)
                else:
                    f_ds, f_rp = 1, 1
                if mconf['minimalbias']['binave']:
                    rp = probe.get_lowedgbins(logcenmin=-np.inf, logcenmax=np.inf)
                else:
                    rp = probe.get_logcenbins(logcenmin=-np.inf, logcenmax=np.inf)
                lensid = self.dataset.get_lensid(name)
                ds = f_ds*self.gal2x2ptmodel[lensid].get_ds(f_rp*rp)
                ds+= f_ds*self.magnificationbias[name].get_ds_mag(zl_rep, f_rp*rp, ktab, ztab, pktabnl)
                # multiplicative bias
                sourceid = self.dataset.get_sourceid(name)
                ds*= (1+pdict['dm_%d'%sourceid])
                probe.set_signal(ds, logcenmin=-np.inf, logcenmax=np.inf)
            if pconf['type'] == 'wp':
                zl_rep = pconf['zl_rep']
                # meas corr
                if mconf['do_meascorr']:
                    f_pimax, f_rp = self.wp_meascorr[name].get_corrs(zl_rep)
                else:
                    f_rp, f_pimax = 1, 1
                pimax = mconf['minimalbias']['pimax_wp'] * f_pimax # measurement correction: integration limit.
                if mconf['minimalbias']['binave']:
                    rp = probe.get_lowedgbins(logcenmin=-np.inf, logcenmax=np.inf)
                else:
                    rp = probe.get_logcenbins(logcenmin=-np.inf, logcenmax=np.inf)
                lensid = self.dataset.get_lensid(name)
                wp = self.gal2x2ptmodel[lensid].get_wp(f_rp*rp, 
                                                       pimax=pimax)
                wp /= f_pimax # measurement correction: amplitude
                probe.set_signal(wp, logcenmin=-np.inf, logcenmax=np.inf)
            if pconf['type'] == 'xip':
                if mconf['cosmicshear']['binave']:
                    t = probe.get_lowedgbins(logcenmin=-np.inf, logcenmax=np.inf)
                else:
                    t = probe.get_logcenbins(logcenmin=-np.inf, logcenmax=np.inf)
                s0id, s1id = self.dataset.get_sourceid(name)
                xip = self.cosmicshear.get_xip(t, s0id, s1id)
                # multiplicative bias
                xip*= (1+pdict['dm_%d'%s0id])*(1+pdict['dm_%d'%s1id])
                # add psf
                a = pdict['alphapsf']; b = pdict['betapsf']
                xip+= probe.get_psfbias_term(a, b, which='xip', logcenmin=-np.inf, logcenmax=np.inf)
                probe.set_signal(xip, logcenmin=-np.inf, logcenmax=np.inf)
            if pconf['type'] == 'xim':
                if mconf['cosmicshear']['binave']:
                    t = probe.get_lowedgbins(logcenmin=-np.inf, logcenmax=np.inf)
                else:
                    t = probe.get_logcenbins(logcenmin=-np.inf, logcenmax=np.inf)
                s0id, s1id = self.dataset.get_sourceid(name)
                xim = self.cosmicshear.get_xim(t, s0id, s1id)
                # multiplicative bias
                xim*= (1+pdict['dm_%d'%s0id])*(1+pdict['dm_%d'%s1id])
                # add psf
                a = pdict['alphapsf']; b = pdict['betapsf']
                xim+= probe.get_psfbias_term(a, b, which='xim', logcenmin=-np.inf, logcenmax=np.inf)
                probe.set_signal(xim, logcenmin=-np.inf, logcenmax=np.inf)
            #print(name, time.time()-t0)
        if self.config['verbose']:
            sig=self.probes_model.get_signal()
            print('Signal = ', sig)
        
    def get_derived(self):
        Omm = self.cosmology.get_Om()
        sigma8 = self.linear_model.get_sigma8()
        S8 = sigma8*(Omm/0.3)**0.5
        signal = self.probes_model.get_signal()
        lnlike = self.get_lnlike()
        lnpost = self.get_lnpost()
        derived = np.hstack([Omm, sigma8, S8, signal, lnlike, lnpost])
        return derived
    
    def get_param_names_derived(self):
        names = ['Omm']
        names+= ['sigma8']
        names+= ['S8']
        for name in self.probe_names:
            n = self.probes_data.get_dof([name])
            names+= ['signal_%s_%d'%(name, i) for i in range(n)]
        names+= ['lnlike', 'lnpost']
        names = np.array(names)
        return names
    
## hod model
try:
    from dark_emulator import model_hod
    print('Imported model hod from', model_hod.__file__)
except:
    print('Failed to import model_hod : from dark_emulator import model_hod')
import time
class darkemu_x_hod_likelihood_class(likelihood_class):
    """
    This is the likelihood using dark emulator halo model and HOD model for HSC 3x2pt.
    """
    def __init__(self, config, dataset, verbose=True):
        super().__init__(config, dataset, verbose=verbose)
        
        # instantiate basics
        mconf = self.config['model']
        # cosmology
        self.cosmology = cosmology_class()
        # linear model
        if mconf['linear_model_name'] == 'darkemu':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_darkemu_class(linconf)
            del linconf
        else:
            raise NotImplemented
        # nonlinear model
        if mconf['nonlinear_model_name'] == 'pyhalofit':
            nlconf = {'verbose':self.config['verbose']}
            self.nonlinear_model = nonlinear_pyhalofit_class(nlconf)
            del nlconf
        else:
            raise NotImplemented
        
        # 2x2pt
        self.gal2x2ptmodel = dict()
        self.magnificationbias = dict()
        self.dSigma_meascorr = dict()
        self.wp_meascorr = dict()
        for name in self.probe_names:
            pconf = self.dataset.get_probe_config(name)
            if pconf['type'] == 'dSigma':
                # add 2x2pt and meascorr model if not added yet
                lensid   = self.dataset.get_lensid(name)
                sourceid = self.dataset.get_sourceid(name)
                if not lensid in self.gal2x2ptmodel:
                    mconf['hod'].update({'verbose':self.config['verbose']})
                    hod = model_hod.darkemu_x_hod()
                    self.gal2x2ptmodel[lensid] = hod
                # add meascorr model
                if mconf['do_meascorr']:
                    meconf = self.dataset.get_dSigma_meascorr_config(name, base=mconf['meascorr'])
                    self.dSigma_meascorr[name] = dSigma_meascorr_class(meconf)
                # add magnification bias model
                mconf['magnificationbias'].update({'verbose':self.config['verbose']})
                magconf = self.dataset.get_magnificationbias_class_config(name, base=mconf['magnificationbias'])
                self.magnificationbias[name] = magnificationbias_class(magconf)
            if pconf['type'] == 'wp':
                # add 2x2pt and meascorr model if not added yet
                lensid   = self.dataset.get_lensid(name)
                if not lensid in self.gal2x2ptmodel:
#                    mconf['hod'].update({'verbose':self.config['verbose']})
                    hod = model_hod.darkemu_x_hod()
                    self.gal2x2ptmodel[lensid] = hod
                # add meascorr model
                if mconf['do_meascorr']:
                    meconf = self.dataset.get_wp_meascorr_config(name)
                    self.wp_meascorr[name] = wp_meascorr_class(meconf)
            
        # cosmic shear
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            mconf['cosmicshear'].update({'verbose':self.config['verbose']})
            csconf = self.dataset.get_cosmicshear_class_config(base=mconf['cosmicshear'])
            self.cosmicshear = cosmicshear_class(csconf)
        
        # update_param
        p = self.get_current_param(which='sampling', dtype='array')
        self.update_param(p)
        
    def update_param(self, sampling_param):
        super().update_param(sampling_param)
        pdict = self.get_current_param(which='full', dtype=dict)
        
        # set cosmology
        cparam = [pdict[name] for name in self.cosmology.get_cparam_name()]
        cparam = np.array(cparam)
        self.cosmology.set_cosmology(cparam)
        self.linear_model.set_cosmology(self.cosmology)
        self.nonlinear_model.set_cosmology(self.cosmology)
        if isinstance(self.nonlinear_model, nonlinear_pyhalofit_class):
            self.nonlinear_model.set_nuisance({'f1h':pdict.get('f1h',1)})
        print("Cosmological parameters: %s" % self.cosmology.get_darkemu_cparam())
        for i in self.gal2x2ptmodel.keys():
            self.gal2x2ptmodel[i].set_cosmology(np.array([self.cosmology.get_darkemu_cparam()]))
        for name in self.magnificationbias.keys():
            self.magnificationbias[name].set_cosmology(self.cosmology)
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            self.cosmicshear.set_cosmology(self.cosmology)
        
        # set nuisance
        # galaxy parameters for hod model
        for i in self.gal2x2ptmodel.keys():
            hod_parameters = {'logMmin': pdict['logMmin_%s' % i], 
                              'sigma_sq': pdict['sigma_sq_%s' % i], 
                              'logM1': pdict['logM1_%s' % i], 
                              'alpha': pdict['alpha_%s' % i],
                              'kappa': pdict['kappa_%s' % i],
                              'poff': pdict['poff_%s' % i], 
                              'Roff': pdict['Roff_%s' % i],
                              'sat_dist_type': self.config['model']['hod']['sat_dist_type_%s' % i],
                              'alpha_inc': pdict['alpha_inc_%s' % i],
                              'logM_inc': pdict['logM_inc_%s' % i]}
            print("HOD parameters: %s" % hod_parameters)
            self.gal2x2ptmodel[i].set_galaxy(hod_parameters)
        # alphamag
        for name in self.magnificationbias.keys():
            lensid   = self.dataset.get_lensid(name)
            alphamag = pdict['alphamag_%s'%lensid]
            sourceid   = self.dataset.get_sourceid(name)
            dpz = pdict['dpz_%d'%sourceid]
            self.magnificationbias[name].set_nuisance({'alphamag':alphamag, 'dpz':dpz})
        # dpz, Om, wde for dSigma meascorr
        mconf = self.config['model']
        if mconf['do_meascorr']:
            for name in self.dSigma_meascorr.keys():
                sourceid   = self.dataset.get_sourceid(name)
                dpz = pdict['dpz_%d'%sourceid]
                Om = self.cosmology.get_Om()
                w0 = self.cosmology.cparam[5]
                self.dSigma_meascorr[name].set_param({'dpz':dpz, 'Om':Om, 'wde':w0})
        # Om, wde for wp meascorr
        if mconf['do_meascorr']:
            for name in self.wp_meascorr.keys():
                Om = self.cosmology.get_Om()
                w0 = self.cosmology.cparam[5]
                self.wp_meascorr[name].set_param({'Om':Om, 'wde':w0})
        del mconf
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            # dpz, AIA, etaIA for cosmic shear
            ns    = self.cosmicshear.n_source
            dpz   = np.array([pdict['dpz_%d'%i] for i in range(ns)])
            dm    = np.array([pdict['dm_%d'%i]  for i in range(ns)])
            AIA   = pdict['AIA']; etaIA = pdict['etaIA']
            self.cosmicshear.set_nuisance({'dpz':dpz, 'dm':dm, 'AIA':AIA, 'etaIA':etaIA})
        
    def compute_model(self):
        s = time.time()
        pdict = self.get_current_param(which='full', dtype=dict)

        # compute pktable
        ktab = np.logspace(-4, 3, 500)
        zl_min=1e-2; zl_max=10; zl_switch=1e-1; zlbin_low=50; zlbin_high=300
        ztab = np.hstack([np.logspace(np.log10(zl_min), np.log10(zl_switch)-0.0001, zlbin_low),
                          np.linspace(zl_switch,zl_max,zlbin_high)])
        pktab   = self.linear_model.get_pklin_table(ktab, ztab)
        pktabnl = self.nonlinear_model.get_pknonlin_table(ktab, ztab, pktab)
        
        # set linear growth rate to cosmic shear model
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            z2D = self.linear_model.get_z2D()
            self.cosmicshear.set_z2D(z2D)        

        # set pktable to cosmic shear model
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            self.cosmicshear.set_kzpktable(ktab, ztab, pktabnl)
        
        mconf = self.config['model']
        for name in self.probe_names:
            pconf = self.dataset.get_probe_config(name)
            probe = self.probes_model.probes[name]
            if pconf['type'] == 'dSigma':
                zl_rep = pconf['zl_rep']
                # photo-z, meas corr
                if mconf['do_meascorr']:
                    f_ds, f_rp = self.dSigma_meascorr[name].get_corrs(zl_rep)
                else:
                    f_ds, f_rp = 1, 1
                if mconf['hod']['binave']:
                    rp = probe.get_lowedgbins(logcenmin=-np.inf, logcenmax=np.inf)
                else:
                    rp = probe.get_logcenbins(logcenmin=-np.inf, logcenmax=np.inf)
                lensid = self.dataset.get_lensid(name)
                if mconf['hod']['binave']:
                    dlnR = np.log(rp[1]/rp[0])
                else:
                    dlnR = 0.
                ds = f_ds*self.gal2x2ptmodel[lensid].get_ds(f_rp*rp, zl_rep, dlnrp = dlnR)
                ds+= f_ds*self.magnificationbias[name].get_ds_mag(zl_rep, f_rp*rp, ktab, ztab, pktabnl, dlnR = dlnR)
                # multiplicative bias
                sourceid = self.dataset.get_sourceid(name)
                ds*= (1+pdict['dm_%d'%sourceid])
                probe.set_signal(ds, logcenmin=-np.inf, logcenmax=np.inf)
            if pconf['type'] == 'wp':
                zl_rep = pconf['zl_rep']
                # meas corr
                if mconf['do_meascorr']:
                    f_pimax, f_rp = self.wp_meascorr[name].get_corrs(zl_rep)
                else:
                    f_rp, f_pimax = 1, 1
                pimax = mconf['hod']['pi_max'] * f_pimax # measurement correction: integration limit.
                if mconf['hod']['binave']:
                    rp = probe.get_lowedgbins(logcenmin=-np.inf, logcenmax=np.inf)
                else:
                    rp = probe.get_logcenbins(logcenmin=-np.inf, logcenmax=np.inf)
                lensid = self.dataset.get_lensid(name)
                if mconf['hod']['binave']:
                    dlnR = np.log(rp[1]/rp[0])
                else:
                    dlnR = 0.
                wp = self.gal2x2ptmodel[lensid].get_wp(f_rp*rp, zl_rep,
                                                       pimax = pimax,
                                                       rsd = mconf['hod']['rsd'], dlnrp = dlnR)
                wp /= f_pimax # measurement correction: amplitude
                probe.set_signal(wp, logcenmin=-np.inf, logcenmax=np.inf)
            if pconf['type'] == 'xip':
                if mconf['cosmicshear']['binave']:
                    t = probe.get_lowedgbins(logcenmin=-np.inf, logcenmax=np.inf)
                else:
                    t = probe.get_logcenbins(logcenmin=-np.inf, logcenmax=np.inf)
                s0id, s1id = self.dataset.get_sourceid(name)
                xip = self.cosmicshear.get_xip(t, s0id, s1id)
                # multiplicative bias
                xip*= (1+pdict['dm_%d'%s0id])*(1+pdict['dm_%d'%s1id])
                # add psf
                a = pdict['alphapsf']; b = pdict['betapsf']
                xip+= probe.get_psfbias_term(a, b, which='xip', logcenmin=-np.inf, logcenmax=np.inf)
                probe.set_signal(xip, logcenmin=-np.inf, logcenmax=np.inf)
            if pconf['type'] == 'xim':
                if mconf['cosmicshear']['binave']:
                    t = probe.get_lowedgbins(logcenmin=-np.inf, logcenmax=np.inf)
                else:
                    t = probe.get_logcenbins(logcenmin=-np.inf, logcenmax=np.inf)
                s0id, s1id = self.dataset.get_sourceid(name)
                xim = self.cosmicshear.get_xim(t, s0id, s1id)
                # multiplicative bias
                xim*= (1+pdict['dm_%d'%s0id])*(1+pdict['dm_%d'%s1id])
                # add psf
                a = pdict['alphapsf']; b = pdict['betapsf']
                xim+= probe.get_psfbias_term(a, b, which='xim', logcenmin=-np.inf, logcenmax=np.inf)
                probe.set_signal(xim, logcenmin=-np.inf, logcenmax=np.inf)
        e = time.time()
        if self.config['verbose']:
            sig=self.probes_model.get_signal()
            print("time to compute model %s" % (e-s))
            print('Signal = ', sig)
        
    def get_derived(self):
        Omm = self.cosmology.get_Om()
        sigma8 = self.linear_model.get_sigma8()
        S8 = sigma8*(Omm/0.3)**0.5
        signal = self.probes_model.get_signal()
        lnlike = self.get_lnlike()
        lnpost = self.get_lnpost()
        derived = np.hstack([Omm, sigma8, S8, signal, lnlike, lnpost])
        return derived
    
    def get_param_names_derived(self):
        names = ['Omm']
        names+= ['sigma8']
        names+= ['S8']
        for name in self.probe_names:
            n = self.probes_data.get_dof([name])
            names+= ['signal_%s_%d'%(name, i) for i in range(n)]
        names+= ['lnlike', 'lnpost']
        names = np.array(names)
        return names

# CAMB for BAO
try:
    import camb
except:
    print('import fail: camb')
from .utils import empty_dict
class bao_camb_class:
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
        else:
            print("Got same cosmological parameters. Keep quantities already computed")
        
    
    def get_rd(self):
        results = camb.get_results(self.camb_pars)
        rd = results.get_derived_params()['rdrag']
        return rd

class sixdf_bao_class(likelihood_class):
    # https://arxiv.org/pdf/1106.3366.pdf
    def __init__(self, config, verbose=True):
        super().__init__(config, None, verbose=verbose)
        self.cosmology = cosmology_class()
        self.c = 299792.458 # speed of light in km/s

        # data
        self.z = 0.106
        self.rs_over_Dv = 0.336
        self.rs_over_Dv_err = 0.015

        # linear mode
        mconf = self.config['model']
        if mconf['linear_model_name'] == 'darkemu':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_darkemu_class(linconf)
            del linconf
        elif mconf['linear_model_name'] == 'camb':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_camb_class(linconf)
        else:
            raise NotImplemented

        # rd
        if mconf['rd'] == 'camb':
            baoconf = {'verbose':self.config['verbose']}
            self.bao_camb = bao_camb_class(baoconf)
        elif mconf['rd'] == 'margenalize':
            self.rd = 147.8 # Mpc 
        else:
            raise NotImplemented 
        
        # update_param
        p = self.get_current_param(which='sampling', dtype='array')
        self.update_param(p)

    def update_param(self, sampling_param):
        super().update_param(sampling_param)
        pdict = self.get_current_param(which='full', dtype=dict)

        # set cosmology
        cparam = [pdict[name] for name in self.cosmology.get_cparam_name()]
        cparam = np.array(cparam)
        self.cosmology.set_cosmology(cparam)
        self.linear_model.set_cosmology(self.cosmology)
        self.bao_camb.set_cosmology(self.cosmology)

        # set r_d
        mconf = self.config['model']
        if mconf['rd'] == 'camb':
            self.rd = self.bao_camb.get_rd()
            print("updated rd:", self.rd)
        elif mconf['rd'] == 'margenalize':
            self.rd = pdict['rd']

    def get_chi2(self):
        apcosmo = self.cosmology.get_astropycosmo()

        DM = apcosmo.comoving_distance(self.z).value # Mpc
        H = 100.*self.linear_model.get_hubble()*apcosmo.efunc(self.z)
        DH = self.c/H # Mpc
        Dv = (DM**2*DH*self.z)**(1./3.)
        rs_over_Dv_pred = 1.027369826*self.rd/Dv
        diff = rs_over_Dv_pred - self.rs_over_Dv
        chi2 = diff**2/self.rs_over_Dv_err**2

        return chi2
        
    def get_derived(self):
        Omm = self.cosmology.get_Om()
        sigma8 = self.linear_model.get_sigma8()
        S8 = sigma8*(Omm/0.3)**0.5
        derived = np.hstack([Omm, sigma8, S8, 1, 1])
        return derived
    
    def get_param_names_derived(self):
        names = ['Omm']
        names+= ['sigma8']
        names+= ['S8']
        names+= ['lnlike', 'lnpost']
        names = np.array(names)
        return names

class sdss_dr7_mgs_class(likelihood_class):
    # https://arxiv.org/pdf/1409.3242.pdf
    def __init__(self, config, verbose=True):
        super().__init__(config, None, verbose=verbose)
        self.cosmology = cosmology_class()
        self.c = 299792.458 # speed of light in km/s

        # data
        self.z = 0.15
        self.Dv_over_rs = 4.465666824
        self.Dv_over_rs_err = 0.1681350461

        # pdf
        chi2 = np.loadtxt(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../../external-likelihoods/sdss_MGS/sdss_MGS_prob.txt'))
        alpha_edges = [0.8005, 1.1985]
        alpha = np.linspace(alpha_edges[0], alpha_edges[1], len(chi2))
        spline = UnivariateSpline(alpha, chi2, s=0)
        self.chi2 = lambda _x: (spline(_x) if alpha_edges[0] <= _x <= alpha_edges[1] else np.inf)
        self.spline = spline

        # linear mode
        mconf = self.config['model']
        if mconf['linear_model_name'] == 'darkemu':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_darkemu_class(linconf)
            del linconf
        elif mconf['linear_model_name'] == 'camb':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_camb_class(linconf)
        else:
            raise NotImplemented

        # rd
        if mconf['rd'] == 'camb':
            baoconf = {'verbose':self.config['verbose']}
            self.bao_camb = bao_camb_class(baoconf)
        elif mconf['rd'] == 'margenalize':
            self.rd = 147.8 # Mpc 
        else:
            raise NotImplemented 

        self.rs_rescale = 4.29720761315
        
        # update_param
        p = self.get_current_param(which='sampling', dtype='array')
        self.update_param(p)

    def update_param(self, sampling_param):
        super().update_param(sampling_param)
        pdict = self.get_current_param(which='full', dtype=dict)

        # set cosmology
        cparam = [pdict[name] for name in self.cosmology.get_cparam_name()]
        cparam = np.array(cparam)
        self.cosmology.set_cosmology(cparam)
        self.linear_model.set_cosmology(self.cosmology)
        self.bao_camb.set_cosmology(self.cosmology)

        # set r_d
        mconf = self.config['model']
        if mconf['rd'] == 'camb':
            self.rd = self.bao_camb.get_rd()
            print("updated rd:", self.rd)
        elif mconf['rd'] == 'margenalize':
            self.rd = pdict['rd']

    def get_chi2(self):
        apcosmo = self.cosmology.get_astropycosmo()

        DM = apcosmo.comoving_distance(self.z).value # Mpc
        H = 100.*self.linear_model.get_hubble()*apcosmo.efunc(self.z)
        DH = self.c/H # Mpc
        Dv = (DM**2*DH*self.z)**(1./3.)
        alpha = Dv/(self.rs_rescale*self.rd)
        chi2 = self.chi2(alpha)

        return chi2
        
    def get_derived(self):
        Omm = self.cosmology.get_Om()
        sigma8 = self.linear_model.get_sigma8()
        S8 = sigma8*(Omm/0.3)**0.5
        derived = np.hstack([Omm, sigma8, S8, 1, 1])
        return derived
    
    def get_param_names_derived(self):
        names = ['Omm']
        names+= ['sigma8']
        names+= ['S8']
        names+= ['lnlike', 'lnpost']
        names = np.array(names)
        return names

class eboss_dr16_bao_dr12_lrg_class(likelihood_class):
    def __init__(self, config, verbose=True):
        super().__init__(config, None, verbose=verbose)
        self.cosmology = cosmology_class()
        self.c = 299792.458 # speed of light in km/s

        # load DR12 LRG
        fname = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../external-likelihoods/eboss_DR16_v1_1_1/BAO-only/sdss_DR12_LRG_BAO_DMDH.txt")
        d = np.genfromtxt(fname)
        self.z_DR12_LRG = d[:,0]
        self.DMDH_over_rd_DR12_LRG = d[:,1] # DM_over_rd(low-z) DH_over_rd(low-z) DM_over_rd(high-z) DH_over_rd(high-z)
        fname = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../external-likelihoods/eboss_DR16_v1_1_1/BAO-only/sdss_DR12_LRG_BAO_DMDH_covtot.txt")
        self.DMDH_over_rd_DR12_LRG_icov = np.linalg.inv(np.loadtxt(fname))

        # linear mode
        mconf = self.config['model']
        if mconf['linear_model_name'] == 'darkemu':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_darkemu_class(linconf)
            del linconf
        elif mconf['linear_model_name'] == 'camb':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_camb_class(linconf)
        else:
            raise NotImplemented

        # rd
        if mconf['rd'] == 'camb':
            baoconf = {'verbose':self.config['verbose']}
            self.bao_camb = bao_camb_class(baoconf)
        elif mconf['rd'] == 'margenalize':
            self.rd = 147.8 # Mpc 
        else:
            raise NotImplemented 
        
        # update_param
        p = self.get_current_param(which='sampling', dtype='array')
        self.update_param(p)

    def update_param(self, sampling_param):
        super().update_param(sampling_param)
        pdict = self.get_current_param(which='full', dtype=dict)

        # set cosmology
        cparam = [pdict[name] for name in self.cosmology.get_cparam_name()]
        cparam = np.array(cparam)
        self.cosmology.set_cosmology(cparam)
        self.linear_model.set_cosmology(self.cosmology)
        self.bao_camb.set_cosmology(self.cosmology)

        # set r_d
        mconf = self.config['model']
        if mconf['rd'] == 'camb':
            self.rd = self.bao_camb.get_rd()
            print("updated rd:", self.rd)
        elif mconf['rd'] == 'margenalize':
            self.rd = pdict['rd']

    def get_chi2(self):
        apcosmo = self.cosmology.get_astropycosmo()

        DM = apcosmo.comoving_distance(self.z_DR12_LRG[[0,2]]).value # Mpc
        H = 100.*self.linear_model.get_hubble()*apcosmo.efunc(self.z_DR12_LRG[[1,3]])
        DH = self.c/H # Mpc
        DMDH_over_rd_DR12_LRG_pred = np.array([DM[0]/self.rd, DH[0]/self.rd, DM[1]/self.rd, DH[1]/self.rd])
        diff = DMDH_over_rd_DR12_LRG_pred - self.DMDH_over_rd_DR12_LRG
        chi2 = np.dot(diff, np.dot(self.DMDH_over_rd_DR12_LRG_icov, diff))

        return chi2
        
    def get_derived(self):
        Omm = self.cosmology.get_Om()
        sigma8 = self.linear_model.get_sigma8()
        S8 = sigma8*(Omm/0.3)**0.5
        derived = np.hstack([Omm, sigma8, S8, 1, 1])
        return derived
    
    def get_param_names_derived(self):
        names = ['Omm']
        names+= ['sigma8']
        names+= ['S8']
        names+= ['lnlike', 'lnpost']
        names = np.array(names)
        return names

class eboss_dr16_bao_dr16_lrg_class(likelihood_class):
    def __init__(self, config, verbose=True):
        super().__init__(config, None, verbose=verbose)
        self.cosmology = cosmology_class()
        self.c = 299792.458 # speed of light in km/s

        # load eBOSS LRG
        fname = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../external-likelihoods/eboss_DR16_v1_1_1/BAO-only/sdss_DR16_LRG_BAO_DMDH.txt")
        d = np.genfromtxt(fname)
        self.z_DR16_LRG = d[:,0]
        self.DMDH_over_rd_DR16_LRG = d[:,1] # DM_over_rd DH_over_rd
        fname = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../external-likelihoods/eboss_DR16_v1_1_1/BAO-only/sdss_DR16_LRG_BAO_DMDH_covtot.txt")
        self.DMDH_over_rd_DR16_LRG_icov = np.linalg.inv(np.loadtxt(fname))

        # linear model
        mconf = self.config['model']
        if mconf['linear_model_name'] == 'darkemu':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_darkemu_class(linconf)
            del linconf
        elif mconf['linear_model_name'] == 'camb':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_camb_class(linconf)
        else:
            raise NotImplemented

        # rd
        if mconf['rd'] == 'camb':
            baoconf = {'verbose':self.config['verbose']}
            self.bao_camb = bao_camb_class(baoconf)
        elif mconf['rd'] == 'margenalize':
            self.rd = 147.8 # Mpc 
        else:
            raise NotImplemented 
        
        # update_param
        p = self.get_current_param(which='sampling', dtype='array')
        self.update_param(p)

    def update_param(self, sampling_param):
        super().update_param(sampling_param)
        pdict = self.get_current_param(which='full', dtype=dict)
        
        # set cosmology
        cparam = [pdict[name] for name in self.cosmology.get_cparam_name()]
        cparam = np.array(cparam)
        self.cosmology.set_cosmology(cparam)
        self.linear_model.set_cosmology(self.cosmology)
        self.bao_camb.set_cosmology(self.cosmology)

        # set r_d
        mconf = self.config['model']
        if mconf['rd'] == 'camb':
            self.rd = self.bao_camb.get_rd()
        elif mconf['rd'] == 'margenalize':
            self.rd = pdict['rd']

    def get_chi2(self):
        apcosmo = self.cosmology.get_astropycosmo()

        # chi2 of DR16 LRG
        print(self.z_DR16_LRG)
        DM = apcosmo.comoving_distance(self.z_DR16_LRG[0]).value # Mpc
        H = 100.*self.linear_model.get_hubble()*apcosmo.efunc(self.z_DR16_LRG[0])
        DH = self.c/H # Mpc
        DMDH_over_rd_DR16_LRG_pred = np.array([DM/self.rd, DH/self.rd])
        diff = DMDH_over_rd_DR16_LRG_pred - self.DMDH_over_rd_DR16_LRG
        chi2 = np.dot(diff, np.dot(self.DMDH_over_rd_DR16_LRG_icov, diff))
        
        return chi2
        
    def get_derived(self):
        Omm = self.cosmology.get_Om()
        sigma8 = self.linear_model.get_sigma8()
        S8 = sigma8*(Omm/0.3)**0.5
        derived = np.hstack([Omm, sigma8, S8, 1, 1])
        return derived
    
    def get_param_names_derived(self):
        names = ['Omm']
        names+= ['sigma8']
        names+= ['S8']
        names+= ['lnlike', 'lnpost']
        names = np.array(names)
        return names

class eboss_dr16_bao_elg_class(likelihood_class):
    def __init__(self, config, verbose=True):
        super().__init__(config, None, verbose=verbose)
        self.cosmology = cosmology_class()
        self.c = 299792.458 # speed of light in km/s

        # load eBOSS ELG DV likelihood
        self.z_DR16_ELG = 0.845
        fname = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../../external-likelihoods/eboss_DR16_v1_1_1/BAO-only/sdss_DR16_ELG_BAO_DVtable.txt")
        _DV_over_rd, _DV_over_rd_like = np.loadtxt(fname, unpack = True)
        self.DV_over_rd_like = ius(_DV_over_rd, _DV_over_rd_like)

        # linear model
        mconf = self.config['model']
        if mconf['linear_model_name'] == 'darkemu':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_darkemu_class(linconf)
            del linconf
        elif mconf['linear_model_name'] == 'camb':
            linconf = {'verbose':self.config['verbose']}
            self.linear_model = linear_camb_class(linconf)
        else:
            raise NotImplemented

        # rd
        if mconf['rd'] == 'camb':
            baoconf = {'verbose':self.config['verbose']}
            self.bao_camb = bao_camb_class(baoconf)
        elif mconf['rd'] == 'margenalize':
            self.rd = 147.8 # Mpc 
        else:
            raise NotImplemented 

        # update_param
        p = self.get_current_param(which='sampling', dtype='array')
        self.update_param(p)

    def update_param(self, sampling_param):
        super().update_param(sampling_param)
        pdict = self.get_current_param(which='full', dtype=dict)
        self.bao_camb.set_cosmology(self.cosmology)
        
        # set cosmology
        cparam = [pdict[name] for name in self.cosmology.get_cparam_name()]
        cparam = np.array(cparam)
        self.cosmology.set_cosmology(cparam)
        self.linear_model.set_cosmology(self.cosmology)

        # set r_d
        mconf = self.config['model']
        if mconf['rd'] == 'camb':
            self.rd = self.bao_camb.get_rd()
        elif mconf['rd'] == 'margenalize':
            self.rd = pdict['rd']

    def get_chi2(self):
        apcosmo = self.cosmology.get_astropycosmo()

        # chi2 of DR16 ELG
        DM = apcosmo.comoving_distance(self.z_DR16_ELG).value # Mpc
        H = 100.*self.linear_model.get_hubble()*apcosmo.efunc(self.z_DR16_ELG)
        DH = self.c/H # Mpc
        DV = (self.z_DR16_ELG*DH*DM**2)**(1./3.)
        like = self.DV_over_rd_like(DV/self.rd)
        chi2 = -2.*np.log(like)

        return chi2
        
    def get_derived(self):
        Omm = self.cosmology.get_Om()
        sigma8 = self.linear_model.get_sigma8()
        S8 = sigma8*(Omm/0.3)**0.5
        derived = np.hstack([Omm, sigma8, S8, 1, 1])
        return derived
    
    def get_param_names_derived(self):
        names = ['Omm']
        names+= ['sigma8']
        names+= ['S8']
        names+= ['lnlike', 'lnpost']
        names = np.array(names)
        return names
        
class prior_cosmology_class(likelihood_class):
    """
    This is the prior class.
    """
    def __init__(self, config, dataset, verbose=True):
        super().__init__(config, dataset, verbose=verbose)
        
        self.cosmology = cosmology_class()
        self.linear_model = linear_darkemu_class({'verbose':config['verbose']})
        
    def update_param(self, sampling_param):
        super().update_param(sampling_param)
        pdict = self.get_current_param(which='full', dtype=dict)
        
        # set cosmology
        cparam = [pdict[name] for name in self.cosmology.get_cparam_name()]
        cparam = np.array(cparam)
        self.cosmology.set_cosmology(cparam)
        self.linear_model.set_cosmology(self.cosmology)
        
    def get_derived(self):
        Omm = self.cosmology.get_Om()
        sigma8 = self.linear_model.get_sigma8()
        S8 = sigma8*(Omm/0.3)**0.5
        derived = np.hstack([Omm, sigma8, S8, 1, 1])
        return derived
    
    def get_param_names_derived(self):
        names = ['Omm']
        names+= ['sigma8']
        names+= ['S8']
        names+= ['lnlike', 'lnpost']
        names = np.array(names)
        return names
        
    def gen(self, nsamples, seed=1):
        generator = np.random.default_rng(seed=seed)
        
        nparam  = len(self.get_param_names_sampling())
        
        cubes = generator.uniform(0, 1, size=nparam*nsamples).reshape(nsamples, nparam)
        
        for cube in cubes:
            self.map_u_v(cube)
            
        # Append derived
        derived = []
        for i, cube in enumerate(cubes):
            self.update_param(cube)
            derived.append(self.get_derived())
        derived = np.array(derived)
            
        samples = np.hstack([cubes, derived])
            
        return samples
        
        
    
def eval_lnP_v(v, config):
    """
    evaluate P(x) at x = v.
    """
    if config['type'] in ['U', 'uniform', 'flat']:
        m, M = config['min'], config['max']
        return np.log(1.0/(M-m))
    if config['type'] in ['N', 'normal', 'gaussian', 'Gaussian']:
        m, s = config['mean'], config['sigma']
        x = (v-m)/s
        if 'min' in config and 'max' in config:
            vmin, vmax = config['min'], config['max']
            xmin, xmax = (vmin-m)/s, (vmax-m)/s
            norm = 0.5*(erf(xmax/2**0.5)-erf(xmin/2**0.5))
            lnP = -0.5*x**2-0.5*np.log(2*np.pi) - np.log(norm)
        else:
            lnP = -0.5*x**2-0.5*np.log(2*np.pi)
        return lnP
    if config['type'] in ['L', 'linear']:
        a = config['min']
        b = config['max']
        s = config['slope']
        t = (1-0.5*s*(b**2-a**2))/(b-a) # normalization
        return np.log(s*v+t)
    
def map_u_v(u, config):
    """
    Map random variable u to a derived random variable p following a prior 
    specified by config.
    Args:
        u : u \in [0,1]
        config : prior config
    """
    if config['type'] in ['U', 'uniform', 'flat']:
        m, M = config['min'], config['max']
        v = (M-m)*u + m
        return v
    if config['type'] in ['N', 'normal', 'gaussian', 'Gaussian']:
        m, s = config['mean'], config['sigma']
        if 'min' in config and 'max' in config:
            vmin, vmax = config['min'], config['max']
            vmin, vmax = (vmin-m)/s, (vmax-m)/s
            v = 2**0.5*erfinv( (1-u)*erf(vmin/2**0.5) + u*erf(vmax/2**0.5) )
        else:
            v = 2**0.5*erfinv( 2*u-1 )
        v = m + s*v
        return v
    if config['type'] in ['L', 'linear']:
        """
        P(x) = sx+t, x \in [a,b]
        Phi(u) = \int_a^v dx P(x) = sv^2/2+tv - sa^2-ta
        Phi(b) = sb^2/2+tb - sa^2-ta = 1
        """
        a = config['min']
        b = config['max']
        s = config['slope']
        t = (1-0.5*s*(b**2-a**2))/(b-a) # normalization
        return ((t**2+s**2*b**2+2*s*t*b-s*(1-u))**0.5-t)/s
        
