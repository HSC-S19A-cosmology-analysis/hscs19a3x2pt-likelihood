import numpy as np
import copy
import pandas
from .utils import setdefault_config
from scipy.special import erfinv, erf
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
        
