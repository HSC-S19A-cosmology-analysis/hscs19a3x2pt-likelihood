import numpy as np
import os
import copy

config_default_probes_mb = {'dSigma0':{'type'  :'dSigma', 
                                       'bins'  :'bin_dSigma_logcen.dat', 
                                       'signal':'signal_dSigma0.dat',
                                       'logcenmin':12, 
                                       'logcenmax':30, 
                                       'lensid':0,
                                       'sourceid':0, 
                                       'zl_rep':0.26,
                                       'fname_sumwlssigcritinvPz':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/sumwlssigcritinvPz_z0_dnnz.dat',
                                       'fname_zsbin':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/photoz_bin_dnnz.dat',
                                       'Om':0.279,
                                       'wde':-1}, 
                            'dSigma1':{'type'  :'dSigma', 
                                       'bins'  :'bin_dSigma_logcen.dat', 
                                       'signal':'signal_dSigma1.dat',
                                       'logcenmin':12, 
                                       'logcenmax':40, 
                                       'lensid':1,
                                       'sourceid':0, 
                                       'zl_rep':0.51,
                                       'fname_sumwlssigcritinvPz':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/sumwlssigcritinvPz_z1_dnnz.dat',
                                       'fname_zsbin':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/photoz_bin_dnnz.dat',
                                       'Om':0.279,
                                       'wde':-1}, 
                            'dSigma2':{'type'  :'dSigma', 
                                       'bins'  :'bin_dSigma_logcen.dat', 
                                       'signal':'signal_dSigma2.dat',
                                       'logcenmin':12, 
                                       'logcenmax':80, 
                                       'lensid':2,
                                       'sourceid':0,
                                       'zl_rep':0.63,
                                       'fname_sumwlssigcritinvPz':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/sumwlssigcritinvPz_z2_dnnz.dat',
                                       'fname_zsbin':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/photoz_bin_dnnz.dat',
                                       'Om':0.279,
                                       'wde':-1}, 
                            'xip'    :{'type'  :'xip',
                                       'bins'  :'bin_xi_logcen.dat',
                                       'signal':'signal_xip.dat',
                                       'logcenmin':10**0.9, 
                                       'logcenmax':10**1.7, 
                                       'source0id':0, 
                                       'source1id':0,
                                       'psfbias':'psf_pp_pq_qq_used.dat'}, 
                            'xim'    :{'type'  :'xim',
                                       'bins'  :'bin_xi_logcen.dat',
                                       'signal':'signal_xim.dat',
                                       'logcenmin':10**1.5, 
                                       'logcenmax':10**2.2, 
                                       'source0id':0, 
                                       'source1id':0,
                                       'psfbias':'psf_pp_pq_qq_used.dat'}, 
                            'wp0'    :{'type'  :'wp', 
                                       'bins'  :'bin_wp_logcen.dat', 
                                       'signal':'signal_wp0_100RSD.dat',
                                       'logcenmin':8, 
                                       'logcenmax':80,
                                       'lensid':0, 
                                       'zl_rep':0.26,
                                       'Om':0.279,
                                       'wde':-1}, 
                            'wp1'    :{'type'  :'wp', 
                                       'bins'  :'bin_wp_logcen.dat', 
                                       'signal':'signal_wp1_100RSD.dat',
                                       'logcenmin':8, 
                                       'logcenmax':80,
                                       'lensid':1, 
                                       'zl_rep':0.51,
                                       'Om':0.279,
                                       'wde':-1}, 
                            'wp2'    :{'type'  :'wp', 
                                       'bins'  :'bin_wp_logcen.dat', 
                                       'signal':'signal_wp2_100RSD.dat',
                                       'logcenmin':8, 
                                       'logcenmax':80,
                                       'lensid':2, 
                                       'zl_rep':0.63,
                                       'Om':0.279,
                                       'wde':-1}}

config_default_probes_hod= {'dSigma0':{'type'  :'dSigma', 
                                       'bins'  :'bin_dSigma_logcen.dat', 
                                       'signal':'signal_dSigma0.dat',
                                       'logcenmin':3, 
                                       'logcenmax':30, 
                                       'lensid':0,
                                       'sourceid':0, 
                                       'zl_rep':0.26,
                                       'fname_sumwlssigcritinvPz':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/sumwlssigcritinvPz_z0_dnnz.dat',
                                       'fname_zsbin':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/photoz_bin_dnnz.dat',
                                       'Om':0.279,
                                       'wde':-1}, 
                            'dSigma1':{'type'  :'dSigma', 
                                       'bins'  :'bin_dSigma_logcen.dat', 
                                       'signal':'signal_dSigma1.dat',
                                       'logcenmin':3, 
                                       'logcenmax':30, 
                                       'lensid':1,
                                       'sourceid':0, 
                                       'zl_rep':0.51,
                                       'fname_sumwlssigcritinvPz':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/sumwlssigcritinvPz_z1_dnnz.dat',
                                       'fname_zsbin':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/photoz_bin_dnnz.dat',
                                       'Om':0.279,
                                       'wde':-1}, 
                            'dSigma2':{'type'  :'dSigma', 
                                       'bins'  :'bin_dSigma_logcen.dat', 
                                       'signal':'signal_dSigma2.dat',
                                       'logcenmin':3, 
                                       'logcenmax':30, 
                                       'lensid':2,
                                       'sourceid':0,
                                       'zl_rep':0.63,
                                       'fname_sumwlssigcritinvPz':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/sumwlssigcritinvPz_z2_dnnz.dat',
                                       'fname_zsbin':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/photoz_bin_dnnz.dat',
                                       'Om':0.279,
                                       'wde':-1}, 
                            'xip'    :{'type'  :'xip',
                                       'bins'  :'bin_xi_logcen.dat',
                                       'signal':'signal_xip.dat',
                                       'logcenmin':10**0.9, 
                                       'logcenmax':10**1.7, 
                                       'source0id':0, 
                                       'source1id':0,
                                       'psfbias':'psf_pp_pq_qq_used.dat'}, 
                            'xim'    :{'type'  :'xim',
                                       'bins'  :'bin_xi_logcen.dat',
                                       'signal':'signal_xim.dat',
                                       'logcenmin':10**1.5, 
                                       'logcenmax':10**2.2, 
                                       'source0id':0, 
                                       'source1id':0,
                                       'psfbias':'psf_pp_pq_qq_used.dat'}, 
                            'wp0'    :{'type'  :'wp', 
                                       'bins'  :'bin_wp_logcen.dat', 
                                       'signal':'signal_wp0_100RSD.dat',
                                       'logcenmin':2, 
                                       'logcenmax':30,
                                       'lensid':0, 
                                       'zl_rep':0.26,
                                       'Om':0.279,
                                       'wde':-1}, 
                            'wp1'    :{'type'  :'wp', 
                                       'bins'  :'bin_wp_logcen.dat', 
                                       'signal':'signal_wp1_100RSD.dat',
                                       'logcenmin':2, 
                                       'logcenmax':30,
                                       'lensid':1, 
                                       'zl_rep':0.51,
                                       'Om':0.279,
                                       'wde':-1}, 
                            'wp2'    :{'type'  :'wp', 
                                       'bins'  :'bin_wp_logcen.dat', 
                                       'signal':'signal_wp2_100RSD.dat',
                                       'logcenmin':2, 
                                       'logcenmax':30,
                                       'lensid':2, 
                                       'zl_rep':0.63,
                                       'Om':0.279,
                                       'wde':-1}}

config_default_probes_mby1={'dSigma0':{'type'  :'dSigma', 
                                       'bins'  :'bin_dSigma_logcen.dat', 
                                       'signal':'signal_dSigma0.dat',
                                       'logcenmin':12, 
                                       'logcenmax':80, 
                                       'lensid':0,
                                       'sourceid':0, 
                                       'zl_rep':0.26,
                                       'fname_sumwlssigcritinvPz':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/sumwlssigcritinvPz_z0_dnnz.dat',
                                       'fname_zsbin':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/photoz_bin_dnnz.dat',
                                       'Om':0.279,
                                       'wde':-1}, 
                            'dSigma1':{'type'  :'dSigma', 
                                       'bins'  :'bin_dSigma_logcen.dat', 
                                       'signal':'signal_dSigma1.dat',
                                       'logcenmin':12, 
                                       'logcenmax':80, 
                                       'lensid':1,
                                       'sourceid':0, 
                                       'zl_rep':0.51,
                                       'fname_sumwlssigcritinvPz':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/sumwlssigcritinvPz_z1_dnnz.dat',
                                       'fname_zsbin':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/photoz_bin_dnnz.dat',
                                       'Om':0.279,
                                       'wde':-1}, 
                            'dSigma2':{'type'  :'dSigma', 
                                       'bins'  :'bin_dSigma_logcen.dat', 
                                       'signal':'signal_dSigma2.dat',
                                       'logcenmin':12, 
                                       'logcenmax':80, 
                                       'lensid':2,
                                       'sourceid':0,
                                       'zl_rep':0.63,
                                       'fname_sumwlssigcritinvPz':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/sumwlssigcritinvPz_z2_dnnz.dat',
                                       'fname_zsbin':'/lustre/work/sunao.sugiyama/hscs19a3x2pt-data-cov/datasets/photoz_bin_dnnz.dat',
                                       'Om':0.279,
                                       'wde':-1}, 
                            'xip'    :{'type'  :'xip',
                                       'bins'  :'bin_xi_logcen.dat',
                                       'signal':'signal_xip.dat',
                                       'logcenmin':10**0.9, 
                                       'logcenmax':10**1.7, 
                                       'source0id':0, 
                                       'source1id':0,
                                       'psfbias':'psf_pp_pq_qq_used.dat'}, 
                            'xim'    :{'type'  :'xim',
                                       'bins'  :'bin_xi_logcen.dat',
                                       'signal':'signal_xim.dat',
                                       'logcenmin':10**1.5, 
                                       'logcenmax':10**2.2, 
                                       'source0id':0, 
                                       'source1id':0,
                                       'psfbias':'psf_pp_pq_qq_used.dat'}, 
                            'wp0'    :{'type'  :'wp', 
                                       'bins'  :'bin_wp_logcen.dat', 
                                       'signal':'signal_wp0_100RSD.dat',
                                       'logcenmin':8, 
                                       'logcenmax':80,
                                       'lensid':0, 
                                       'zl_rep':0.26,
                                       'Om':0.279,
                                       'wde':-1}, 
                            'wp1'    :{'type'  :'wp', 
                                       'bins'  :'bin_wp_logcen.dat', 
                                       'signal':'signal_wp1_100RSD.dat',
                                       'logcenmin':8, 
                                       'logcenmax':80,
                                       'lensid':1, 
                                       'zl_rep':0.51,
                                       'Om':0.279,
                                       'wde':-1}, 
                            'wp2'    :{'type'  :'wp', 
                                       'bins'  :'bin_wp_logcen.dat', 
                                       'signal':'signal_wp2_100RSD.dat',
                                       'logcenmin':8, 
                                       'logcenmax':80,
                                       'lensid':2, 
                                       'zl_rep':0.63,
                                       'Om':0.279,
                                       'wde':-1}}


probe2latex_dict = {'dSigma0':r'$\Delta\!\Sigma_0$', 
                    'dSigma1':r'$\Delta\!\Sigma_1$', 
                    'dSigma2':r'$\Delta\!\Sigma_2$', 
                    'xip'    :r'$\xi_+$', 
                    'xim'    :r'$\xi_-$', 
                    'wp0'    :r'$w_{\rm p, 0}$', 
                    'wp1'    :r'$w_{\rm p, 1}$', 
                    'wp2'    :r'$w_{\rm p, 2}$'}


config_default_covariance =  {'fname':'covariance.dat', 
                              'names':['dSigma0', 
                                       'dSigma1', 
                                       'dSigma2', 
                                       'xip', 
                                       'xim', 
                                       'wp0', 
                                       'wp1', 
                                       'wp2'], 
                              'nbins':[30,30,30,30,30,30,30,30], 
                              'Nreal':107*13,
                              'Hartlap':True}

config_default_sample = {'nzl':['nzl_z0_100bin.dat',
                                'nzl_z1_100bin.dat',
                                'nzl_z2_100bin.dat'],
                         'nzs':['stacked_pofz_all.dat']}


class probe_class:
    def __init__(self, name, config):
        """
        Args:
            name (str): name of this probe
            probe_config (dict): config of this probe, must include the following keys.
            - type: type of this probe: xip, xim, dSigma, wp
            - bins: fname of bins
            - signal : fname of signal
            - logcenmin: lower side scale cut for the logarithmic bin
            - logcenmax: upper side scale cut for the logarithmic bin
            
            If probe type is cosmicshear, must include
            - nzs: fname of nzs
            - psfbias: list of fnames of psf terms, pp+, pp-, pq+, pq-, qq+, qq-
            
            If probe type is gglensing, must include
            - nzs: fname of nzs
            - nzl: fname of nzl
            
            If probe is clustering, must include, ...
            
        """
        self.config = config
        self.name   = name
        self.bins   = np.loadtxt(self.config['bins'])
        self.signal = np.loadtxt(self.config['signal'])
        self.load_files()
        
    def load_files(self):
        """
        loads additional files
        """
        if self.config['type'] in ['xip', 'xim']:
            try:
                self.psfbias = np.loadtxt(self.config['psfbias'])                
            except:
                print('psf bias file does not exists. set psfbias=0')
                self.psfbias = np.zeros(180).reshape(30,6)
            
    def _get_bins_mask(self, logcenmin=None, logcenmax=None):
        bins = self.bins.copy()
        sel  = np.ones(bins.size, dtype=bool)
        if logcenmin is None:
            logcenmin = self.config['logcenmin']
        if logcenmax is None:
            logcenmax = self.config['logcenmax']
        sel &= (logcenmin <= bins) & (bins <= logcenmax)
        return sel
        
    def get_logcenbins(self, logcenmin=None, logcenmax=None):
        sel = self._get_bins_mask(logcenmin=logcenmin, logcenmax=logcenmax)
        bins = self.bins.copy()
        return bins[sel]
    
    def get_lowedgbins(self, logcenmin=None, logcenmax=None):
        sel = self._get_bins_mask(logcenmin=logcenmin, logcenmax=logcenmax)
        bins = self.bins.copy()
        dlnb = np.log(bins[1]/bins[0])
        bins*= np.exp(-0.5*dlnb)
        return bins[sel]
    
    def get_volavebins(self, logcenmin=None, logcenmax=None):
        sel = self._get_bins_mask(logcenmin=logcenmin, logcenmax=logcenmax)
        bins = self.bins.copy()
        dlnb = np.log(b[1]/b[0])
        bins*= 3.0/4.0 * (np.exp(4*dlnb)-1)/(np.exp(3*dlnb)-1)*np.exp(-0.5*dlnb)
        return bins[sel]
    
    def get_signal(self, logcenmin=None, logcenmax=None):
        sel = self._get_bins_mask(logcenmin=logcenmin, logcenmax=logcenmax)
        signal = self.signal.copy()
        return signal[sel]
    
    def set_signal(self, signal, logcenmin=None, logcenmax=None):
        sel = self._get_bins_mask(logcenmin=logcenmin, logcenmax=logcenmax)
        self.signal[sel] = signal
    
    def get_dof(self, logcenmin=None, logcenmax=None):
        sel = self._get_bins_mask(logcenmin=logcenmin, logcenmax=logcenmax)
        return np.sum(sel)
    
    def get_psfbias_term(self, a, b, which='xip', logcenmin=None, logcenmax=None):
        if which == 'xip':
            pp = self.psfbias[:,0]
            pq = self.psfbias[:,2]
            qq = self.psfbias[:,4]
        elif which == 'xim':
            pp = self.psfbias[:,1]
            pq = self.psfbias[:,3]
            qq = self.psfbias[:,5]
        psfbias = a**2*pp + 2*a*b*pq + b**2*qq
        sel = self._get_bins_mask(logcenmin=logcenmin, logcenmax=logcenmax)
        return psfbias[sel]

class probes_class:
    def __init__(self, config):
        self.config = config
        self.probes = dict()
        self.probe_names = []
        for name, probe_config in self.config.items():
            self.probes[name] = probe_class(name, probe_config)
            self.probe_names.append(name)

    def _prep_get(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        """
        Args:
            probes (list): list of probe names to get
            logcenmin_dict (dict) : dict of logcenmins
            logcenmax_dict (dict) : dict of logcenmaxs
        """
        if probes is None:
            probes = self.probe_names.copy()
        if logcenmin_dict is None:
            logcenmin_dict = dict()
        if logcenmax_dict is None:
            logcenmax_dict = dict()
        return probes, logcenmin_dict, logcenmax_dict
    
    def get_logcenbins(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        probes, lmin, lmax = self._prep_get(probes, logcenmin_dict, logcenmax_dict)
        bins = []
        for name in probes:
            plmin = lmin.get(name, None)
            plmax = lmax.get(name, None)
            bins.append(self.probes[name].get_logcenbins(plmin, plmax))
        return np.hstack(bins)
    
    def get_lowedgbins(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        probes, lmin, lmax = self._prep_get(probes, logcenmin_dict, logcenmax_dict)
        bins = []
        for name in probes:
            plmin = lmin.get(name, None)
            plmax = lmax.get(name, None)
            bins.append(self.probes[name].get_lowedgbins(plmin, plmax))
        return np.hstack(bins)
    
    def get_signal(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        probes, lmin, lmax = self._prep_get(probes, logcenmin_dict, logcenmax_dict)
        signal = []
        for name in probes:
            plmin = lmin.get(name, None)
            plmax = lmax.get(name, None)
            signal.append(self.probes[name].get_signal(plmin, plmax))
        return np.hstack(signal)
    
    def set_signal(self, signals, probes, logcenmin_dict=None, logcenmax_dict=None):
        """
        signals (list): list of signal arrays
        probes  (list): list of probe's names
        """
        probes, lmin, lmax = self._prep_get(probes, logcenmin_dict, logcenmax_dict)
        for signal, name in zip(signals, probes):
            plmin = lmin.get(name, None)
            plmax = lmax.get(name, None)
            self.probes[name].set_signal(signal, plmin, plmax)
            
    def get_dof(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        probes, lmin, lmax = self._prep_get(probes, logcenmin_dict, logcenmax_dict)
        dof = 0
        for name in probes:
            plmin = lmin.get(name, None)
            plmax = lmax.get(name, None)
            dof += self.probes[name].get_dof(plmin, plmax)
        return dof
    
    def copy(self):
        return copy.deepcopy(self)

class covariance_class:
    def __init__(self, config, probes):
        """
        Args:
            config (dict): config file of this covariance, must include
            - fname: file name of the covariance
            - names: ordered list of the probe names for which covariance is calculated
            - nbins: list of nbins for each probes
            - Nreal: the number of realization
            - Hartlap: bool whether apply Hatlap factor to the inverse covariance.
        """
        self.config = config
        self.covariance_full = np.loadtxt(self.config['fname'])
        self.probes = probes
        
    def _get_idx(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        map_idx = dict()
        inc = 0
        for name, nbin in zip(self.config['names'], self.config['nbins']):
            map_idx[name] = np.arange(nbin)+inc
            inc += nbin
            
        if probes is None:
            probes = self.probes.probe_names
        if logcenmin_dict is None:
            logcenmin_dict = dict()
        if logcenmax_dict is None:
            logcenmax_dict = dict()
            
        idx = []
        for name in probes:
            plmin = logcenmin_dict.get(name, None)
            plmax = logcenmax_dict.get(name, None)
            sel = self.probes.probes[name]._get_bins_mask(plmin, plmax)
            idx.append(map_idx[name][sel])
        return np.hstack(idx)
    
    def _get_probe_sizes(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        map_idx = dict()
        inc = 0
        for name, nbin in zip(self.config['names'], self.config['nbins']):
            map_idx[name] = np.arange(nbin)+inc
            inc += nbin
            
        if probes is None:
            probes = self.probes.probe_names
        if logcenmin_dict is None:
            logcenmin_dict = dict()
        if logcenmax_dict is None:
            logcenmax_dict = dict()
            
        sizes = []
        for name in probes:
            plmin = logcenmin_dict.get(name, None)
            plmax = logcenmax_dict.get(name, None)
            sel = self.probes.probes[name]._get_bins_mask(plmin, plmax)
            sizes.append(np.sum(sel))
        return np.array(sizes)
        
    def get_cov(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        idx = self._get_idx(probes, logcenmin_dict, logcenmax_dict)
        return self.covariance_full[np.ix_(idx, idx)]
    
    def get_std(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        cov = self.get_cov(probes, logcenmin_dict, logcenmax_dict)
        return np.diag(cov)**0.5
        
    
    def get_icov(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        cov = self.get_cov(probes, logcenmin_dict, logcenmax_dict)
        icov= np.linalg.inv(cov)
        if self.config['Hartlap']:
            # https://arxiv.org/abs/astro-ph/0608064
            Nreal = self.config['Nreal']
            p = cov.shape[0]
            f_Hartlap = (Nreal-p-2)/(Nreal-1)
            icov *= f_Hartlap
        return icov
    
    def get_rcc(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        cov = self.get_cov(probes, logcenmin_dict, logcenmax_dict)
        from .statutils import cov2correlation_coeff
        rcc = cov2correlation_coeff(cov)
        return rcc
        
    def plot_rcc(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        rcc = self.get_rcc(probes, logcenmin_dict, logcenmax_dict)
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(12,10))
        ax=sns.heatmap(rcc,cmap="RdBu_r", square=True, vmin=-1, vmax=1)
        ax.invert_yaxis()
        sizes = self._get_probe_sizes(probes, logcenmin_dict, logcenmax_dict)
        
        cumsum_size = np.hstack([0, np.cumsum(sizes)])

        for v in cumsum_size:
            ax.axvline(v, color='k', alpha=0.3)
            ax.axhline(v, color='k', alpha=0.3)

        loc = cumsum_size[:-1] + sizes*0.5
        
        if probes is None:
            probes = self.probes.probe_names
        ax.set_xticks(loc)
        ax.set_xticklabels([probe2latex_dict[probe] for probe in probes])
        ax.set_yticks(loc)
        ax.set_yticklabels([probe2latex_dict[probe] for probe in probes])
        ax.tick_params(axis='both', which='both', length=0)
        return fig
    
class dataset_class:
    config_default = {'dirname':'',
                      'probes' :None,
                      'covariance':None,
                      'samples':None}
    def __init__(self, config):
        """
        Args:
            config (dict): configure must includes following keys.
            - dirname (str): dirname of dataset
            - probes (dict): config of probes
            - covariance (dict) : config of covariance
            - samples (dict): config of samples
        """
        self.config = config
        if 'abspath' in config:
            datasets_abs_path = config['abspath']
        else:
            print('config does not include `abspath` key, assumes the directry of the datasets to be ${hscs19a3x2pt}/hscs19a3x2pt-data-cov')
            datasets_abs_path = os.path.join(os.environ['hscs19a3x2pt'], 'hscs19a3x2pt-data-cov')
        # probes
        update_probe_config_file_path(datasets_abs_path, config['dirname'], config['probes'])
        self.probes = probes_class(config['probes'])
        # covariance
        update_covariance_config_file_path(datasets_abs_path, config['dirname'], config['covariance'])
        self.covariance = covariance_class(config['covariance'], self.probes)
        # samples
        update_sample_config_file_path(datasets_abs_path, config['dirname'], config['samples'])
        
    def get_covariance(self):
        return self.covariance
    
    def get_probes(self):
        return self.probes
        
    def dump(self, dirname):
        raise NotImplemented
        
    def link(self, symbname):
        """
        make a symbolic link to the dataset
        """
        raise NotImplemented
        
    def get_sn(self, probes=None, logcenmin_dict=None, logcenmax_dict=None):
        """
        get signal-to-noise ratio
        """
        icov = self.covariance.get_icov(probes, logcenmin_dict, logcenmax_dict)
        sig  = self.probes.get_signal(probes, logcenmin_dict, logcenmax_dict)
        sn = np.dot(sig, np.dot(icov, sig))**0.5
        return sn
    
    def get_cumsn(self, probes=None, logcenmin_dict=None, logcenmax_dict=None, direction=-1):
        bins = self.probes.get_logcenbins(probes, logcenmin_dict, logcenmax_dict)
        unibins = np.sort(np.unique(bins))
        sig = self.probes.get_signal(probes, logcenmin_dict, logcenmax_dict)
        cov = self.covariance.get_cov(probes, logcenmin_dict, logcenmax_dict)
        
        cum_snr = np.empty(unibins.shape)
        for i,b in enumerate(unibins):
            sel = direction*(bins-b) <= 0
            sig_sub = sig[sel]
            cov_sub = cov[np.ix_(sel, sel)]
            icov_sub= np.linalg.inv(cov_sub)
            if self.covariance.config['Hartlap']:
                # https://arxiv.org/abs/astro-ph/0608064
                Nreal = self.covariance.config['Nreal']
                p = cov_sub.shape[0] # size of data vector you use
                f_Hartlap = (Nreal-p-2)/(Nreal-1)
                icov_sub *= f_Hartlap
            snr = np.dot(sig_sub, np.dot(icov_sub, sig_sub))**0.5
            cum_snr[i] = snr
        return unibins, cum_snr
    
    def plot_cumsn(self, fig=None, plot_tot=True, plot_probe=True, ls='-', show_legend=True):
        if fig is None:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1,3, figsize=(14,4))
            plt.subplots_adjust(wspace=0.3)
        else:
            axes = fig.get_axes()
            
        # g-g lensing
        probes = ['dSigma0', 'dSigma1', 'dSigma2']
        bins_list   = []
        cumsnr_list = []
        if plot_tot:
            bins, cumsnr = self.get_cumsn(probes)
            bins_list.append(bins)
            cumsnr_list.append(cumsnr)
        if plot_probe:
            for probe in probes:
                bins, cumsnr = self.get_cumsn([probe])
                bins_list.append(bins)
                cumsnr_list.append(cumsnr)
        from .pltutils import plot_cumsnr_dSigma
        plot_cumsnr_dSigma(bins_list, cumsnr_list, axes[0], ls=ls, show_legend=show_legend)

        # clustering
        probes = ['wp0', 'wp1', 'wp2']
        bins_list   = []
        cumsnr_list = []
        if plot_tot:
            bins, cumsnr = self.get_cumsn(probes)
            bins_list.append(bins)
            cumsnr_list.append(cumsnr)
        if plot_probe:
            for probe in probes:
                bins, cumsnr = self.get_cumsn([probe])
                bins_list.append(bins)
                cumsnr_list.append(cumsnr)
        from .pltutils import plot_cumsnr_wp
        plot_cumsnr_wp(bins_list, cumsnr_list, axes[1], ls=ls, show_legend=show_legend)

        # cosmic shear
        probes = ['xip', 'xim']
        bins_list   = []
        cumsnr_list = []
        if plot_tot:
            bins, cumsnr = self.get_cumsn(probes)
            bins_list.append(bins)
            cumsnr_list.append(cumsnr)
        if plot_probe:
            for probe in probes:
                bins, cumsnr = self.get_cumsn([probe])
                bins_list.append(bins)
                cumsnr_list.append(cumsnr)
        bins2, cumsnr2 = self.get_cumsn(['wp2'])
        from .pltutils import plot_cumsnr_xipm
        plot_cumsnr_xipm(bins_list, cumsnr_list, axes[2], ls=ls, show_legend=show_legend)
        
        return fig
    
    def get_probe_config(self, probename):
        pconf    = self.config['probes']
        return pconf[probename]
    
    def get_magnificationbias_class_config(self, probename, base=None):
        if base is None:
            base2 = {}
        else:
            base2 = copy.deepcopy(base)
        pconf    = self.config['probes']
        sconf    = self.config['samples']
        lensid   = pconf[probename]['lensid']
        sourceid = pconf[probename]['sourceid']
        magconf  = {'nzl':sconf['nzl'][lensid], 'nzs':sconf['nzs'][sourceid]}
        base2.update(magconf)
        return base2
    
    def get_cosmicshear_class_config(self, base=None):
        if base is None:
            base2 = {}
        else:
            base2 = copy.deepcopy(base)
        sconf    = self.config['samples']
        csconf  = {'nzs':sconf['nzs']}
        base2.update(csconf)
        return base2
    
    def get_wp_meascorr_config(self, probename, base=None):
        if base is None:
            base2 = {}
        else:
            base2 = copy.deepcopy(base)
        pconf = self.config['probes'][probename]
        keys = ['Om', 'wde']
        for key in keys:
            base2.setdefault(key, pconf[key])
        return base2
    
    def get_dSigma_meascorr_config(self, probename, base=None):
        if base is None:
            base2 = {}
        else:
            base2 = copy.deepcopy(base)
        pconf = self.config['probes'][probename]
        keys = ['Om', 'wde', 'fname_sumwlssigcritinvPz', 'fname_zsbin']
        for key in keys:
            base2.setdefault(key, pconf[key])
        return base2
    
    def get_lensid(self, probename):
        return self.config['probes'][probename]['lensid']
    
    def get_sourceid(self, probename):
        pconf = self.config['probes'][probename]
        if pconf['type'] == 'dSigma':
            return pconf['sourceid']
        elif pconf['type'] in ['xip', 'xim']:
            return pconf['source0id'], pconf['source1id']

    def _plot_signal(self, name, ax, fmt='.', color='C0', blindy=True, shift_x=1.0, add_text=True, label=None, plot_err=True, fill_range=True, fill_color='skyblue', fill_alpha=0.2):
        p = self.get_probes()
        c = self.get_covariance()

        import matplotlib.pyplot as plt        
        ax.set_xscale('log')
        bins = p.probes[name].get_logcenbins(-np.inf, np.inf)
        sigs = p.probes[name].get_signal(-np.inf, np.inf)
        cov  = c.get_cov([name], {name:-np.inf}, {name:np.inf})
        stds = np.diag(cov)**0.5
        if plot_err:
            ax.errorbar(bins*shift_x, bins*sigs, bins*stds, fmt=fmt, color=color, label=label)
        else:
            ax.plot(bins*shift_x, bins*sigs, color=color, label=label, marker='.')
        label = {'dSigma0':'LOWZ',#r'$\Delta\!\Sigma_\mathrm{LOWZ}$', 
                 'dSigma1':'CMASS1',#r'$\Delta\!\Sigma_\mathrm{CMASS1}$', 
                 'dSigma2':'CMASS2',#r'$\Delta\!\Sigma_\mathrm{CMASS2}$', 
                 'wp0':'LOWZ',#r'$w_\mathrm{p, LOWZ}$', 
                 'wp1':'CMASS1',#r'$w_\mathrm{p, CMASS1}$', 
                 'wp2':'CMASS2',#r'$w_\mathrm{p, CMASS2}$', 
                 'xip':r'$\xi_{+}$', 
                 'xim':r'$\xi_{-}$'}[name]
        if add_text:
            ax.text(0.1, 0.1, label, transform=ax.transAxes)
        if fill_range:
            dlnb = np.diff(np.log(bins))[0]
            bins = p.probes[name].get_logcenbins()
            ax.axvspan(bins[0]*np.exp(-dlnb/2), bins[-1]*np.exp(dlnb/2), color='skyblue', alpha=fill_alpha)
            
        ax.tick_params(axis='y', which='both', length=0) if blindy else None
        plt.setp(ax.get_yticklabels(), visible=False) if blindy else None
        
    def plot_signal(self, probe_type, fmt='.', color='C0', blindy=True, fig=None, shift_x=1.0, add_text=True, label=None, show_legend=False, plot_err=True, fill_range=True, fill_color='skyblue', fill_alpha=0.2):
        """
        Plot signals whose type matches to given probe_type.
        Args:
          probe_type (str): type of probe. 'dSigma', 'wp', 'xip', or 'xim'
        """

        # Extract probes whose type matches to the given probe_type
        names = []
        for name in self.config['probes'].keys():
            if probe_type in self.config['probes'][name]['type']:
                names.append(name)

        import matplotlib.pyplot as plt
        if fig is None:
            n = len(names)
            fig, axes = plt.subplots(1,n, sharey=True, figsize=(4*n, 3))
            if n == 1:
                axes = [axes]
        else:
            axes = fig.get_axes()

        for name, ax in zip(names, axes):
            self._plot_signal(name, ax, fmt=fmt, color=color, blindy=blindy, shift_x=shift_x, add_text=add_text, label=label, plot_err=plot_err, fill_range=fill_range, fill_color=fill_color, fill_alpha=fill_alpha)
        
        if probe_type == 'dSigma':
            ax = axes[0]
            ax.set_ylabel(r'$R\Delta\!\Sigma~[10^6M_\odot/{\rm pc}]$')
            for ax in axes:
                ax.set_xlabel(r'$R~[h^{-1}{\rm Mpc}]$')
                ax.set_xlim(0.9, 110)
        elif probe_type == 'wp':
            ax = axes[0]
            ax.set_ylabel(r'$R w_{\rm p}~[h^{-1}{\rm Mpc}]$')
            for ax in axes:
                ax.set_xlabel(r'$R~[h^{-1}{\rm Mpc}]$')
                ax.set_xlim(0.9, 110)
        elif probe_type == 'xip':
            ax = axes[0]
            ax.set_ylabel(r'$\theta\xi_+~[{\rm arcmin}]$')
            for ax in axes:
                ax.set_xlabel(r'$\theta~[{\rm arcmin}]$')
                ax.set_xlim(10**0.7, 10**2.5)
        elif probe_type == 'xim':
            ax = axes[0]
            ax.set_ylabel(r'$\theta\xi_-~[{\rm arcmin}]$')
            for ax in axes:
                ax.set_xlabel(r'$\theta~[{\rm arcmin}]$')
                ax.set_xlim(10**0.7, 10**2.5)
        elif probe_type == 'xi':
            ax = axes[0]
            ax.set_ylabel(r'$\theta\xi~[{\rm arcmin}]$')
            for ax in axes:
                ax.set_xlabel(r'$\theta~[{\rm arcmin}]$')
                ax.set_xlim(10**0.7, 10**2.5)
                
        if show_legend:
            axes[-1].legend()
            
        plt.subplots_adjust(wspace=0.1)

        return fig

def update_probe_config_file_path(datasetdir, dirname, config):
    """
    This function updates path to the filenames included the config if files does not exists.
    """
    for name in config.keys():
        # define keys to replace if needed
        if config[name]['type'] in ['xip', 'xim']:
            keys = ['bins', 'signal', 'psfbias']
        elif config[name]['type'] == 'dSigma':
            keys = ['bins', 'signal', 'fname_sumwlssigcritinvPz', 'fname_zsbin']
        elif config[name]['type'] == 'wp':
            keys = ['bins', 'signal']
        # replace if needed
        for key in keys:
            config[name][key] = os.path.join(datasetdir, dirname, config[name][key])

def update_covariance_config_file_path(datasetdir, dirname, config):
    config['fname'] = os.path.join(datasetdir, dirname, config['fname'])
    
def update_sample_config_file_path(datasetdir, dirname, config):
    for key in ['nzl', 'nzs']:
        for i, fname in enumerate(config[key]):
            config[key][i] = os.path.join(datasetdir, dirname, config[key][i])
