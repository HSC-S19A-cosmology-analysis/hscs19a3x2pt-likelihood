## hod model
from .likelihood_base import *
from .meascorr import dSigma_meascorr_class, wp_meascorr_class
from .gglensing import magnificationbias_class
from .gglensing import minimal_bias_class
from .cosmicshear import cosmicshear_class
from .cosmology import cosmology_class
from .linear_power import linear_darkemu_class, linear_camb_class
from .nonlinear_power import nonlinear_pyhalofit_class
try:
    from dark_emulator import model_hod
    print('Imported model hod from', model_hod.__file__)
except:
    print('Failed to import model_hod : from dark_emulator import model_hod')
import time
try:
    from DarkEmuPowerRSD import pkmu_nn, pkmu_hod
    print('Imported rsd emulator')
except:
    print('Failed to import rsd model.')

class wlxrsd_likelihood_class(likelihood_class):
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
        
        # g-g lensing
        self.gglensingmodel = dict()
        self.magnificationbias = dict()
        self.dSigma_meascorr = dict()
        for name in self.probe_names:
            pconf = self.dataset.get_probe_config(name)
            if pconf['type'] == 'dSigma':
                # add 2x2pt and meascorr model if not added yet
                lensid   = self.dataset.get_lensid(name)
                sourceid = self.dataset.get_sourceid(name)
                if not lensid in self.gglensingmodel:
                    mconf['hod'].update({'verbose':self.config['verbose']})
                    hod = model_hod.darkemu_x_hod()
                    self.gglensingmodel[lensid] = hod
                # add meascorr model
                if mconf['do_meascorr']:
                    meconf = self.dataset.get_dSigma_meascorr_config(name, base=mconf['meascorr'])
                    self.dSigma_meascorr[name] = dSigma_meascorr_class(meconf)
                # add magnification bias model
                mconf['magnificationbias'].update({'verbose':self.config['verbose']})
                magconf = self.dataset.get_magnificationbias_class_config(name, base=mconf['magnificationbias'])
                self.magnificationbias[name] = magnificationbias_class(magconf)
            
        # cosmic shear
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            mconf['cosmicshear'].update({'verbose':self.config['verbose']})
            csconf = self.dataset.get_cosmicshear_class_config(base=mconf['cosmicshear'])
            self.cosmicshear = cosmicshear_class(csconf)
            
        # P_l(k): rsd power spectrum
        self.rsdmodel = dict()
        self.pkrsd_meascorr = dict()
        for name in self.probe_names:
            pconf = self.dataset.get_probe_config(name)
            if pconf['type'] == 'pkrsd':
                lensid   = self.dataset.get_lensid(name)
                if not lensid in self.rsdmodel:
                    pgg = pkmu_hod()
                    self.rsdmodel[lensid] = pgg
                # add meascorr model
                if mconf['do_meascorr']:
                    meconf = self.dataset.get_pkrsd_meascorr_config(name)
                    self.pkrsd_meascorr[name] = pkrsd_meascorr_class(meconf)
        
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
        cparam = np.array([self.cosmology.get_darkemu_cparam()])
        for i in self.gglensingmodel.keys():
            self.gglensingmodel[i].set_cosmology(cparam)
        for i in self.rsdmodel.keys():
            name = 'pkrsd_mono_{}'.format(i)
            pconf = self.dataset.get_probe_config(name)
            redshift = pconf['zl_rep']
            self.rsdmodel[i].set_cosmology(cparam, redshift)
        for name in self.magnificationbias.keys():
            self.magnificationbias[name].set_cosmology(self.cosmology)
        if ('xip' in self.probe_names) or ('xim' in self.probe_names):
            self.cosmicshear.set_cosmology(self.cosmology)
        
        # set nuisance
        # galaxy parameters for hod model for gglensing
        for i in self.gglensingmodel.keys():
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
            print("HOD parameters for GGlens: %s" % hod_parameters)
            self.gglensingmodel[i].set_galaxy(hod_parameters)
        # galaxy parameters for rsd model
        for i in self.rsdmodel.keys():
            hod_parameters = {'logMmin': pdict['logMmin_%s' % i], 
                              'sigma_sq': pdict['sigma_sq_%s' % i], 
                              'logM1': pdict['logM1_%s' % i], 
                              'alpha': pdict['alpha_%s' % i],
                              'kappa': pdict['kappa_%s' % i],
                              'poff': pdict['poff_%s' % i], 
                              'Roff': pdict['Roff_%s' % i],
                              'alpha_inc': pdict['alpha_inc_%s' % i],
                              'logM_inc': pdict['logM_inc_%s' % i], 
                              'cM_fac': pdict['cM_fac_%s' % i], 
                              'sigv_fac': pdict['sigv_fac_%s' % i], 
                              'P_shot': pdict['P_shot_%s' % i]}
            print("HOD parameters for RSD: %s" % hod_parameters)
            self.rsdmodel[i].set_galaxy(hod_parameters)
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
        # Om, wde for Alcock-Paczy´nski effect correction
        if mconf['do_meascorr']:
            for name in self.wp_meascorr.keys():
                Om = self.cosmology.get_Om()
                w0 = self.cosmology.cparam[5]
                self.pkrsd_meascorr[name].set_param({'Om':Om, 'wde':w0})
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
                ds = f_ds*self.gglensingmodel[lensid].get_ds(f_rp*rp, zl_rep, dlnrp = dlnR)
                ds+= f_ds*self.magnificationbias[name].get_ds_mag(zl_rep, f_rp*rp, ktab, ztab, pktabnl, dlnR = dlnR)
                # multiplicative bias
                sourceid = self.dataset.get_sourceid(name)
                ds*= (1+pdict['dm_%d'%sourceid])
                probe.set_signal(ds, logcenmin=-np.inf, logcenmax=np.inf)
            if pconf['type'] == 'pkrsd':
                zl_rep = pconf['zl_rep']
                l = pconf['multipole_l']
                # meas corr
                if mconf['do_meascorr']:
                    alpha_perp, alpha_para = self.pkrsd_meascorr[name].get_corrs(zl_rep)
                else:
                    alpha_perp, alpha_para = 1, 1
                if mconf['hod']['binave']:
                    kref = probe.get_lowedgbins(logcenmin=-np.inf, logcenmax=np.inf)
                else:
                    kref = probe.get_logcenbins(logcenmin=-np.inf, logcenmax=np.inf)
                lensid = self.dataset.get_lensid(name)
                pkrsd = self.rsdmodel[lensid].get_pl_gg_ref(l, kref, alpha_perp, alpha_para)
                probe.set_signal(pkrsd, logcenmin=-np.inf, logcenmax=np.inf)
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