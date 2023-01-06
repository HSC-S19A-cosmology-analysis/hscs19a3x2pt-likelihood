import getdist
import os
import yaml
import numpy as np
import pandas
import glob

names_label_dict = {'Ombh2':r'$\Omega_{\rm b}h^2$', 
                    'Omch2':r'$\Omega_{\rm cdm}h^2$', 
                    'Omde' :r'$\Omega_{\rm de}$',
                    'ln10p10As': r'$\ln 10^{10}A_{\rm s}$',
                    'ns':r'$n_{\rm s}$',
                    'w0':r'$w_0$',
                    'mnu':r'$\sum m_{\rm nu}$',
                    'wa':r'$w_a$',
                    'Omk':r'$\Omega_{K}$',
                    'b1_0':r'$b_{1,0}$',
                    'b1_1':r'$b_{1,1}$',
                    'b1_2':r'$b_{1,2}$',
                    'alphamag_0':r'$\alpha_{{\rm mag}, 0}$',
                    'alphamag_1':r'$\alpha_{{\rm mag}, 1}$',
                    'alphamag_2':r'$\alpha_{{\rm mag}, 2}$',
                    'dm_0':r'$\Delta m$',
                    'dpz_0':r'$\Delta z_{\rm ph}$',
                    'dpz_1':r'$\Delta z_{\rm ph,1}$',
                    'dpz_2':r'$\Delta z_{\rm ph,2}$',
                    'dpz_3':r'$\Delta z_{\rm ph,3}$',
                    'AIA':r'$A_{\rm IA}$', 
                    'etaIA':r'$\eta_{\rm IA}$',
                    'alphapsf':r'$\alpha_{\rm psf}$',
                    'betapsf':r'$\beta_{\rm psf}$',
                    'Omm':r'$\Omega_{\rm m}$', 
                    'sigma8':r'$\sigma_8$',
                    'S8':r'$S_8$',
                    'logMpm_0':r'$\log M_{\rm pm,0}$',
                    'logMpm_1':r'$\log M_{\rm pm,1}$',
                    'logMpm_2':r'$\log M_{\rm pm,2}$',
                    # HOD parameters
                    'logMmin_0': r'\log M_{{\rm min}}(z_{\rm LOWZ})',
                    'logMmin_1': r'\log M_{{\rm min}}(z_{\rm CMASS1})',
                    'logMmin_2': r'\log M_{{\rm min}}(z_{\rm CMASS2})',    
                    'sigma_sq_0': r'\sigma^2_{\log M}(z_{\rm LOWZ})',
                    'sigma_sq_1': r'\sigma^2_{\log M}(z_{\rm CMASS1})',
                    'sigma_sq_2': r'\sigma^2_{\log M}(z_{\rm CMASS2})',
                    'logM1_0': r'\log M_1(z_{\rm LOWZ}',
                    'logM1_1': r'\log M_1(z_{\rm CMASS1})',
                    'logM1_2': r'\log M_1(z_{\rm CMASS2})',
                    'alpha_0': r'\alpha(z_{\rm LOWZ})',
                    'alpha_1': r'\alpha(z_{\rm CMASS1})',
                    'alpha_2': r'\alpha(z_{\rm CMASS2})',
                    'kappa_0': r'\kappa(z_{\rm LOWZ})',
                    'kappa_1': r'\kappa(z_{\rm CMASS1})',
                    'kappa_2': r'\kappa(z_{\rm CMASS2})'}

names_mockinput_dict = {'Ombh2':0.02225,
                        'Omch2':0.1198,
                        'Omde' :0.6844,
                        'ln10p10As':3.094,
                        'ns':0.9645,
                        'w0':-1,
                        'mnu':0.06,
                        'wa':0.0,
                        'Omk':0.0,
                        'alphamag_0':2.26,
                        'alphamag_1':3.56,
                        'alphamag_2':3.73,
                        'dm_0':0.0,
                        'dpz_0':0.0,
                        'AIA':0.0,
                        'etaIA':0.0,
                        'alphapsf':0.0,
                        'betapsf':0.0,
                        'Omm':1.0-0.6844,
                        'sigma8':0.831,
                        'S8':0.831*((1.0-0.6844)/0.3)**0.5}

def get_MCSamples(dirname, label=None, blindby=None, sampler='MN', print_warning=True, reweight_names1=None, reweight_names2=None, rmdoll=False, append_bf=True, append_map=True, use_equal_weight=True, n_burnin=1000, sampler_blindby='MN', name_tag=None):
    """
    Args:
        dirname          (str) : directory name of MultiNest output
        use_equal_weight (bool): whether to use the equal weight output or not. equal weight sample file is smaller than weighted sample file.
        blindby          (str) : The directory of a chain to shift (blind). The chain is shifted (blinded) by mode values of the chain in `blindby`.
        sampler          (str) : name of sampler, which is MN (MultiNest) by default. Others are
                                 - MP (Metropolis Hasting) implemented by Yosuke Kobayashi.
        
        use_equal_weight (bool): Option for MN output, to specify whether to use equal weighted samples or not. MultiNest gives two chains, 
                                 one is raw output of nested sampling and another is processed one so that all the chosen samples have equal weights. 
                                 The latter chain is subsample of the former chain, and smaller number of samples and can be used for quick file 
                                 loading and github uploading etc. When something wired is found in equal weighted chain, 
                                 we should go back to the original chain.
        n_burnin         (int) : Option for MP output, to specify the number of discarded samples at the head of chain as burnin.
        dirname_mp       (str) : Option for MP, to specify where the MP output is saved under dirname.
    """
    
    if sampler == 'MP':
        dirname_mp = dirname
        dirname = os.path.dirname(dirname)
    
    
    getdist.chains.print_load_details=False
    with open(os.path.join(dirname, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    if config['blind'] and blindby is None and print_warning:
        import warnings
        warnings.simplefilter('always', UserWarning)
        blindby = dirname
        warnings.warn(f'{dirname} is a blinded analysis. If you want to unblind the chain, ignore this warning or edit the blind section in {dirname}/config.yaml')
    
    ## Read chain
    if sampler == 'MN':
        if use_equal_weight:
            if os.path.exists(os.path.join(dirname, 'mn-post_equal_weights.dat')):
                fname = os.path.join(dirname, 'mn-post_equal_weights.dat')
            else:
                fnames = glob.glob(os.path.join(dirname, '*post_equal_weights.dat'))
                fname  = fnames[0]
            samps   = np.loadtxt(fname)
            weights = None
            samps   = samps[:,:-1]
        else:
            samps   = np.loadtxt(os.path.join(dirname, 'mn-.txt'))
            # 0:weight, 1: lnlike, 2-:samples
            weights = samps[:,0]
            samps   = samps[:,2:]
    elif sampler == 'MP':
        fnames = glob.glob(os.path.join(dirname_mp, 'mp-*.txt'))
        samps = []
        for fname in fnames:
            d = np.loadtxt(fname)
            samps.append(d[n_burnin:, 1:]) # the 0th index in the second dimention is prob in Kobayashi MP.
        samps = np.vstack(samps)
        weights = None
    
    # parameter names
    names   = np.loadtxt(os.path.join(dirname, 'param_names.dat'), dtype=str)
    names   = names[:]
    
    # append BF and MAP with small weights=1e-3.
    for point_fname, append in zip(['bf-.dat', 'map-.dat'], [append_bf, append_map]):
        if os.path.exists(os.path.join(dirname, point_fname)) and append:
            bf = np.loadtxt(os.path.join(dirname, point_fname))
            samps = np.vstack([samps, bf])
            if weights is not None:
                weights = np.hstack([weights, 1e-3])

    # parameter labels
    labels  = get_predefined_better_label(names)
    if rmdoll:
        labels = [label.replace('$','') for label in labels]

    # reweight
    reweights = np.ones(samps.shape[0])
    if reweight_names1 is not None:
        for name in reweight_names1:
            if not name in names:
                continue
            i = np.where(names ==name)[0][0]
            reweights *= samps[:,i]
    if reweight_names2 is not None:
        for name in reweight_names2:
            if not name in names:
                continue
            i = np.where(names ==name)[0][0]
            reweights /= samps[:,i]
    if np.any(reweights!=1):
        if weights is None:
            weights  = reweights
        else:
            weights *= reweights
    
    ranges = dict()
    for name in names:
        pconf = config['likelihood']['param']
        if name in pconf:
            ranges[name] = np.array([pconf[name]['min'], pconf[name]['max']])
        if name == 'Omm' and 'Omde' in pconf:
            ranges[name] = np.array([1.0-pconf['Omde']['max'], 1.0-pconf['Omde']['min']])

    if isinstance(blindby, str):
        blindMCSamples = get_MCSamples(blindby, blindby=None, print_warning=False, sampler=sampler_blindby, n_burnin=n_burnin)
        
        print('Blinding cosmological parameters by shifting by modes of %s'%blindby)
        modes  = get_mode(blindMCSamples)
        names2 = get_names_from_MCSamples(blindMCSamples)

        for name in ['Ombh2', 'Omch2', 'Omde', 'ln10p10As', 'ns', 'mnu', 'wa', 'Omk', 'Omm', 'sigma8', 'S8']:
            if name in names and name in names2:
                i = np.where(names ==name)[0][0]
                j = np.where(names2==name)[0][0]
                samps[:,i] -= modes[j] # shift samples
                if name in ranges:
                    ranges[name] -= modes[j]
        
    samples = getdist.MCSamples(samples=samps,names=names,weights=weights,labels=labels,ranges=ranges,label=label, name_tag=name_tag)
    
    # Set derived or not
    for ParamInfo in samples.paramNames.names:
        if not ParamInfo.name in config['likelihood']['param']:
            ParamInfo.isDerived = True
    
    return samples

def get_predefined_better_label(names):
    labels = []
    for name in names:
        if name in names_label_dict:
            label = names_label_dict.get(name, name)
        elif 'signal_' in name:
            # this is signal as derived parameter
            # name should have the following format: signal_[*][m]_[n], where n=0,1,2,...
            n = int(name.split('_')[-1])
            if 'dSigma' in name:
                m = int(name.split('_')[-2].replace('dSigma',''))
                label = r'$\Delta\!\Sigma_{%d,%d}$'%(m,n)
            elif 'xip' in name:
                label = r'$\xi_{+,%d}$'%n
            elif 'xim' in name:
                label = r'$\xi_{-,%d}$'%n
            elif 'wp' in name:
                m = int(name.split('_')[-2].replace('wp',''))
                label = r'$w_{\rm p,%d,%d}$'%(m,n)
        else:
            # No idea for this parameter
            label = name
        labels.append(label)
    return labels
            

def cast(samples, arr, dtype):
    if dtype == 'array':
        return arr
    elif dtype == 'dict' or dtype == dict:
        names = get_names_from_MCSamples(samples)
        return dict(zip(names, arr))
    else:
        print('Cannot cast array to %s. Just returning array.'%dtype)
        return arr

# point estimate
def get_MAP(samples, dtype='array'):
    """
    get MAP from a MCSamples instance
    """
    try:
        lnpost = samples.getParams().lnpost
        i = np.argmax(lnpost)
        return cast(samples, samples.samples[i,:].copy(), dtype=dtype)
    except:
        l = samples.samples.shape[1]
        temp = np.empty(l)
        temp[:] = np.nan
        return cast(samples, temp, dtype=dtype)

def get_BF(samples, dtype='array'):
    """
    get BF(best fit) from a MCSamples instance
    """
    try:
        lnlike = samples.getParams().lnlike
        i = np.argmax(lnlike)
        return cast(samples, samples.samples[i,:].copy(), dtype=dtype)
    except:
        l = samples.samples.shape[1]
        temp = np.empty(l)
        temp[:] = np.nan
        return cast(samples, temp, dtype=dtype)
    
def get_mode(samples, dtype='array'):
    names = get_names_from_MCSamples(samples)
    modes = []
    for name in names:
        try:
            d = samples.get1DDensity(name)
            mode = d.x[np.argmax(d.P)]
        except:
            print('failed to append mode for %s.'%name)
            mode = np.nan
        modes.append(mode)
    modes = np.array(modes)
    return cast(samples, modes, dtype=dtype)

def get_mean(samples, dtype='array'):
    names = get_names_from_MCSamples(samples)
    means = []
    weights = samples.weights
    for i, name in enumerate(names):
        s = samples.samples[:,i]
        mean = np.average(s, weights=weights)
        means.append(mean)
    mean = np.array(means)
    return cast(samples, means, dtype=dtype)

# interval estimate
def _get_HDI(x, p, alphas=[0.0, 0.68, 0.95]):
    """
    Computes the highest density intervals (HDI).
    Args:
        x: x bin
        p: probability density at x
        alphas: listy of credible probability to which the HDI are computed.
    """
    argsort = np.argsort(p)
    p_sorted = p[argsort]
    x_sorted = x[argsort]

    cumsum = np.cumsum(p_sorted)
    cumsum/= cumsum[-1]
    
    HDIS = []
    for a in alphas:
        if a == 0.0:
            idx = np.argmax(p)
            HDIS.append([x[idx], x[idx]])
        else:
            idx = np.argmax(cumsum[cumsum < 1-a])
            p_height = p_sorted[idx]
            x_range  = x[p>p_height]
            hdi_min = min(x_range)
            hdi_max = max(x_range)
            HDIS.append([hdi_min, hdi_max])
        
    return HDIS

def get_HDI(samples, alphas=[0.68, 0.95], dtype='array'):
    names = get_names_from_MCSamples(samples)
    hdis_list = []
    for name in names:
        try:
            d = samples.get1DDensity(name)
            hdis = _get_HDI(d.x, d.P, alphas=alphas)
            hdis_list.append(hdis)
        except:
            print('Failed to append HDI for %s.'%name)
            hdis_list.append([[np.nan,np.nan]]*len(alphas))
    return cast(samples, np.array(hdis_list), dtype=dtype)

def get_HDIerr(samples, dtype='array'):
    hdis = get_HDI(samples, alphas=[0.68])
    hdierrs = []
    for hdi in hdis:
        hdierrs.append((hdi[0][1]-hdi[0][0])/2.0)
    return cast(samples, np.array(hdierrs), dtype=dtype)

def get_std(samples, dtype='array'):
    names = get_names_from_MCSamples(samples)
    stds = []
    weights = samples.weights
    for i, name in enumerate(names):
        s = samples.samples[:,i]
        std = np.cov(s, aweights=weights)**0.5
        stds.append(std)
    stds = np.array(stds)
    return cast(samples, stds, dtype=dtype)

def get_names_from_MCSamples(samples):
    return np.array([name.name for name in samples.getParamNames().names])


# bias
def get_1d_bias_prob(samples, name, val):
    from scipy.interpolate import interp1d
    d = samples.get1DDensity(name)
    p = interp1d(d.x, d.P)(val)
    sel = d.P>p
    pint = np.sum(d.P[sel])/np.sum(d.P)
    return pint

def get_2d_bias_prob(samples, name1, name2, val1, val2):
    from scipy.interpolate import interp2d
    d = samples.get2DDensity(name1, name2)
    p = interp2d(d.x, d.y, d.P)(val1, val2)
    sel = d.P>p
    pint = np.sum(d.P[sel])/np.sum(d.P)
    return pint

def prob2sigma(p):
    from scipy.special import erf
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    x = np.linspace(0.0, 4)
    y = erf(x/2**0.5)
    return ius(y,x)(p)


# plot marginal HDI at signal level as derived parameters
def plot_fitting_result(dirname, alphas= [0.68, 0.95], color='red', alpha=0.6, figs=None, shift_x=1.0, flag_plot_data = True, flag_plot_HDI = True, blindy=True, **kwargs_data):
    # call config, datasets, and samples
    from .configutils import config_class
    config = config_class(os.path.join(dirname, 'config.yaml'))
    dataset = config.get_dataset()

    if config.config['blind']:
        samples = get_MCSamples(dirname, blindby=dirname)
    else:
        samples = get_MCSamples(dirname)

    # compute 68%, 95% credible intervals of signals as derived parameters
    alphas = sorted(alphas)
    signals = dict()
    for a in alphas:
        signals[a] = dict(zip(['lower', 'upper'], [dataset.probes.copy(), dataset.probes.copy()]))

    for name in dataset.probes.probe_names:
        n = dataset.probes.get_dof([name])
        sigs = []
        for i in range(n):
            d = samples.get1DDensity('signal_%s_%d'%(name, i))
            hdis = _get_HDI(d.x, d.P, alphas=alphas)
            sigs.append(hdis)
        sigs = np.array(sigs)
        for i, a in enumerate(alphas):
            for j, side in enumerate(['lower' ,'upper']):
                signals[a][side].set_signal([sigs[:,i,j]], [name])

    # probe types
    probe_types = []
    names_per_type = {}
    for probe in dataset.config['probes'].keys():
        probe_type = dataset.config['probes'][probe]['type']
        if probe_type in ['xip', 'xim']:
            probe_type = 'xi'
        probe_types.append(probe_type)
        names = names_per_type.get(probe_type, []) 
        names = names + [probe]
        names_per_type[probe_type] = names
    probe_types = np.unique(probe_types)

    # init figs
    if figs is None:
        figs = dict()
    
    # plot data points
    if flag_plot_data:
        for probe_type in probe_types:
            fig = figs.get(probe_type, None)
            figs[probe_type] = dataset.plot_signal(probe_type, blindy=blindy, fmt=kwargs_data.get('fmt', '.'), color=kwargs_data.get('color_data', 'k'), fig=fig, shift_x=shift_x)

    # set parler color
    from . import pltutils
    parler_colors = pltutils.get_paler_colors(color, len(alphas))

    # plot HDI signal
    if flag_plot_HDI:
        for probe_type in probe_types:
            axes = figs[probe_type].get_axes()
            names= names_per_type[probe_type]
            for name, ax in zip(names, axes):
                for pcolor, a in zip(parler_colors, alphas[::-1]):
                    b  = signals[a]['lower'].get_logcenbins([name])
                    sl = signals[a]['lower'].get_signal([name])
                    su = signals[a]['upper'].get_signal([name])
                    ax.fill_between(b*shift_x, b*sl, b*su, color=pcolor, alpha=alpha)

    return figs

# compare result with prior
def constraint_vs_prior(dirname):
    import matplotlib.pyplot as plt
    from .likelihood import eval_lnP_v

    # load config
    with open(os.path.join(dirname, 'config.yaml')) as f:
        config = yaml.safe_load(f)
    pconf = config['likelihood']['param']

    # extract non blinded parameters and parameters with non-flat-prior
    names = []
    for name in pconf.keys():
        if name in ['Ombh2', 'Omch2', 'Omde', 'ln10p10As', 'ns', 'mnu', 'wa', 'Omk', 'Omm', 'sigma8', 'S8']:
            continue
        if pconf[name]['type'] in ['U', 'uniform', 'flat']:
            continue
        names.append(name)

    # get MCSamples
    if config['blind']:
        samples = get_MCSamples(dirname, blindby=dirname)
    else:
        samples = get_MCSamples(dirname)

    # plot
    fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 3))
    plt.subplots_adjust(wspace=0.05)
    for i, name in enumerate(names):
        d = samples.get1DDensity(name)
        ax = axes[i]
        ax.set_xlabel(names_label_dict[name])
        ax.plot(d.x, d.P, label='Estimates')
        lnp = eval_lnP_v(d.x, pconf[name])
        ax.plot(d.x, np.exp(lnp-lnp.max()), label='Priors')
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel(r'$P/P_{\rm max}$')
    axes[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(0,1))
    plt.show()


# summarizing estimation bias in table
def stat_table(samples_list, names, statnames = ['mode', 'BF', 'MAP', 'mean', 'sigma', 'std']):
    data  = []
    index = []
    for samples in samples_list:
        mode  = get_mode(samples, dtype=dict)
        BF    = get_BF(samples, dtype=dict)
        MAP   = get_MAP(samples, dtype=dict)
        mean  = get_mean(samples, dtype=dict)
        sigma = get_HDIerr(samples, dtype=dict)
        std   = get_std(samples, dtype=dict)
        hdi   = get_HDI(samples, alphas=[0.68], dtype=dict)
        data.append([])
        index.append(samples.label)
        for name in names:
            d = []
            for statname in statnames:
                if 'mode' == statname:
                    d.append(mode.get(name, np.nan))
                elif 'BF'   == statname:
                    d.append(BF.get(name, np.nan))
                elif 'MAP'  == statname:
                    d.append(MAP.get(name, np.nan))
                elif 'mean' == statname:
                    d.append(mean.get(name, np.nan))
                elif 'sigma'== statname:
                    d.append(sigma.get(name, np.nan))
                elif 'std'  == statname:
                    d.append(std.get(name, np.nan))
                elif 'MHDImin' == statname:
                    d.append(hdi.get(name, [[np.nan, np.nan]])[0][0])
                elif 'MHDImax' == statname:
                    d.append(hdi.get(name, [[np.nan, np.nan]])[0][1])
                else:
                    print('No matched stat for %s'%statname)
                    d.append(np.nan)
            data[-1].extend(d)

    col1 = []
    for name in names:
        col1 = col1 + [name]*len(statnames)
    columns = pandas.MultiIndex.from_arrays([col1, statnames*len(names)])
    df = pandas.DataFrame(data, columns=columns, index=index)

    return df

def bias_table(tab, normalize=True, true=None):
    """
    Generate a table of parameter bias.

    Definitions:
    mode : (mode - true)/(MHDI 1sigma)
    BF   : (BF   - true)/(MHDI 1sigma)
    MAP  : (MAP  - true)/(MHDI 1sigma)
    mean : (mean - true)/(MHDI 1sigma)
    """
    statnames = ['mode', 'BF', 'MAP', 'mean']
    names_base = tab.columns.get_level_values(0)
    _, idx = np.unique(names_base, return_index=True)
    if true is None:
        true = names_mockinput_dict.copy()
    
    names = []
    for name in [names_base[i] for i in sorted(idx)]:
        if name not in true:
            continue
        names.append(name)

    data = []
    for name in names:
        if normalize:
            sigma = tab[name]['sigma'].values
        else:
            sigma = 1
        mode  = (tab[name]['mode'].values - true[name])/sigma
        BF    = (tab[name]['BF'].values   - true[name])/sigma
        MAP   = (tab[name]['MAP'].values  - true[name])/sigma
        mean  = (tab[name]['mean'].values - true[name])/sigma
        data.extend([mode, BF, MAP, mean])
    data = np.transpose(data)

    col1 = []
    for name in names:
        col1 = col1 + [name]*len(statnames)
    columns = pandas.MultiIndex.from_arrays([col1, statnames*len(names)])
    df = pandas.DataFrame(data, columns=columns, index=tab.index.values)

    return df

def summary_table(samples_list, names):
    data  = []
    index = []
    for samples in samples_list:
        mode  = get_mode(samples, dtype=dict)
        HDI   = get_HDI(samples, alphas=[0.68], dtype=dict)
        MAP   = get_MAP(samples, dtype=dict)
        data.append([])
        index.append(samples.label)
        for name in names:
            lim = HDI.get(name, [np.nan, np.nan])[0]
            m   = mode.get(name, np.nan)
            text = '$%.3f_{%.3f}^{+%.3f} (%.3f)$'%(m, lim[0]-m, lim[1]-m, MAP.get(name, np.nan))
            data[-1].extend([text])
    columns = [names_label_dict.get(name, name) for name in names]

    df = pandas.DataFrame(data, columns=columns, index=index)
    
    return df

def align_table_left(tab):
    df = tab.style.set_table_styles([{'selector': 'th','props': [('text-align', 'left')]}])
    return df

def nestcheck_plot(dirnames, names, root='mn', kde='getdist', n_simulate=100, blindby=None):
    """
    Install the forked repository of nestcheck, https://github.com/git-sunao/nestcheck.git .
    This repository enables to plot posterior distirbution with getdist kde, using FFT.
    
    kde = 'getdist' or 'nestcheck'
    """
    from nestcheck import data_processing, plots
    
    if isinstance(dirnames, str):
        dirnames = [dirnames]
    
    runs = [data_processing.process_multinest_run('mn', dirname) for dirname in dirnames]
    
    # check blind or not
    blind = False
    for dirname in dirnames:
        with open(os.path.join(dirname, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        blind = blind or config['blind']
    
    if blind and blindby is None:
        dirname = dirnames[0]
        print('blindby %s'%dirname)
    elif blind:
        dirname = blindby
        print('blindby %s'%dirname)
    else:
        dirname = dirnames[0]
    
    # load MCSamples
    samples = get_MCSamples(dirname, print_warning=False)
    
    HDIe = get_HDIerr(samples, dtype=dict)
    MODE = get_mode(samples, dtype=dict)
    
    # Set range
    ranges = []
    for name in names:
        lower = samples.ranges.getLower(name)
        upper = samples.ranges.getUpper(name)
        if lower is None:
            lower = MODE[name] - 4*HDIe[name]
        if upper is None:
            upper = MODE[name] + 4*HDIe[name]
        ranges.append([lower, upper])
            
    # Get referene name array
    names_ref = np.loadtxt(os.path.join(dirname, 'param_names.dat'), dtype=str)
            
    def get_i(name):
        return np.where(names_ref==name)[0][0]
    
    # Shift ranges
    if blind:
        for j, name in enumerate(names):
            ranges[j][0] -= MODE[name]
            ranges[j][1] -= MODE[name]
            for run in runs:
                run['theta'][:,get_i(name)] -= MODE[name]
            
    # Nest check
    fthetas     = [eval('lambda x: x[:,%d]'%get_i(name)) for name in names]
    labels      = [names_label_dict.get(name, name) for name in names]
    fig = plots.param_logx_diagram(runs, 
                                   fthetas=fthetas, 
                                   ftheta_lims=ranges,
                                   labels=labels, 
                                   parallel=None, 
                                   n_simulate=n_simulate, 
                                   kde=kde)
    
    return fig

class mcmc_tester:
    def __init__(self, chains):
        """
        chains : list of chains. Each chain must be 1d array. 
                 Chain is after burnin. All chains must have the same length.
        """
        self.chains = chains
        self.m = len(chains)
        self.n = chains[0].size
        assert self.m > 0, 'm must be >0'
        assert self.n > 0, 'n must be >0'
        
    def autocorr(self, j, kmax=100):
        """
        j : index of a chain to test
        """
        chain = self.chains[j]
        n = self.n
        k = np.arange(kmax, dtype=int)
        mean = np.mean(chain)
        var  = np.sum((chain-mean)**2)
        rho  = np.array([np.sum((chain[:n-_k]-mean)*(chain[_k:]-mean)) for _k in k])/var
        return k, rho
        
    def GelmanRubin(self):
        n, m = self.n, self.m
        # within-chains variance
        w = 0
        means = []
        for chain in self.chains:
            mean = np.mean(chain)
            v = np.sum((chain-mean)**2)
            w+= v
            means.append(mean)
        w/= (m-1)*n
        # between-chain variance
        b = np.sum((np.array(mean)-np.mean(means))**2) * n/(m-1)
        # v
        v = (n-1)/n*w + b/n
        # Potential scale reduction factor (PSRF)
        r = (v/w)**0.5
        return r
        
def test_KobayashiMetropolis_GelmanRubin(dirname, n_burnin=1000):
    """
    dirname : directory name where chanins are saved.
    """
    # Load chains
    fnames = glob.glob(os.path.join(dirname, 'mp-_*'))
    chains = []
    n = np.inf
    for fname in fnames:
        chain = np.loadtxt(fname)
        ni = chain.shape[0]
        if ni > n_burnin:
            chains.append(chain)
            n = ni if ni < n else n
        
    names = np.loadtxt(os.path.join(os.path.dirname(dirname),'param_names.dat'), dtype=str)
    
    r = []
    for i, name in enumerate(names):
        subchains = [chain[n_burnin:n,i] for chain in chains]
        mct = mcmc_tester(subchains)
        
        r.append(mct.GelmanRubin())
    r = pandas.DataFrame(r, index=names)
    return r

def plot_mcmc_chain_evolution(dirname, i=1):
    chains = []
    fnames = glob.glob(os.path.join(dirname, 'mp-*.txt'))
    
    for fname in fnames:
        d = np.loadtxt(fname)
        chains.append(d)
        
    names = np.loadtxt(os.path.join(os.path.dirname(dirname),'param_names.dat'), dtype=str)
    name  = names[i]
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.ylabel(names_label_dict.get(name, name))
    for chain in chains:
        plt.plot(chain[:,i], color='gray', lw=0.5, alpha=0.3)
    plt.show()

def write_stat_table(dirname, blindby=None):
    samples = get_MCSamples(dirname, blindby=blindby)
    tab = stat_table([samples], names=["Omm", "sigma8", "S8", "lnlike", "lnpost"])
    with open(os.path.join(dirname, 'stat_table.csv'), 'w') as f:
        if blindby is not None:
            f.write('# blinded by %s'%blindby)
        tab.to_csv(f)
        
def load_stat_table(dirname):
    df = pandas.read_csv(os.path.join(dirname, 'stat_table.csv'), header=[0, 1])#, skipinitialspace=True)
    return df

def write_GLM_dof(dirname, verbose=True):
    """
    This analysis need to be run with noiseless mock. 
    Otherwise you will double-count the statistical scatter for the calculation of the effective dof.
    """
    from . import configutils
    configfname = os.path.join(dirname,'config.yaml')
    
    config = configutils.config_class(configfname)
    like = config.get_like(verbose=verbose)
    edof_ML, edof_MAP = like.estimate_effective_dof_by_GLM_from_noiseless_mock(dlnp=0.1)
    
    dof_data = like.probes_data.get_dof()
    dof_param= like.get_param_names_sampling().size
    
    data = np.array([edof_MAP, edof_ML, dof_data, dof_param])
    np.savetxt(os.path.join(dirname, 'GLMdof.dat'), data, header='0:effective dof @ MAP, 1:effective dof @ ML, 2:dof of data, 3:dof of param')
    
def load_GLM_dof(dirname):
    return np.loadtxt(os.path.join(dirname, 'GLMdof.dat'))


class ChainListManager(dict):
    """
    Manages the array of MCSamples of getdist.
    """
    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)
    
    def len(self):
        return len(self)
    
    def set_MCSamples(self, mcsamples, name_tag=None):
        if name_tag is None and hasattr(mcsamples, 'name_tag'):
            name_tag = mcsamples.name_tag
        else:
            name_tag = str(len(self))
        self[name_tag] = mcsamples
        
    def list(self):
        return list(self.values())
    
    def keys(self):
        return list(super().keys())
    