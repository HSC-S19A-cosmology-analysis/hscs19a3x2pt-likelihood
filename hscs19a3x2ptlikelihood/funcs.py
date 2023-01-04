"""
This module includes various functions.
"""

from .configutils import config_class
import numpy as np
import os
from .chainutils import get_MCSamples, get_HDI

def plot_fitting_result(dirname):
    returns = {'fig':{}}
    
    # call config, datasets, and samples
    config = config_class(os.path.join(dirname, 'config.yaml'))
    dataset = config.get_dataset()
    samples = get_MCSamples(dirname)

    # compute 68%, 95% credible intervals of signals as derived parameters
    alphas= [0.68, 0.95]
    signals = dict()
    for alpha in alphas:
        signals[alpha] = dict(zip(['lower', 'upper'], [dataset.probes.copy(), dataset.probes.copy()]))

    for name in dataset.probes.probe_names:
        n = dataset.probes.get_dof([name])
        sigs = []
        for i in range(n):
            d = samples.get1DDensity('signal_%s_%d'%(name, i))
            hdis = get_HDI(d.x, d.P, alphas=alphas)
            sigs.append(hdis)
        sigs = np.array(sigs)   
        for i, alpha in enumerate(alphas):
            for j, side in enumerate(['lower' ,'upper']):
                signals[alpha][side].set_signal([sigs[:,i,j]], [name])
    
    # blind plot or not
    blind = config.config['blind']
    
    # plot dSigma
    try:
        fig, axes = dataset.plot_signal(['dSigma0', 'dSigma1', 'dSigma2'], blind=blind)
        for alpha in alphas:
            for i in range(3):
                ax = axes[i]
                b = signals[alpha]['lower'].get_logcenbins(['dSigma%d'%i])
                sl = signals[alpha]['lower'].get_signal(['dSigma%d'%i])
                su = signals[alpha]['upper'].get_signal(['dSigma%d'%i])
                ax.fill_between(b, b*sl, b*su, color='C1', alpha=0.3)
        returns['fig']['dSigma'] = fig
    except:
        print('No dSigma signals in this analysis.')
            
    # plot wp
    try:
        fig, axes = dataset.plot_signal(['wp0', 'wp1', 'wp2'], blind=blind)
        for alpha in alphas:
            for i in range(3):
                ax = axes[i]
                b = signals[alpha]['lower'].get_logcenbins(['wp%d'%i])
                sl = signals[alpha]['lower'].get_signal(['wp%d'%i])
                su = signals[alpha]['upper'].get_signal(['wp%d'%i])
                ax.fill_between(b, b*sl, b*su, color='C1', alpha=0.3)
        returns['fig']['wp'] = fig
    except:
        print('No wp signals in this analysis.')
            
    # plot xipm
    try:
        fig, axes = dataset.plot_signal(['xip', 'xim'], blind=blind)
        for alpha in alphas:
            for i in range(2):
                ax = axes[i]
                b = signals[alpha]['lower'].get_logcenbins([['xip', 'xim'][i]])
                sl = signals[alpha]['lower'].get_signal([['xip', 'xim'][i]])
                su = signals[alpha]['upper'].get_signal([['xip', 'xim'][i]])
                ax.fill_between(b, b*sl, b*su, color='C1', alpha=0.3)
        returns['fig']['xi'] = fig
    except:
        print('No xipm signals in this analysis.')
    
    
    returns['config'] = config
    returns['dataset'] = dataset
    returns['samples'] = samples

    return returns