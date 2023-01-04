import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.integrate import simps
from cycler import cycler

# flag to blind plots
blind_plot_default=True


# useful class
class figaxes_class:
    def __init__(self, fig, axes):
        self.fig = fig
        self.axes= axes
    
    def export(self, fname):
        self.fig.savefig(fname, bbox_inches='tight')
        
# signal plot
def plot_dSigma_3zbins(R, dSigma_list, dSigma_std_list, blind_plot=None, config=None, figaxes=None, labels=['LOWZ', 'CMASS1', 'CMASS2']):
    if blind_plot is None:
        blind_plot = blind_plot_default
    if figaxes is None:
        n = len(dSigma_list)
        fig, axes = plt.subplots(1,n,sharey=True, figsize=(12, 3))
        figaxes = figaxes_class(fig, axes)
    fig, axes = figaxes.fig, figaxes.axes
    plt.subplots_adjust(wspace=0.05)

    for zbin, ax in enumerate(axes):
        ax.set_xscale('log')
        ax.errorbar(R, R*dSigma_list[zbin], R*dSigma_std_list[zbin], fmt='.')
        ax.text(0.1, 0.1, labels[zbin], transform=ax.transAxes)
        ax.set_xlabel(r'$R~[h^{-1}{\rm Mpc}]$')
        ax.set_ylabel(r'$R\Delta\!\Sigma~[10^6M_\odot/{\rm pc}]$') if zbin==0 else None
        ax.set_xlim(0.9, 110)
        ax.tick_params(axis='both', which='both', length=0)
    plt.setp(axes[0].get_yticklabels(), visible=False) if blind_plot else None

    if config is not None:
        for i, ax in enumerate(axes):
            ax.axvspan(1e-3, config['dSigma%d'%i]['logcenmin'], color='gray', alpha=0.3)
            ax.axvspan(config['dSigma%d'%i]['logcenmax'], 1e3 , color='gray', alpha=0.3)
    
    return figaxes
    

def plot_xipm(theta, xipm_list, xipm_std_list, blind_plot=None, config=None, figaxes=None, labels=[r'$\xi_+$', r'$\xi_-$']):
    if blind_plot is None:
        blind_plot = blind_plot_default
    if figaxes is None:
        n = len(xipm_list)
        fig, axes = plt.subplots(1,n,sharey=True, figsize=(8, 3))
        figaxes = figaxes_class(fig, axes)
    fig, axes = figaxes.fig, figaxes.axes
    plt.subplots_adjust(wspace=0.05)

    for pm, ax in enumerate(axes):
        ax.set_xscale('log')
        ax.errorbar(theta, theta*xipm_list[pm]*1e4, theta*xipm_std_list[pm]*1e4, fmt='.')
        ax.text(0.1, 0.1, labels[pm], transform=ax.transAxes)
        ax.set_xlabel(r'$\theta~[{\rm arcmin}]$')
        ax.set_ylabel(r'$\theta\xi_\pm~[10^4{\rm arcmin}]$') if pm==0 else None
        ax.set_xlim(10**0.7, 10**2.5)
        ax.tick_params(axis='both', which='both', length=0)
    plt.setp(axes[0].get_yticklabels(), visible=False) if blind_plot else None

    if config is not None:
        for i, ax in enumerate(axes):
            ax.axvspan(1e-3, config[['xip','xim'][i]]['logcenmin'], color='gray', alpha=0.3)
            ax.axvspan(config[['xip','xim'][i]]['logcenmax'], 1e3 , color='gray', alpha=0.3)
            
    return figaxes
    
def plot_wp_3zbins(R, wp_list, wp_std_list, blind_plot=None, config=None, figaxes=None, labels=['LOWZ', 'CMASS1', 'CMASS2']):
    if blind_plot is None:
        blind_plot = blind_plot_default
    if figaxes is None:
        n = len(wp_list)
        fig, axes = plt.subplots(1,n,sharey=True, figsize=(12, 3))
        figaxes = figaxes_class(fig, axes)
    fig, axes = figaxes.fig, figaxes.axes
    plt.subplots_adjust(wspace=0.05)

    for zbin, ax in enumerate(axes):
        ax.set_xscale('log')
        ax.errorbar(R, R*wp_list[zbin], R*wp_std_list[zbin], fmt='.')
        ax.text(0.1, 0.1, labels[zbin], transform=ax.transAxes)
        ax.set_xlabel(r'$R~[h^{-1}{\rm Mpc}]$')
        ax.set_ylabel(r'$Rw_{\rm p}~[h^{-1}{\rm Mpc}]$') if zbin==0 else None
        ax.set_xlim(0.9, 110)
        ax.tick_params(axis='both', which='both', length=0)
    plt.setp(axes[0].get_yticklabels(), visible=False) if blind_plot else None

    if config is not None:
        for i, ax in enumerate(axes):
            ax.axvspan(1e-3, config['wp%d'%i]['logcenmin'], color='gray', alpha=0.3)
            ax.axvspan(config['wp%d'%i]['logcenmax'], 1e3 , color='gray', alpha=0.3)
               
    return figaxes
    
# signal-to-noise ratio
def plot_cumsnr_dSigma(bins_list, cumsnr_list, ax, ls='-', marker='o', show_legend=True):
    ax.set_xscale('log')
    ax.set_xlabel(r'$R_{\rm min}~[h^{-1}{\rm Mpc}]$')
    ax.set_ylabel(r'cumulative $S/N(R_{\rm min}<R)$')
    ax.plot(bins_list[0], cumsnr_list[0], marker=marker, color='k', label='total', ls=ls)
    for i in range(len(bins_list)-1):
        ax.plot(bins_list[i+1], cumsnr_list[i+1], 
                marker=marker, color='C%d'%i, ls=ls,
                label=['LOWZ', 'CMASS1', 'CMASS2'][i])
    ax.legend(frameon=False) if show_legend else None
    ax.invert_xaxis()
    xlim = ax.get_xlim()
    dlnx = np.log(xlim[0]/xlim[1])
    dlnx = max([dlnx, np.log(10.0)])
    ax.set_xlim(xlim[0]*np.exp(dlnx*0.1), xlim[1]*np.exp(-dlnx*0.1))
    #ax.set_xlim(110, 9)
    return ax
    
def plot_cumsnr_wp(bins_list, cumsnr_list, ax, ls='-', marker='o', show_legend=True):
    ax.set_xscale('log')
    ax.set_xlabel(r'$R_{\rm min}~[h^{-1}{\rm Mpc}]$')
    ax.set_ylabel(r'cumulative $S/N(R_{\rm min}<R)$')
    ax.plot(bins_list[0], cumsnr_list[0], marker=marker, color='k', label='total', ls=ls)
    for i in range(len(bins_list)-1):
        ax.plot(bins_list[i+1], cumsnr_list[i+1], 
                marker=marker, color='C%d'%i, ls=ls,
                label=['LOWZ', 'CMASS1', 'CMASS2'][i])
    ax.legend(frameon=False) if show_legend else None
    ax.invert_xaxis()
    xlim = ax.get_xlim()
    dlnx = np.log(xlim[0]/xlim[1])
    dlnx = max([dlnx, np.log(10.0)])
    ax.set_xlim(xlim[0]*np.exp(dlnx*0.1), xlim[1]*np.exp(-dlnx*0.1))
    #ax.set_xlim(140, 5)
    return ax

def plot_cumsnr_xipm(bins_list, cumsnr_list, ax, binranges=None, ls='-', marker='o', show_legend=True):
    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta_{\rm min}~[{\rm arcmin}]$')
    ax.set_ylabel(r'cumulative $S/N(\theta_{\rm min}<\theta)$')
    
    ax.plot(bins_list[0], cumsnr_list[0], marker=marker, color='k', label='total', ls=ls)
    for i in range(len(bins_list)-1):
        ax.plot(bins_list[i+1], cumsnr_list[i+1], 
                marker=marker, color='C%d'%i, ls=ls,
                label=[r'$\xi_+$', r'$\xi_-$'][i])
    ax.legend(frameon=False) if show_legend else None
    ax.invert_xaxis()
    xlim = ax.get_xlim()
    dlnx = np.log(xlim[0]/xlim[1])
    dlnx = max([dlnx, np.log(10.0)])
    ax.set_xlim(xlim[0]*np.exp(dlnx*0.1), xlim[1]*np.exp(-dlnx*0.1))
    return ax

def plot_covariance(cov,vmin=None,vmax=None):
    from matplotlib.colors import SymLogNorm
    import seaborn as sns
    fig=plt.figure(figsize=(12,10))
    norm=SymLogNorm(linthresh=1e-13,vmin=vmin,vmax=vmax)
    ax=sns.heatmap(cov,cmap="RdBu",norm=norm, square=True,)
    ax.invert_yaxis()
    return fig

# chi2 distribution
def plot_chi2_dist(ddof, ax, c='k',ls='--'):
    from scipy.stats import chi2 as chi2dist
    x = np.linspace(0.0,ddof*4.0, 1000)
    y = chi2dist.pdf(x, ddof)
    ax.plot(x,y, c=c, ls=ls)
    
# 
def _get_levels(P, alphas):
    p = np.reshape(P, -1)
    p_sorted = np.sort(p)[::-1]
    cumsum = np.cumsum(p_sorted)
    cumsum/= cumsum[-1]
    
    levels = np.empty(len(alphas))
    for i, a in enumerate(alphas):
        levels[i] = p_sorted[cumsum > a].max()
        
    return np.sort(levels)

def get_paler_colors(color, n_levels, pale_factor=None):
    # convert a color into an array of colors for used in contours
    color = matplotlib.colors.colorConverter.to_rgb(color)
    pale_factor = pale_factor or 0.6
    cols = [color]
    for _ in range(1, n_levels):
        cols = [[c * (1 - pale_factor) + pale_factor for c in cols[0]]] + cols
    return cols
    
def plot_2dcontour(x,y,P,ax,alphas=[0.68, 0.95], color='C0', label=None, alpha=0.8, fill=True, lw=0.5):
    if not 0 in alphas:
        alphas.append(0)
    levels = _get_levels(P, alphas)
    pale_colors = get_paler_colors(color, levels.size-1)
    
    #for color, level in zip(pale_colors, levels):
    if fill:
        cs = ax.contourf(x,y,P,levels=levels,colors=pale_colors, alpha=alpha)
    else:
        cs = ax.contour(x,y,P,levels=levels,colors=pale_colors, alpha=alpha, linewidths=lw)
    proxy = plt.Rectangle((0, 0), 1, 1, fc=color, label=label, alpha=alpha)
    ax.patches += [proxy]
    return ax

def plot_2dgaussian(xc,yc,cov,ax,alphas=[0.68, 0.95], color='C0',nsigma=3, label=None,alpha=0.8):
    dx = cov[0,0]**0.5
    dy = cov[1,1]**0.5
    x = np.linspace(xc-dx*nsigma,xc+dx*nsigma,128)
    y = np.linspace(yc-dy*nsigma,yc+dy*nsigma,128)
    X,Y = np.meshgrid(x,y)
    icov = np.linalg.inv(cov)
    chi2 = (X-xc)**2*icov[0,0]+2*(X-xc)*(Y-yc)*icov[0,1]+(Y-yc)**2*icov[1,1]
    P = np.exp(-0.5*chi2)
    ax= plot_2dcontour(x,y,P,ax,alphas,color,label,alpha)
    return ax

def _errbar_plot(MCSamples, pnames, axes, y, fmt, color, alphas, label=None, capsize=1):
    from . import chainutils
    names = chainutils.get_names_from_MCSamples(MCSamples)
    # get for all param
    modes = chainutils.get_mode(MCSamples)
    hdis  = chainutils.get_HDI(MCSamples, alphas=alphas)
    # cast to dict
    hdis_dict  = dict(zip(names, hdis))
    modes_dict = dict(zip(names, modes))
    
    parler_colors = get_paler_colors(color, len(alphas))
    for ax, pname in zip(axes, pnames):
        if (pname not in modes_dict) or (pname not in hdis_dict):
            continue
        # plot mode and hdis
        mode = modes_dict[pname]
        hdis = hdis_dict[pname]
        for pcolor, hdi in zip(parler_colors, hdis[::-1]):
            lower , upper = hdi
            x = mode
            xerr = np.array([[ x-lower, upper-x ]]).T
            ax.errorbar(x, y, xerr=xerr, fmt=fmt, color=pcolor, capsize=capsize)

def errbar_plot(MCSamples_list, pnames, alphas=[0.68, 0.95], color='C0', fmt='o', capsize=0, padding_y=0.0, padding_x=0.2, offset_y=0.0, figsize=None, fig=None, label_fontsize=18, xlabel_fontsize=23, last_ax_ratio=2.5, partition=None, partition_color='gray', partition_ls=':', partition_lw=1, overplot=False, MCSamples_labels=None, fill=False, fill_idx=0, fill_color='C0', fill_alpha=0.2, markers=None, highlight_id=-1, highlight_color='red', highlight_alpha=0.1):
    if fig is None:
        if figsize is None:
            figsize = (5*len(pnames), len(MCSamples_list)/2.0)
        width_ratios = [1 for pname in pnames]
        width_ratios[-1] = last_ax_ratio
        fig, axes = plt.subplots(1, len(pnames), gridspec_kw={'width_ratios':width_ratios}, figsize=figsize)
    else:
        axes = fig.get_axes()
    alphas = np.sort(alphas)

    from . import chainutils
    # xlabel: param label
    plabels_dict = dict(zip(chainutils.get_names_from_MCSamples(MCSamples_list[0]), MCSamples_list[0].getLatex()[0]))
    for i, (ax, pname) in enumerate(zip(axes, pnames)):
        ax.set_ylim((0.0, 1.0))
        if i == len(pnames)-1:
            ax.set_xlabel(plabels_dict[pname], position=(1.0/2.0/last_ax_ratio, 0.0), fontsize=xlabel_fontsize)
        else:
            ax.set_xlabel(plabels_dict[pname], fontsize=xlabel_fontsize)

    # plot err bar
    N = len(MCSamples_list)
    for i, samples in enumerate(MCSamples_list):
        y = 1.0 - (0.5 + i) * 1.0/N + padding_y*1.0/N + offset_y*1.0/N
        _errbar_plot(samples, pnames, axes, y, fmt, color, alphas, samples.getLabel(), capsize)

    if overplot:
        # If this is overploting to the previous plot, then we will skip the process below.
        return fig

    # xlim
    for ax in axes:
        xlim = ax.get_xlim()
        dx = abs(np.diff(xlim))
        ax.set_xlim(xlim[0]-padding_x*dx, xlim[1]+padding_x*dx)
    
    # 200% expand xlim toward upper side
    ax = axes[-1]
    xticks = ax.get_xticks()
    xticks = xticks[::round(len(xticks)/3)] # reduce # of ticks vals
    ax.set_xticks(xticks)
    xlim = ax.get_xlim()
    dx = xlim[1]-xlim[0]
    ax.set_xlim((xlim[0], xlim[1]+dx*(last_ax_ratio-1)))

    # MCSample label
    N = len(MCSamples_list)
    for i, samples in enumerate(MCSamples_list):
        y = 1.0 - (0.5 + i) * 1.0/N + padding_y*1.0/N
        if MCSamples_labels is None:
            label = samples.getLabel()
        else:
            label = MCSamples_labels[i]
        #ax.text(1/last_ax_ratio, y, label.replace("_","\_"), ha='left', va='center', transform=ax.transAxes, fontsize=label_fontsize)    
        ax.text(1/last_ax_ratio, y, label, ha='left', va='center', transform=ax.transAxes, fontsize=label_fontsize)    

    # remoev y ticks
    for ax in axes:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='y', which='both', length=0)

    # partition
    if partition is not None:
        assert sum(partition) == len(MCSamples_list), 'sum of partition must be equal to the length of MCSamples_list'
        y = 1.0
        for p in partition:
            y -= p/N
            for ax in axes:
                ax.axhline(y, color=partition_color, linestyle=partition_ls, linewidth=partition_lw)
                
    # fill between errorbar of specified MCSamples for comparison
    axes = fig.get_axes()
    for i, pname in enumerate(pnames):
        lim1, lim2 = chainutils.get_HDI(MCSamples_list[fill_idx], dtype=dict, alphas=[0.68])[pname][0]
        ax = axes[i]
        ax.axvspan(lim1, lim2, color=fill_color, alpha=fill_alpha)
        
    if markers is not None:
        for ax, pname in zip(axes, pnames):
            ax.axvline(markers.get(pname, np.nan), color='k', ls='-.', lw=0.5)
            
    # highlight
    if highlight_id != -1:
        if isinstance(highlight_id, int):
            ids = [highlight_id]
        elif isinstance(highlight_id, list):
            ids = highlight_id
        for ax in axes:
            for i in ids:
                y = 1.0 - (0.5 + i) * 1.0/N + padding_y*1.0/N + offset_y*1.0/N
                ymin, ymax = y-0.5/N, y+0.5/N
                ax.axhspan(ymin, ymax, color=highlight_color, alpha=highlight_alpha)
        
    return fig


def multiple_errbar_plot(MCSamples_list_list, pnames, 
                         alphas=[0.68], colors=None, 
                         legend_labels=None,
                         fmt='o', capsize=0, 
                         padding_y=0.0, padding_x=0.2, 
                         figsize=None, 
                         label_fontsize=18, 
                         xlabel_fontsize=23, 
                         last_ax_ratio=2.5, 
                         partition=None, 
                         partition_color='gray', 
                         partition_ls=':', 
                         partition_lw=1, 
                         overplot=False, 
                         MCSamples_labels=None, 
                         fill=False, 
                         fill_idx=0, 
                         fill_color='C0', 
                         fill_alpha=0.2):
    
    if not isinstance(MCSamples_list_list[0], list):
        MCSamples_list_list = [MCSamples_list_list]
        
    if colors is None:
        colors = ['C%d'%i for i, MCSamples_list in enumerate(MCSamples_list_list)]
    
    fig, overplot = None, False
    n = len(MCSamples_list_list)
    for i, MCSamples_list in enumerate(MCSamples_list_list):
        color = colors[i]
        if n!=0:
            offset_y = 0.15*(1 - 2*i/(n-1))
        else:
            offset_y = 0
        fig = errbar_plot(MCSamples_list, pnames, fmt=fmt, alphas=alphas, partition=partition, color=color, offset_y=offset_y, fig=fig, overplot=overplot, capsize=capsize, last_ax_ratio=last_ax_ratio)
        overplot = True
        
    N = len(pnames)
    if legend_labels is not None:
        ax = fig.get_axes()[0]
        for color, label in zip(colors, legend_labels):
            ax.errorbar(10, 0, 1, fmt=fmt, color=color, capsize=capsize, label=label)
        x = (N + last_ax_ratio-1)/2.0
        ax.legend(bbox_to_anchor=(x, 1), loc='lower center', ncol=n)
        
    return fig