import numpy as np
import time
import os
from scipy.special import erfinv, erf

def gen_normal(sigma, size, random=None):
    if random is None:
        random = np.random
    u = random.uniform(0, 1, size)
    x = (2*u-1) * erf(sigma*2**-0.5)
    x = erfinv(x)*2**0.5
    return x

def gen_multivariate_normal(mean, cov, sigma, size, random=None):
    d, r = np.linalg.eig(cov) # c = r * d * r.T
    ndim = d.size
    u = gen_normal(sigma, size=size*ndim).reshape(size, ndim)
    p = np.dot(r, (d**0.5*u).T).T
    p+= mean
    return p

class nested_minimizer:
    def __init__(self, root, N, M, nsigma=2, tol=1e-2, maxitern=10000, seed=0, verbose=True, resume=False):
        self.root = root
        self.random = np.random
        self.seed = seed
        self.random.seed(seed)
        self.N = N
        self.M = M
        self.nsigma= nsigma
        self.tol = tol
        self.maxitern = maxitern
        self.verbose = verbose
        self.progress        = []
        self.progress_wmean  = []
        self.progress_std    = []
        self.progress_mean   = []
        self.progress_bf     = []
        if resume:
            self._set_live_from_previous_run()
            self._set_progress_from_previous_run()
        
    def set_lnlike_func(self, mn_lnlike, ndim, nparams):
        self.mn_lnlike = mn_lnlike
        self.ndim = ndim
        self.nparams = nparams
        
    def set_live_from_external(self, init_chain, init_lnlike):
        self.init_samples_live, self.init_lnlike_live = self._extract_live(init_chain, init_lnlike, self.N)
        
    def _set_live_from_previous_run(self):
        fname = self.root+'samples_live.dat'
        if os.path.exists(fname):
            self.init_samples_live = np.loadtxt(fname)
        else:
            print(fname, 'is not found. Init with external chain.')
        fname = self.root+'lnlike_live.dat'
        if os.path.exists(fname):
            self.init_lnlike_live  = np.loadtxt(fname)
        else:
            print(fname, 'is not found. Init with external chain.')
        
    def _set_progress_from_previous_run(self):
        fname = self.root+'progress.dat'
        if os.path.exists(fname):
            self.progress = np.loadtxt(fname)
            self.progress_wmean  = list(np.loadtxt(fname.replace('.dat', '_wmean.dat')))
            self.progress_std    = list(np.loadtxt(fname.replace('.dat', '_std.dat')))
            self.progress_mean   = list(np.loadtxt(fname.replace('.dat', '_mean.dat')))
            self.progress_bf     = list(np.loadtxt(fname.replace('.dat', '_bf.dat')))
        else:
            print(fname, 'is not found. Init with external chain.')
            
    def set_MHDIerr_full(self, MHDIerr_full):
        """Setting the marginalhighest density interval err size for full parameter set (sampling and derived parameters)"""
        self.MHDIerr_full = MHDIerr_full
        
    def get_MHDIerr_full(self):
        return self.MHDIerr_full.copy()
        
    def _extract_live(self, chain, lnlike, N):
        idx = np.argsort(lnlike)[-N:]
        return chain[idx, :].copy(), lnlike[idx].copy()
    
    def _estimate_mean_cov(self, samples_live, lnlike_live):
        mean = np.average(samples_live, axis=0)
        cov  = np.cov(samples_live.T)
        return mean, cov

    def _generate_candidates_for_sampling_param(self, mean, cov, M):
        mean_sampling = mean[:self.ndim]
        cov_sampling  = cov[:self.ndim, :self.ndim]
        samples_cand = gen_multivariate_normal(mean_sampling, cov_sampling, self.nsigma, size=M, random=self.random)
        return samples_cand
    
    def get_std_s_max(self, cov, MHDIerr):
        return np.max(np.diag(cov[:self.ndim, :self.ndim])**0.5/MHDIerr[:self.ndim])
    
    def _get_like_weighted_mean(self, samples_live, lnlike_live):
        w = np.exp(lnlike_live-lnlike_live.max())
        mean = np.average(samples_live, axis=0, weights=w)
        return mean
        
    def get_bestfit_from_live(self, samples_live, lnlike_live):
        bf_idx = np.argmax(lnlike_live)
        bestfit = samples_live[bf_idx,:]
        return bestfit
        
    def run(self):
        t0 = time.time()
        samples_live, lnlike_live = self.init_samples_live, self.init_lnlike_live
        mean, cov = self._estimate_mean_cov(samples_live, lnlike_live)
        bf = self.get_bestfit_from_live(samples_live, lnlike_live)
        wmean = self._get_like_weighted_mean(samples_live, lnlike_live)
        MHDIerr = self.get_MHDIerr_full()
        std_s_max = self.get_std_s_max(cov, MHDIerr)
        
        itern = len(self.progress)
        while std_s_max>self.tol and (itern < self.maxitern):
            self.progress_mean.append(mean)
            self.progress_bf.append(bf)
            self.progress_wmean.append(wmean)
            self.progress_std.append(np.diag(cov)**0.5)
            samples_cand_sampling = self._generate_candidates_for_sampling_param(wmean, cov, self.M)
            
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            
            # Compute lnlike with parallelized threads
            samples_cand = []
            lnlike_cand  = []
            for sample in samples_cand_sampling[rank::size]:
                cube = np.hstack([sample, np.zeros(self.nparams-self.ndim)])
                lnlike = self.mn_lnlike(cube, self.ndim, self.nparams)
                samples_cand.append(cube)
                lnlike_cand.append(lnlike)
                
            # Gathering objects
            samples_cand = comm.gather(samples_cand, root=0)
            lnlike_cand  = comm.gather(lnlike_cand , root=0)

            # Reshape
            if rank == 0:
                samples_cand = np.vstack(samples_cand)
                lnlike_cand  = np.hstack(lnlike_cand)
            else:
                samples_cand = np.empty((self.M, self.nparams))
                lnlike_cand  = np.empty(self.M)

            # Broadcasting objects
            samples_cand = comm.bcast(samples_cand, root=0)
            lnlike_cand  = comm.bcast(lnlike_cand , root=0)

            # Stack live and candidate samples
            samp = np.vstack([samples_live, samples_cand])
            lnl  = np.hstack([lnlike_live, lnlike_cand])
            
            # Select live samples
            samples_live, lnlike_live = self._extract_live(samp, lnl, self.N)
            del samp, lnl
            
            # Update mean and cov with new live samples
            mean, cov = self._estimate_mean_cov(samples_live, lnlike_live)
            bf = self.get_bestfit_from_live(samples_live, lnlike_live)
            wmean = self._get_like_weighted_mean(samples_live, lnlike_live)
            std_s_max = self.get_std_s_max(cov, MHDIerr)

            # Update the ratio of std (searching area width) and sigma (marginal MHDI size)
            self.progress = np.append(self.progress, std_s_max)
            
            if self.verbose:
                print('-------------------------------')
                print('>>> nested_minimizer is now....')
                print('>>> itern          = %d'%itern  )
                print('>>> max(std/sigma) = %.4f'%std_s_max)
                print('-------------------------------')

            # Save current status
            eltime = time.time()-t0
            header = 'Running. Optimized by random_lnlike_minimizer. time=%f sec, N=%d, M=%d, nsigma=%f, tol=%f, maxitern=%d, seed=%d'%(eltime, self.N, self.M, self.nsigma, self.tol, self.maxitern, self.seed)
            self.dump(samples_live, lnlike_live, self.progress, self.progress_mean, self.progress_std, self.progress_bf, self.progress_wmean, header)

            itern += 1

        # Save final status
        eltime = time.time()-t0
        header = 'Done. Optimized by random_lnlike_minimizer. time=%f sec, N=%d, M=%d, nsigma=%f, tol=%f, maxitern=%d, seed=%d'%(eltime, self.N, self.M, self.nsigma, self.tol, self.maxitern, self.seed)
        self.dump(samples_live, lnlike_live, self.progress, self.progress_mean, self.progress_std, self.progress_bf, self.progress_wmean, self.progress_nsigma, header)
        
    def dump(self, samples_live, lnlike_live, progress, progress_mean, progress_std, progress_bf, progress_wmean, header):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        if rank == 0:
            root = self.root
            bf = self.get_bestfit_from_live(samples_live, lnlike_live)
            np.savetxt(root+'.dat', bf, header=header)
            np.savetxt(root+'progress.dat', progress)
            np.savetxt(root+'progress_wmean.dat', progress_wmean)
            np.savetxt(root+'progress_std.dat', progress_std)
            np.savetxt(root+'samples_live.dat', samples_live)
            np.savetxt(root+'lnlike_live.dat', lnlike_live)
            np.savetxt(root+'progress_bf.dat', progress_bf)
            np.savetxt(root+'progress_mean.dat', progress_mean)

class plotter_nested_minimizer:
    def __init__(self, root):
        self.root = root
        self.load_output()

    def load_output(self):
        # output
        self.progress      = np.loadtxt(self.root+'progress.dat')
        self.wmean         = np.loadtxt(self.root+'progress_wmean.dat')
        self.std           = np.loadtxt(self.root+'progress_std.dat')
        self.bf            = np.loadtxt(self.root+'progress_bf.dat')
        # setup
        with open(self.root+'.dat', 'r') as f:
            line = f.readline()
            line = '.'.join(line.split('.')[2:])
            line.replace('sec', '')
            sep  = line.split(',')
            del line
        self.tol      = float(sep[4].split('=')[1])
        self.maxitern = float(sep[5].split('=')[1])
        self.nsigma   = float(sep[3].split('=')[1])

    def plot_tol_progress(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.set_yscale('log')
        ax.plot(self.progress, label='progress')
        ax.axhline(self.tol, color='k', ls='--', label='tol')
        ax.legend()
        ax.set_xlabel('step i')
        ax.set_ylabel('max( std/MHDIerr )')
        plt.show()

    def plot_param_progress(self, nmax, nrow=1, titles=None, figsize=2):
        import matplotlib.pyplot as plt
        nx, ny = int(nmax/nrow+nmax%nrow), int(nrow)
        fig, axes = plt.subplots(ny, nx, figsize=(figsize*nx,figsize*ny), sharex=True)
        axes = fig.get_axes()
        for i in range(nmax):
            ax = axes[i]
            if i//nx == ny-1:
                ax.set_xlabel('step i')
            x = np.arange(self.wmean.shape[0])
            ax.plot(x, self.wmean[:,i], color='C0')
            ax.fill_between(x, self.wmean[:,i]+self.std[:,i]*self.nsigma, self.wmean[:,i]-self.std[:,i]*self.nsigma, label='searching region (%d-sigma)'%(self.nsigma), alpha=0.3)
            ax.plot(x, self.bf[:,i], color='C1', label='bf')
            if titles is not None:
                ax.text(0.95, 0.95, titles[i], transform=ax.transAxes, va='top', ha='right', fontsize=13)
        axes[0].legend(loc='lower center', bbox_to_anchor=(nx/2+0.5, 1.0), ncol=2)
        for j in range(i+1, int(nx*ny)):
            ax = axes[j]
            ax.axis('off')
        return fig

