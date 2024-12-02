import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.optimize import fsolve

# Original paper : Takahashi et al. (2012)
# https://arxiv.org/abs/1208.2701
# DOI : 10.1088/0004-637X/761/2/152

def pklintable2pkhalotable(k_in, z_in, pklintable_in, Omm0, Omde0, w0, wa, h, mnu, f1h=1, dz=0.05):
    """
    Compute halofit power spectrum from linear power spectrum.
    This function works on the array of redshifts.

    Args:
        k_in (array): k [h/Mpc]
        z_in (array): redshift
        pklintable_in (array): linear power spectrum [(Mpc/h)^3]
        Omm0 (float): Omega_m0
        Omde0 (float): Omega_de0
        w0 (float): w0
        wa (float): wa
        h (float): h
        mnu (float): sum of neutrino mass [eV]
        f1h (float): f1h, relative amplitude of the halo term
        dz (float): redshift bin width

    Returns:
        pkhalo (array): halofit power spectrum [(Mpc/h)^3]
    """
    # Change unit
    k            = k_in.copy()*h # [1/Mpc]
    pklintable   = pklintable_in.copy()/h**3 # [Mpc^3]
    Delta_L      = (pklintable.T*k**3/(2.*np.pi**2)).T
    z            = z_in*1
    
    # cosmological paramters
    Omdez = _get_Omdez(z, Omm0, Omde0, w0, wa)
    Ommz  = _get_Ommz(z, Omm0, Omde0, w0, wa)
    fnu0  = _get_fnu0(mnu, Omm0, h)

    # calculate coeffs on sparse redshift bins
    i_sub = [0]
    for i in range(1, z.size):
        if z[i_sub[-1]]+dz < z[i]:
            i_sub.append(i)
    if i_sub[-1] != z.size-1:
        i_sub.append(z.size-1)
    i_sub = np.array(i_sub)
    
    R_sigma_sub = np.empty(i_sub.size)
    coeffs_sub  = np.empty((i_sub.size, 8))
    for j,i in enumerate(i_sub):
        # compute nonlinear scale
        R_sigma = _get_R_sigma(k, z[i], Delta_L[:,i])
        neff, C = _get_neff_C(k, Delta_L[:,i], R_sigma)
        coeffs  = _get_coeffs(neff, C, Omdez[i], w0, fnu0)
        # append
        R_sigma_sub[j] = R_sigma
        coeffs_sub[j]  = coeffs
    R_sigma_interp = interp1d(z[i_sub], R_sigma_sub, kind='linear', bounds_error=False, fill_value=(R_sigma_sub[0], R_sigma_sub[-1]))
    coeffs_interp  = [ius(z[i_sub], coeff_sub) for coeff_sub in coeffs_sub.T]
    
    # interpolate
    R_sigma = R_sigma_interp(z)
    coeffs  = np.array([ [coeffs_interp[j](zi) for j in range(8)] for zi in z])
    
    # get halofit prediction
    pkhalo = np.empty(pklintable.shape)
    for i in range(z.size):
        pkhalo[:,i] = _get_pkhalo(k, Delta_L[:,i], R_sigma[i], Ommz[i], Omdez[i], Omm0, fnu0, coeffs[i], f1h=f1h)
    
    # get back unit
    pkhalo = pkhalo*h**3
    
    return pkhalo

def pklin2pkhalofit(k_in, z_in, pklin_in, Omm0, Omde0, w0, wa, h, mnu, f1h=1):
    """
    Compute halofit power spectrum from linear power spectrum.
    This function works on a single redshift (float, int).

    Args:
        k_in (array): k [h/Mpc]
        z_in (float): redshift
        pklin_in (array): linear power spectrum [(Mpc/h)^3]
        Omm0 (float): Omega_m0
        Omde0 (float): Omega_de0
        w0 (float): w0
        wa (float): wa
        h (float): h
        mnu (float): sum of neutrino mass [eV]
        f1h (float): f1h, relative amplitude of the halo term

    Returns:
        pkhalo (array): halofit power spectrum [(Mpc/h)^3]
    """
    # Change unit
    k       = k_in*h # [1/Mpc]
    pklin   = pklin_in/h**3 # [Mpc^3]
    Delta_L = pklin*k**3/(2.*np.pi**2)
    z       = z_in*1
    
    # cosmological paramters
    Omdez = _get_Omdez(z, Omm0, Omde0, w0, wa)
    Ommz  = _get_Ommz(z, Omm0, Omde0, w0, wa)
    fnu0  = _get_fnu0(mnu, Omm0, h)

    # compute nonlinear scale
    R_sigma = _get_R_sigma(k, z, Delta_L)
    neff, C = _get_neff_C(k, Delta_L, R_sigma)
    coeffs  = _get_coeffs(neff, C, Omdez, w0, fnu0)

    # get halofit prediction
    pkhalo = _get_pkhalo(k, Delta_L, R_sigma, Ommz, Omdez, Omm0, fnu0, coeffs, f1h=f1h)
    
    # get back unit
    pkhalo = pkhalo*h**3
    
    return pkhalo

def _sigma(k, Delta_L, R):
    """Computing the variance of linear matter power spectrum.
    .. math:
        \\sigma(R) \\equiv \\int_0^\\infty {\\rm d}\\ln k \\Delta_{\\rm L}^2(k)e^{-k^2R^2}

    Args:
        R (float): Smoothing scale to compute the variance of linear power spectrum.
    Returns:
        sigma (float): variance of linear power smoothed at scale :math:`R`.
    """
    return integrate.simps(Delta_L*np.exp(-(k*R)**2), np.log(k))

def _get_R_sigma_max(k):
    return 0.01/k.max()

def _get_R_sigma(k, z, Delta_L):
    """Computing R satisfying 
    .. math:
        \\sigma(R) \\equiv \\int_0^\\infty {\\rm d}\\ln k \\Delta_{\\rm L}^2(k)e^{-k^2R^2}=1.
    Args:
        k (array): k [1/Mpc]
        z (float): cosmological redshift.
    Returns:
        R (float): :math:`R` satisfying :math:`\\sigma(R)=1`.
    """
    if _sigma(k, Delta_L, 0.0) < 1:
        print('Warning pyhalofit: sigma(0)<0 for z=%f. Set R_sigma=%f.'%(z, _get_R_sigma_max(k)))
        R_sigma = 0.01/k.max()
    else:
        # init guess : [1/Mpc]   Takada & Jain (2004) fiducial model
        k_sigma = 10.**(-1.065+4.332e-1*(1.+z)-2.516e-2*pow(1.+z,2)+9.069e-4*pow(1.+z,3))
        def eq(R):
            return _sigma(k, Delta_L, R) - 1.0
        R_sigma = abs(fsolve(eq, [1./k_sigma]))
    return R_sigma

def _get_neff_C(k, Delta_L, R_sigma, ep=1e-2):
    """
    Compute effective spectral index and curvature of the power spectrum.

    Args:
        k (array): k [1/Mpc]
        Delta_L (array): linear power spectrum [(Mpc/h)^3]
        R_sigma (float): smoothing scale to compute the variance of linear power spectrum.
        ep (float): error of R_sigma.

    Returns:
        neff (float): effective spectral index of the power spectrum.
        C (float): curvature of the power spectrum.
    """
    R = R_sigma*(1+np.linspace(-ep, ep, 3))
    sigma = np.array([_sigma(k, Delta_L, _R) for _R in R])

    dlnsigma_dlnR = np.diff(np.log(sigma))/np.diff(np.log(R))
    R2 = 0.5*(R[1:]+R[:-1])

    d2lnsigma_d2lnR = np.diff(dlnsigma_dlnR)/np.diff(np.log(R2))
    R3 = 0.5*(R2[1:]+R2[:-1])

    neff = -3 - np.mean(dlnsigma_dlnR)
    C    = - d2lnsigma_d2lnR[0]

    return neff, C

def _get_coeffs(neff, C, Omdez, w0, fnu0):
    """
    Compute coefficients of the halofit power spectrum.

    Args:
        neff (float): effective spectral index of the power spectrum.
        C (float): curvature of the power spectrum.
        Omdez (float): Omega_de(z)
        w0 (float): w0
        fnu0 (float): sum of neutrino mass fraction at z=0

    Returns:
        an (float): an
        bn (float): bn
        cn (float): cn
        gamman (float): gamman
        alphan (float): alphan
        betan (float): betan
        mun (float): mun
        nun (float
    """
    an = 10.**( 1.5222 + 2.8553*neff + 2.3706*neff**2 + 0.9903*neff**3 + 0.2250*neff**4 - 0.6038*C + 0.1749*Omdez*(1.+w0) )
    bn = 10.**(-0.5642 + 0.5864*neff + 0.5716*neff**2 - 1.5474*C + 0.2279*Omdez*(1.+w0))
    cn = 10.**( 0.3698 + 2.0404*neff + 0.8161*neff**2 + 0.5869*C)
    gamman = 0.1971 - 0.0843*neff + 0.8460*C
    alphan = abs( 6.0835 + 1.3373*neff - 0.1959*neff**2 - 5.5274*C)
    betan  = 2.0379 - 0.7354*neff + 0.3157*neff**2 + 1.2490*neff**3 + 0.3980*neff**4 - 0.1682*C + fnu0*(1.081 + 0.395*neff**2)
    mun    = 0.
    nun    = 10.**(5.2105 + 3.6902*neff)
    return an, bn, cn, gamman, alphan, betan, mun, nun

def _get_pkhalo(k, Delta_L, R_sigma, Ommz, Omdez, Omm0, fnu0, coeffs, f1h=1):
    """
    Compute halofit power spectrum based on the computed coefficients.

    Args:
        k (array): k [1/Mpc]
        Delta_L (array): linear power spectrum [(Mpc/h)^3]
        R_sigma (float): smoothing scale to compute the variance of linear power spectrum.
        Ommz (float): Omega_m(z)
        Omdez (float): Omega_de(z)
        Omm0 (float): Omega_m0
        fnu0 (float): sum of neutrino mass fraction at z=0
        coeffs (array): coefficients of the halofit power spectrum.
        f1h (float): f1h, relative amplitude of the halo term
    
    Returns:
        pkhalo (array): halofit power spectrum [(Mpc/h)^3]
    """
    an, bn, cn, gamman, alphan, betan, mun, nun = coeffs
    y = k * R_sigma
    f = y/4. + y**2/8.
    f1b, f2b, f3b = Ommz**-0.0307, Ommz**-0.0585, Ommz**0.0743
    f1a, f2a, f3a = Ommz**-0.0732, Ommz**-0.1423, Ommz**0.0725
    frac = Omdez/(1-Ommz)
    f1, f2, f3 = frac*f1b+(1-frac)*f1a, frac*f2b+(1-frac)*f2a, frac*f3b+(1-frac)*f3a
    Delta_Laa = Delta_L*(1.+fnu0*47.48*k**2/(1.+1.5*k**2))
    Delta_Q = Delta_L * ((1.+Delta_Laa)**betan)/(1.+alphan*Delta_Laa) * np.exp(-f)
    Delta_H = an*y**(3.*f1) / (1.+bn*y**f2 + (cn*y*f3)**(3.-gamman))
    Delta_H = Delta_H / (1. + mun/y + nun/y**2) * (1+fnu0*(0.977-18.015*(Omm0-0.3)))
    pkhalo = (Delta_Q + f1h * Delta_H) * (2.*np.pi**2) / k**3 
    return pkhalo

def _get_Ommz(z, Omm0, Omde0, w0, wa):
    """returns Omega_m(z)"""
    a = 1.0/(1+z)
    Qa2 = a**(-1.0-3.0*(w0+wa))*np.exp(-3.0*(1-a)*wa)
    Omt =1.0+(Omm0+Omde0-1.0)/(1-Omm0-Omde0+Omde0*Qa2+Omm0/a)
    Ommz=Omt*Omm0/(Omm0+Omde0*a*Qa2)
    return Ommz

def _get_Omdez(z, Omm0, Omde0, w0, wa):
    """returns Omega_de(z)""" 
    a = 1.0/(1+z)
    Qa2 = a**(-1.0-3.0*(w0+wa))*np.exp(-3.0*(1-a)*wa)
    Omt =1.0+(Omm0+Omde0-1.0)/(1-Omm0-Omde0+Omde0*Qa2+Omm0/a)
    Omde=Omt*Omde0*Qa2/(Omde0*Qa2+Omm0/a)
    return Omde

def _get_fnu0(mnu, Omm0, h):
    """return Omega_nu(z=0)"""
    Omnuh2   = 0.00064*(mnu/0.06)
    fnu = Omnuh2/h**2 / Omm0
    return fnu
