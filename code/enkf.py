import numpy as np


def construct_GC(cut, mlocs, ylocs):
    """
    Construct the Gaspari and Cohn localization matrix for a 1D field.

    Parameters:
        cut (float): Localization cutoff distance.
        mlocs (array): 1D coordinates of the model grids [x1, x2, ...].
        ylocs (array): 1D coordinates of the observations locations [y1, y2, ...].

    Returns:
        np.ndarray: Localization matrix of shape (len(ylocs), len(mlocs)).
    """
    L = len(mlocs)
    
    diff = np.abs(ylocs[:, None] - mlocs[None, :])
    dist = np.minimum.reduce([diff, np.abs(diff - L), np.abs(diff + L)])
    r = dist / (0.5 * cut)
    
    V = np.zeros_like(r)
    mask1 = (dist < 0.5 * cut)
    mask2 = (dist >= 0.5 * cut) & (dist < cut)

    r1 = r[mask1]
    V[mask1] = (
        -0.25 * r1**5
        + 0.5 * r1**4
        + (5.0 / 8.0) * r1**3
        - (5.0 / 3.0) * r1**2
        + 1.0
    )

    r2 = r[mask2]
    V[mask2] = (
        (r2**5 / 12.0)
        - 0.5 * r2**4
        + (5.0 / 8.0) * r2**3
        + (5.0 / 3.0) * r2**2
        - 5.0 * r2
        + 4.0
        - (2.0 / (3.0 * r2))
    )
    return V

def construct_GC_2d_general(cut, mlocs, ylocs, Nx=None):
    """
    Construct the Gaspari and Cohn localization matrix for a 2D field.

    Parameters:
        cut (float): Localization cutoff distance.
        mlocs (array of shape (nstates,2)): 2D coordinates of the model states [(x1, y1), (x2, y2), ...].
        ylocs (array of shape (nobs,2)): 2D coordinates of the observations [[x1, y1], [x2, y2], ...].
        Nx (int, optional): Number of grid points in each direction.

    Returns:
        np.ndarray: Localization matrix of shape (len(ylocs), len(mlocs)).
    """
    ylocs = ylocs[:, np.newaxis, :]  # Shape (nobs, 1, 2)
    mlocs = mlocs[np.newaxis, :, :]  # Shape (1, nstates, 2)

    # Compute distances
    dist = np.linalg.norm((mlocs - ylocs + Nx // 2) % Nx - Nx // 2, axis=2)

    # Normalize distances
    r = dist / (0.5 * cut)

    # Compute localization function
    V = np.zeros_like(dist)

    mask2 = (0.5 * cut <= dist) & (dist < cut)
    mask3 = (dist < 0.5 * cut)

    V[mask2] = (
        r[mask2]**5 / 12.0 - r[mask2]**4 / 2.0 + r[mask2]**3 * 5.0 / 8.0
        + r[mask2]**2 * 5.0 / 3.0 - 5.0 * r[mask2] + 4.0 - 2.0 / (3.0 * r[mask2])
    )
    
    V[mask3] = (
        -r[mask3]**5 * 0.25 + r[mask3]**4 / 2.0 + r[mask3]**3 * 5.0 / 8.0 
        - r[mask3]**2 * 5.0 / 3.0 + 1.0
    )

    return V

def stochastic_enkf(ensemble_size, nobsgrid, xens, Hk, obs_error_var, localize, CMat, zobs):
    rn = 1.0 / (ensemble_size - 1)

    for iobs in range(0, nobsgrid):
        xmean = np.mean(xens, axis=0)  # 1xn
        xprime = xens - xmean
        hxens = (Hk[iobs, :] * xens.T).T  # 40*1
        hxmean = np.mean(hxens, axis=0)
        hxprime = hxens - hxmean
        hpbht = (hxprime.T * hxprime * rn)[0, 0]
        pbht = (xprime.T * hxprime) * rn
    
        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = np.multiply(Cvect.T, (pbht / (hpbht + obs_error_var)))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        inc = (kfgain * (zobs[:,iobs] - hxens).T).T

        xens = xens + inc

    return xens


def eakf(ensemble_size, nobs, xens, Hk, obs_error_var, localize, CMat, obs):
    """
    Ensemble Adjustment Kalman Filter (EAKF)

    Parameters:
        ensemble_size (int): Number of ensemble members.
        nobs (int): Number of observations.
        xens (np.ndarray): Ensemble matrix of shape (ensemble_size, nmod).
        Hk (np.ndarray): Observation operator matrix of shape (nobs, nmod).
        obs_error_var (float): Observation error variance.
        localize (int): Flag for localization (1 for applying localization, 0 otherwise).
        CMat (np.ndarray): Localization matrix of shape (nobs, nmod).
        obs (np.ndarray): Observations of shape (nobs,).

    Returns:
        np.ndarray: Updated ensemble matrix.
    """
    rn = 1.0 / (ensemble_size - 1)
    for iobs in range(0, nobs):
        xmean = np.mean(xens, axis=0) 
        xprime = xens - xmean
        hxens = Hk[iobs, :] @ xens.T
        hxmean = np.mean(hxens)
        hxprime = hxens - hxmean
        hpbht = hxprime @ hxprime.T * rn
        gainfact = (hpbht + obs_error_var) / hpbht * (1.0 - np.sqrt(obs_error_var / (hpbht + obs_error_var)))
        pbht = (hxprime @ xprime) * rn

        if localize == 1:
            Cvect = CMat[iobs, :]
            kfgain = Cvect * (pbht / (hpbht + obs_error_var))
        else:
            kfgain = pbht / (hpbht + obs_error_var)

        obs_inc = obs[iobs] - hxmean
        mean_inc = kfgain * obs_inc
        prime_inc = - (gainfact * kfgain[:, None] @ hxprime[None, :]).T
        xens = xens + mean_inc + prime_inc
    return xens