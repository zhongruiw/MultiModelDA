import numpy as np
from enkf import eakf, construct_GC
from L63_noisy import L63RegimeModel
from cluster import FCMEntropy
import pickle
from time import time


def allocate_ensemble(ensemble_size, weights):
    """
    Allocate an integer number of ensemble members to each model, summing to ensemble_size,
    approximately proportional to weights (largest remainder method).
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    w = w / total
    raw = w * ensemble_size
    alloc = np.floor(raw).astype(int)
    remainder = ensemble_size - alloc.sum()
    frac = raw - np.floor(raw)
    order = np.argsort(-frac)  # descending
    alloc[order[:remainder]] += 1
    return alloc.tolist()

def reallocate_ens(initial_allocation, target_counts):
    '''reallocate ensemble members with minimal movements (pen for ensemble member, drawer for model)'''
    from collections import deque
    M = len(initial_allocation)
    initial_counts = [len(pens) for pens in initial_allocation]
    delta = [target_counts[i] - initial_counts[i] for i in range(M)]
    drawers = [list(pens) for pens in initial_allocation] # Create copy of allocation
    surplus = deque()
    deficit = deque()
    for i in range(M):
        if delta[i] < 0:
            surplus.append((i, -delta[i]))
        elif delta[i] > 0:
            deficit.append((i, delta[i]))
    while surplus and deficit:
        s_idx, s_amt = surplus[0]
        d_idx, d_amt = deficit[0]
        move_amt = min(s_amt, d_amt)
        for _ in range(move_amt):
            pen_id = drawers[s_idx].pop()
            drawers[d_idx].append(pen_id)
        # Update queue entries
        if s_amt > d_amt:
            surplus[0] = (s_idx, s_amt - move_amt)
            deficit.popleft()
        elif d_amt > s_amt:
            deficit[0] = (d_idx, d_amt - move_amt)
            surplus.popleft()
        else:
            surplus.popleft()
            deficit.popleft()

    return drawers

def markov_generator(mu0, muT, T):
    """
    Find a time-homogeneous generator matrix Q such that: muT ≈ mu0 @ expm(T * Q)
    Returns:
        Q_opt (np.ndarray): Optimal generator matrix (n x n)
        h (np.ndarray): Holding parameter (n,)
        R (np.ndarray): Routing matrix (n x n)
    """
    from scipy.optimize import minimize
    from scipy.linalg import expm

    n = len(mu0)
    def objective(Q_flat):
        Q = Q_flat.reshape((n, n))
        P = expm(T * Q)
        mu_pred = mu0 @ P
        return np.sum((mu_pred - muT) ** 2)
    # Off-diagonal entries >= 0
    ineq_constr = {
        'type': 'ineq',
        'fun': lambda Q_flat: Q_flat.reshape((n, n))[~np.eye(n, dtype=bool)]
    }
    # Row sums == 0
    eq_constr = {
        'type': 'eq',
        'fun': lambda Q_flat: Q_flat.reshape((n, n)).sum(axis=1)
    }
    Q0 = np.random.rand(n, n)
    np.fill_diagonal(Q0, 0)
    Q0 = Q0 / Q0.sum(axis=1, keepdims=True)  # row-stochastic off-diagonal
    np.fill_diagonal(Q0, -Q0.sum(axis=1))    # make row sums 0
    Q0_flat = Q0.flatten()
    result = minimize(objective, Q0_flat, constraints=[ineq_constr, eq_constr], method='SLSQP')
    success = result.success
    if success:
        Q_opt = result.x.reshape((n, n))
        h = -np.diag(Q_opt)
        h = np.clip(h, 0.0, None)   # clip tiny negatives from numerics
        R = np.zeros_like(Q_opt)
        for i in range(len(Q_opt)):
            if h[i] > 0:
                row = Q_opt[i, :] / h[i]
                row[i] = 0.0  # no self-transition
                row = np.clip(row, 0.0, None) # clip tiny negatives from numerics
                s = row.sum()
                if s > 0:
                    row /= s  # renormalize to sum to 1
                else:
                    row = np.zeros(n)
                    row[i] = 1.0   # degenerate case: if all zeros after clipping, stay in the same model
            else:
                row = np.zeros(n)  # stay in the same model
                row[i] = 1.0
            R[i, :] = row
        return Q_opt, h, R
    else:
        raise RuntimeError("Optimization failed: could not find a valid generator matrix.")
        
def bootstrap_spawn(ens_m, n_new, jitter=1e-3):
    idx = np.random.randint(ens_m.shape[0], size=n_new)   # ens_m: (Ne, Nv)
    new = ens_m[idx].copy()
    if jitter > 0:
        new += jitter * np.random.randn(*new.shape)
    return new

def lowrank_gaussian_spawn(ens, Ne_new, ridge=1e-6):
    Nv, Ne = ens.shape
    mu = ens.mean(axis=1, keepdims=True)               # (Nv, 1)
    X  = ens - mu                                      # (Nv, Ne)
    A = X / np.sqrt(Ne - 1)                            # (Nv, Ne)
    z = np.random.randn(Ne, Ne_new)                    # (Ne, Ne_new)
    ens_new = mu + A @ z                               # (Nv, Ne_new)
    # optional tiny ridge jitter to avoid rank-deficiency artifacts
    if ridge > 0:
        ens_new += np.sqrt(ridge) * np.random.randn(*ens_new.shape)
    return ens_new

np.random.seed(0)

########################## load data ############################
data = np.load('../data/L63/L63_data.npz')
N_gap = data['N_gap'].item()
truth_full = np.concatenate((data['x_truth'][:,None], data['y_truth'][:,None], data['z_truth'][:,None]), axis=1)[::N_gap]
obs_full = np.concatenate((data['x_obs'][:,None], data['y_obs'][:,None], data['z_obs'][:,None]), axis=1)
S_obs_full = data['S_obs'] # true regimes
train_size = 6400 # training data size
test_size = 1600  # test data size

######################### clustering ############################
with open('../data/L63/L63_FCM_2regimes.pkl', 'rb') as f:
    cluster_file = pickle.load(f)
cluster_model = FCMEntropy(**cluster_file['config'])
cluster_model.centers = cluster_file['centers']
cluster_model.weights = cluster_file['weights']
cluster_scaler = cluster_file['scaler']
n_regimes = cluster_file['n_cluster'] # number of clusters (regimes)
L = cluster_file['t_window'] # time delay steps

######################### multi-model DA ###########################
# truth = truth_full[:train_size]
# obs = obs_full[:train_size]
# S_obs = S_obs_full[:train_size]
truth = truth_full[train_size:train_size+test_size]
obs = obs_full[train_size:train_size+test_size]
S_obs = S_obs_full[train_size:train_size+test_size]
Nt, Nx = obs.shape

# ---------------------- model parameters ---------------------
# Noise levels shared by all models
sigma_x = np.sqrt(2.0)
sigma_y = 1.0
sigma_z = 1.0
sigma_obs = 4
# Models
models = [
    {'sigma': 10, 'beta': 8/3, 'rho': 28},
    {'sigma': 20, 'beta': 5,   'rho': 10},
]
# Regimes
regimes = [
    {'sigma': 10, 'beta': 8/3, 'rho': 28},
    {'sigma': 20, 'beta': 5,   'rho': 10},
]
n_models = len(models) # number of models
n_regimes = len(regimes) # number of regimes
dt = 5e-3 # Time step size
mlocs = np.array([ix for ix in range(Nx)])
nmod = mlocs.shape[0] # number of model variables

# Model error
model_error = False
if model_error == True:
    model_error_data = np.load('../data/L63/L63_model_eval.npz')
    weight_matrix = model_error_data['weight_matrix']

# ------------------- observation parameters ------------------
obs_error_var = sigma_obs**2
dt_obs = 0.25
obs_freq_timestep = int(round(dt_obs / dt))
ylocs = mlocs
nobs = ylocs.shape[0]
nobstime = obs.shape[0]
R = np.eye(nobs) * obs_error_var
Hk = np.zeros((nobs, nmod))
for iobs in range(nobs):
    Hk[iobs, ylocs[iobs]] = 1.0

# ------------------------ DA parameters ------------------------
iobsbeg = 40  # analysis period start
iobsend = -1  # analysis period end
ensemble_size = 100
inflation_values = [1.025] #[1, 1.025, 1.05]    # provide multiple values if for tuning
localization_values = [3] #[1, 2, 3]     # provide multiple values if for tuning
ninf = len(inflation_values)
nloc = len(localization_values)
localize = 1 # localization: 1 for on / 0 for off
inflate = 1  # inflatin: 1 for on / 0 for off

# ---------------------- initialization -----------------------
ics = truth_full[:train_size]
n_ics = ics.shape[0]
prior_rmse_t = np.zeros((nobstime, ninf, nloc))
analy_rmse_t = np.zeros((nobstime, ninf, nloc))
prior_rmse = np.zeros((ninf, nloc))
analy_rmse = np.zeros((ninf, nloc))
pattern_corr = np.zeros((nmod, ninf, nloc))

# ----------------------- assimilation -------------------------
for iinf in range(ninf):
    inflation_value = inflation_values[iinf]
    print('----------------------------------')
    print('inflation=', inflation_value)
    for iloc in range(nloc):
        localization_value = localization_values[iloc]
        print('----------------------------------')
        print('localization=', localization_value)

        CMat = construct_GC(localization_value, mlocs, ylocs)   # localization matrix 
        prior_mean_model = np.zeros((nobstime, nmod, n_models))
        analy_mean_model = np.zeros((nobstime, nmod, n_models))
        prior_spread_model = np.zeros((nobstime, nmod, n_models))
        analy_spread_model = np.zeros((nobstime, nmod, n_models))
        prior_mean_mixture = np.zeros((nobstime, nmod))
        analy_mean_mixture = np.zeros((nobstime, nmod))
        prior_spread_mixture = np.zeros((nobstime, nmod))
        analy_spread_mixture = np.zeros((nobstime, nmod))
        prior_weights = np.zeros((nobstime, n_models))
        posterior_weights = np.zeros((nobstime, n_models))
        regime_weights = np.zeros((nobstime, n_regimes))
        S_ens = np.zeros((nobstime, ensemble_size))
        prior_mean_mixture[:L-1] = obs[:L-1]  # fill in obs
        analy_mean_mixture[:L-1] = obs[:L-1]  # fill in obs

        # initial states
        weights = np.array([1/n_models] * n_models, dtype=float) # uniform initial weights
        ensemble_sizes = allocate_ensemble(ensemble_size, weights)
        ensemble_indices = [np.arange(start, start + size).tolist() for start, size in zip(np.cumsum([0] + ensemble_sizes[:-1]), ensemble_sizes)]
        S0_ens = np.zeros(ensemble_size, dtype=int) # initial model
        for m in range(n_models):
            S0_ens[ensemble_indices[m]] = m
        ens = ics[np.random.randint(n_ics, size=ensemble_size)]

        t0 = time()
        for iassim in range(L-1, nobstime):
            # print(iassim)
            prior_weight = weights.copy()
            prior_weight[np.array(ensemble_sizes)==0] = 0.0
            prior_weights[iassim] = prior_weight / prior_weight.sum()
            log_weights = np.log(prior_weights[iassim] + 1e-30)

            S_ens[iassim] = S0_ens

            for m in range(n_models):
                if ensemble_sizes[m] == 0:
                    # weights[m] = 0 # posterior weight equals to zero
                    log_weights[m] = -np.inf
                elif ensemble_sizes[m] == 1:
                    # (no posterior updates for model weight and ensemble member)
                    ens_m = ens[ensemble_indices[m]]
                    prior_mean_model[iassim, :, m] = np.mean(ens_m, axis=0)
                    prior_spread_model[iassim, :, m] = prior_spread_model[iassim-1, :, m]
                    analy_mean_model[iassim, :, m] = np.mean(ens_m, axis=0)
                    analy_spread_model[iassim, :, m] = analy_spread_model[iassim-1, :, m]
                    # weights[m] = 0 # posterior weight equals to zero
                    log_weights[m] = -np.inf
                else:
                    ens_m = ens[ensemble_indices[m]]
                    prior_mean_m = np.mean(ens_m, axis=0)

                    # inflation RTPP
                    ens_m = prior_mean_m + (ens_m - prior_mean_m) * inflation_value if inflate == 1 else ens_m

                    # posterior model weights
                    obs_inc = obs[iassim] - Hk @ prior_mean_m
                    cov = Hk @ np.cov(ens_m.T) @ Hk.T + R
                    # likelihood = 1/(np.sqrt(np.linalg.det(cov))) * np.exp(-0.5 * obs_inc @ np.linalg.solve(cov, obs_inc))
                    # weights[m] = weights[m] * likelihood
                    # likelihood in log-space to avoid overflow
                    sign, logdet = np.linalg.slogdet(cov)
                    if sign <= 0:
                        raise ValueError("cov not SPD")
                    log_likelihood = -0.5 * (obs_inc @ np.linalg.solve(cov, obs_inc) + obs_inc.shape[0] * np.log(2*np.pi) + logdet)
                    log_weights[m] = log_weights[m] + log_likelihood

                    # EnKF serial update
                    prior_mean_model[iassim, :, m] = prior_mean_m
                    prior_spread_model[iassim, :, m] = np.std(ens_m, axis=0, ddof=1)
                    ens_m = eakf(ensemble_sizes[m], nobs, ens_m, Hk, obs_error_var, localize, CMat, obs[iassim])
                    ens[ensemble_indices[m]] = ens_m
                    analy_mean_model[iassim, :, m] = np.mean(ens_m, axis=0)
                    analy_spread_model[iassim, :, m] = np.std(ens_m, axis=0, ddof=1)

            # normalize to get posterior weights
            weights = np.exp(log_weights - np.max(log_weights))   # safe, no overflow
            weights = weights / np.sum(weights)
            posterior_weights[iassim] = weights

            # Gaussian mixture mean and covariance
            prior_mean_mixture[iassim] = np.sum(prior_weights[iassim] * prior_mean_model[iassim], axis=1)
            prior_spread_mixture[iassim] = np.sqrt(np.sum(prior_weights[iassim] * (prior_spread_model[iassim]**2 + (prior_mean_model[iassim] - prior_mean_mixture[iassim][:,None])**2), axis=1))
            analy_mean_mixture[iassim] = np.sum(weights * analy_mean_model[iassim], axis=1)
            analy_spread_mixture[iassim] = np.sqrt(np.sum(weights * (analy_spread_model[iassim]**2 + (analy_mean_model[iassim] - analy_mean_mixture[iassim][:,None])**2), axis=1))

            # allocate ensemble members according to posterior weights (distribution)
            ensemble_sizes = allocate_ensemble(ensemble_size, weights)
            ensemble_indices_new = reallocate_ens(ensemble_indices, ensemble_sizes) # greedy strategy with minimal movements
            ens_old = ens.copy()
            for m in range(n_models):
                S0_ens[ensemble_indices_new[m]] = m
                # adjusting initial condtions as well in order to match the posterior distribution
                add_indices_m = [x for x in ensemble_indices_new[m] if x not in set(ensemble_indices[m])] # get indices of the additional ensemble members to be added to model m
                # ens[add_indices_m] = analy_mean_model[iassim, :, m] + analy_spread_model[iassim, :, m] * np.random.randn(len(add_indices_m), nmod) # sample from the Gaussian posterior of model m
                ens_m = ens_old[ensemble_indices[m]]
                # sample new members that preserves cross-var correlations
                if len(add_indices_m) > 0:
                    if ens_m.shape[0] >= 20:
                        ens[add_indices_m] = lowrank_gaussian_spawn(ens_m.T, len(add_indices_m), ridge=1e-6).T
                    elif ens_m.shape[0] >= 2:
                        ens[add_indices_m] = bootstrap_spawn(ens_m, len(add_indices_m), jitter=1e-3 * np.nanstd(ens_m, axis=0).mean())
                    elif ens_m.shape[0] == 1:
                        print(f"model {m} has ensemble size 1!")
                        # replicate + jitter
                        jitter = 1e-3 * max(np.nanstd(ens, axis=0).mean(), 1e-6)
                        ens[add_indices_m] = ens_m[0] + jitter * np.random.randn(len(add_indices_m), nmod)  
                    else:
                        print(f"model {m} has ensemble size 0 but wants to be allocated more!")
            ensemble_indices = ensemble_indices_new

            if iassim < nobstime - 1:
                # compute prior weights of the next assimilation step via clustering
                features = []
                analy_mean_mixture[iassim+1] = obs[iassim+1]
                z_window_mean = np.mean(analy_mean_mixture[iassim-L+2:iassim+2, 2]) # (z)
                dz_window_mean = np.mean(np.abs(analy_mean_mixture[iassim-L+3:iassim+2, 2] - analy_mean_mixture[iassim-L+2:iassim+1, 2])) # (|dz|)
                features += [z_window_mean, dz_window_mean]
                feature_embedded = np.array(features)[None,:]  # (1, #features)
                feature_scaled = cluster_scaler.transform(feature_embedded)  # (1, #features)
                regime_weight = cluster_model.predict(feature_scaled)[0,:]
                regime_weights[iassim+1] = regime_weight
                if model_error:
                    weights = weight_matrix @ regime_weight  # adjust prior weights according to model errors (weight_matrix shape: M x K)
                    weights = np.clip(weights, 0.0, None)
                    weights = weights / weights.sum()   
                else:
                    weights = regime_weight
                    
                # multi-model ensemble forecast with continuous-time Markov process for model switching
                _, holding_parameters, routing_matrix = markov_generator(posterior_weights[iassim], weights, dt_obs)
                model = L63RegimeModel(models, routing_matrix, holding_parameters, sigma_x, sigma_y, sigma_z)
                x1_ens, y1_ens, z1_ens, S1_ens = model.ensemble_forecast(obs_freq_timestep, dt, ens[:,0], ens[:,1], ens[:,2], S0_ens, ensemble_size)
                ens[:,0] = x1_ens[:, -1]
                ens[:,1] = y1_ens[:, -1]
                ens[:,2] = z1_ens[:, -1]
                S0_ens = S1_ens[:, -1]
                ensemble_sizes = [np.sum(S0_ens==m) for m in range(n_models)]
                ensemble_indices = [np.where(S0_ens==m)[0].tolist() for m in range(n_models)]

        prior_rmse_t[:, iinf, iloc] = np.sqrt(np.mean((truth - prior_mean_mixture) ** 2, axis=1))
        analy_rmse_t[:, iinf, iloc] = np.sqrt(np.mean((truth - analy_mean_mixture) ** 2, axis=1))
        prior_rmse[iinf, iloc] = np.sqrt(np.mean((truth - prior_mean_mixture)[iobsbeg - 1: iobsend] ** 2))
        analy_rmse[iinf, iloc] = np.sqrt(np.mean((truth - analy_mean_mixture)[iobsbeg - 1: iobsend] ** 2))
        pattern_corr[:, iinf, iloc] = np.array([np.corrcoef(truth[iobsbeg-1:iobsend, ix], analy_mean_mixture[iobsbeg-1:iobsend, ix])[0, 1] for ix in range(nmod)])
        t1 = time()
        print('time used: {:.2f} hours'.format((t1-t0)/3600))

save = {
    'prior_mean_model': prior_mean_model,
    'analy_mean_model': analy_mean_model,
    'prior_spread_model': prior_spread_model,
    'analy_spread_model': analy_spread_model,
    'prior_mean_mixture': prior_mean_mixture,
    'analy_mean_mixture': analy_mean_mixture,
    'prior_spread_mixture': prior_spread_mixture,
    'analy_spread_mixture': analy_spread_mixture,
    'prior_weights': prior_weights,
    'posterior_weights': posterior_weights,
    'regime_weights': regime_weights,
    'S_ens': S_ens,
    'S_obs': S_obs,
    'prior_rmse_t': prior_rmse_t,
    'analy_rmse_t': analy_rmse_t,
    'prior_rmse': prior_rmse,
    'analy_rmse': analy_rmse,
    'pattern_corr': pattern_corr,
}
np.savez('../data/L63/L63_GMMMEnKF_full_obs.npz', **save)

prior_err = np.nan_to_num(prior_rmse, nan=999999)
analy_err = np.nan_to_num(analy_rmse, nan=999999)

# # uncomment these if for tuning inflation and localization
# minerr = np.min(prior_err)
# inds = np.where(prior_err == minerr)
# print('min prior mean rmse = {0:.6e}, inflation = {1:.3e}, localizaiton = {2:d}'.format(minerr, inflation_values[inds[0][0]], localization_values[inds[1][0]]))
# minerr = np.min(analy_err)
# inds = np.where(analy_err == minerr)
# print('min analy mean rmse = {0:.6e}, inflation = {1:.3e}, localizaiton = {2:d}'.format(minerr, inflation_values[inds[0][0]], localization_values[inds[1][0]]))

# uncomment these if for test
print('prior mean rmse = {0:.6e}, inflation = {1:.3e}, localizaiton = {2:d}'.format(prior_err[0,0], inflation_values[0], localization_values[0]))
print('analy mean rmse = {0:.6e}, inflation = {1:.3e}, localizaiton = {2:d}'.format(analy_err[0,0], inflation_values[0], localization_values[0]))
print('analy pattern corr = ', pattern_corr[:,0,0])