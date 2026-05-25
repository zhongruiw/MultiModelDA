import numpy as np
from enkf import eakf, construct_GC
from L63_noisy import L63RegimeModel
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
inflation_values = [10] #[1, 1.025, 1.05, 2, 5, 10]    # provide multiple values if for tuning
localization_values = [1] #[1, 2, 3]     # provide multiple values if for tuning
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
        prior_mean = np.zeros((nobstime, nmod))
        analy_mean = np.zeros((nobstime, nmod))
        prior_spread = np.zeros((nobstime, nmod))
        analy_spread = np.zeros((nobstime, nmod))
        S_ens = np.zeros((nobstime, ensemble_size))
        prior_mean[:L-1] = obs[:L-1]  # fill in obs
        analy_mean[:L-1] = obs[:L-1]  # fill in obs

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
            S_ens[iassim] = S0_ens

            # Inflate each model ensemble before assimilation
            if inflate:
                for m in range(n_models):
                    idx = ensemble_indices[m]
                    ens_m = ens[idx]
                    mean_m = np.mean(ens_m, axis=0, keepdims=True)
                    ens[idx] = mean_m + (ens_m - mean_m) * inflation_value
        
            prior_mean[iassim, :]   = np.mean(ens, axis=0)
            prior_spread[iassim, :] = np.std(ens, axis=0, ddof=1)
            
            # assimilate forecasts of M-1 models sequentially
            ens_0 = ens[ensemble_indices[0]]
            for m in range(1, n_models):
                # model m forecast as pseudo-observation
                obs_m = (Hk @ ens[ensemble_indices[m]].T).T  # (Nens_m, nobs), model m in obs space
                obs_mean_m = np.mean(obs_m, axis=0)          # (nobs,)
                obs_var_m  = np.var(obs_m, axis=0, ddof=1)   # (nobs,)
                # EnKF serial update
                ens_0 = eakf(ensemble_sizes[0], nobs, ens_0, Hk, obs_var_m, localize, CMat, obs_mean_m)
                
            # last assimilate real observation
            ens_0 = eakf(ensemble_sizes[0], nobs, ens_0, Hk, obs_error_var, localize, CMat, obs[iassim])
            
            analy_mean[iassim, :] = np.mean(ens_0, axis=0)
            analy_spread[iassim, :] = np.std(ens_0, axis=0, ddof=1)

            # set ICs for all models
            for m in range(n_models):
                if ensemble_sizes[m] == ensemble_sizes[0]:
                    ens[ensemble_indices[m]] = ens_0
                else:
                    ens[ensemble_indices[m]] = ens_0[:ensemble_sizes[m]]

            if iassim < nobstime - 1:
                # multi-model ensemble forecast with continuous-time Markov process for model switching
                routing_matrix = np.eye(n_models) # no model switching in forecast
                holding_parameters = np.array([1e-6]*n_models) # arbitrary values since the routing matrix forbids transition
                model = L63RegimeModel(models, routing_matrix, holding_parameters, sigma_x, sigma_y, sigma_z)
                x1_ens, y1_ens, z1_ens, S1_ens = model.ensemble_forecast(obs_freq_timestep, dt, ens[:,0], ens[:,1], ens[:,2], S0_ens, ensemble_size)
                ens[:,0] = x1_ens[:, -1]
                ens[:,1] = y1_ens[:, -1]
                ens[:,2] = z1_ens[:, -1]
                S0_ens = S1_ens[:, -1]

        prior_rmse_t[:, iinf, iloc] = np.sqrt(np.mean((truth - prior_mean) ** 2, axis=1))
        analy_rmse_t[:, iinf, iloc] = np.sqrt(np.mean((truth - analy_mean) ** 2, axis=1))
        prior_rmse[iinf, iloc] = np.sqrt(np.mean((truth - prior_mean)[iobsbeg - 1: iobsend] ** 2))
        analy_rmse[iinf, iloc] = np.sqrt(np.mean((truth - analy_mean)[iobsbeg - 1: iobsend] ** 2))
        pattern_corr[:, iinf, iloc] = np.array([np.corrcoef(truth[iobsbeg-1:iobsend, ix], analy_mean[iobsbeg-1:iobsend, ix])[0, 1] for ix in range(nmod)])
        t1 = time()
        print('time used: {:.2f} hours'.format((t1-t0)/3600))

save = {
    'prior_mean': prior_mean,
    'analy_mean': analy_mean,
    'prior_spread': prior_spread,
    'analy_spread': analy_spread,
    'S_ens': S_ens,
    'S_obs': S_obs,
    'prior_rmse': prior_rmse,
    'analy_rmse': analy_rmse,
    'pattern_corr': pattern_corr,
}
np.savez('../data/L63/L63_StdMMEnKF_full_obs.npz', **save)

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