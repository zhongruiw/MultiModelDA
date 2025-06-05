import numpy as np
from enkf import eakf, construct_GC
from L63_noisy import L63RegimeModel
from time import time

np.random.seed(0)

# --------------------- load data --------------------------
train_size = 1600
test_size = 400
data = np.load('../data/L63_data_singleregime.npz')
dt = data['dt'].item()
N_gap = data['N_gap'].item()
sigma_obs = data['sigma_obs'].item()
dt_obs = data['dt_obs'].item()
truth_full = np.concatenate((data['x_truth'][:,None], data['y_truth'][:,None], data['z_truth'][:,None]), axis=1)
truth_full = truth_full[::N_gap]
obs_full = np.concatenate((data['x_obs'][:,None], data['y_obs'][:,None], data['z_obs'][:,None]), axis=1)

# split training and test data set (training means tuning inflation and localzaiton)
data_size = train_size
# truth = truth_full[:train_size]
# obs = obs_full[:train_size]
truth = truth_full[train_size:train_size+data_size]
obs = obs_full[train_size:train_size+data_size]

# ---------------------- model parameters ---------------------
sigma_x = np.sqrt(2.0)
sigma_y = 1.0
sigma_z = 1.0
sigma_obs = 2 * np.sqrt(2)
# Regime switching (Continuous-time Markov process)
regimes = [
    {'sigma': 10, 'beta': 8/3, 'rho': 28},
    {'sigma': 20, 'beta': 5,   'rho': 10},
    {'sigma': 15, 'beta': 4,   'rho': 35}
]
n_regimes = len(regimes)
routing_matrix = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [0.8, 0.2, 0]
])
holding_parameters = np.array([0.2, 0.3, 0.4])
# dt = 5e-3 # Time step size
Nx = 3 # Number of grid points in each direction
mlocs = np.array([ix for ix in range(Nx)])
nmod = mlocs.shape[0]
model = L63RegimeModel(regimes, routing_matrix, holding_parameters, sigma_x, sigma_y, sigma_z)

# ------------------- observation parameters ------------------
obs_error_var = sigma_obs**2
obs_freq_timestep = N_gap
ylocs = mlocs
nobs = ylocs.shape[0]
nobstime = obs.shape[0]
Hk = np.zeros((nobs, nmod))
for iobs in range(nobs):
    Hk[iobs, ylocs[iobs]] = 1.0

# --------------------- DA parameters -----------------------
# analysis period
iobsbeg = 20
iobsend = -1

# eakf parameters
ensemble_size = 40
inflation_values = [1] # provide multiple values if for tuning
localization_values = [1] # provide multiple values if for tuning
ninf = len(inflation_values)
nloc = len(localization_values)
localize = 0 # localization: 1 for on / 0 for off
inflate = 0 # inflatin: 1 for on / 0 for off

# ---------------------- initialization -----------------------
ics = truth_full[:train_size]
n_ics = ics.shape[0]
S0_ens = np.tile(np.array(0), ensemble_size) # initial regime
ens0 = ics[np.random.randint(n_ics, size=ensemble_size)] # shape (Nens,3)
prior_rmse = np.zeros((nobstime,ninf,nloc))
analy_rmse = np.zeros((nobstime,ninf,nloc))
prior_err = np.zeros((ninf,nloc))
analy_err = np.zeros((ninf,nloc))

# ---------------------- assimilation -----------------------
for iinf in range(ninf):
    inflation_value = inflation_values[iinf]
    print('inflation:',inflation_value)
    
    for iloc in range(nloc):
        localization_value = localization_values[iloc]
        print('localization:',localization_value)

        ens = ens0
        prior_mean = np.zeros((nobstime, nmod))
        analy_mean = np.empty((nobstime, nmod))
        prior_spread = np.empty((nobstime, nmod))
        analy_spread = np.empty((nobstime, nmod))
        S_ens = np.empty((ensemble_size, nobstime))

        t0 = time()
        for iassim in range(0, nobstime):
            # print(iassim)
            obsstep = iassim * obs_freq_timestep + 1
            prior_mean[iassim] = np.mean(ens, axis=0)
            S_ens[:, iassim] = S0_ens
            # inflation RTPP
            if inflate == 1:
                ensp = (zens - prior_mean[iassim]) * inflation_value
                ens = prior_mean[iassim] + ensp
            prior_spread[iassim] = np.std(ens, axis=0, ddof=1)

            # localization matrix        
            CMat = construct_GC(localization_value, mlocs, ylocs)
            
            # serial update
            ens = eakf(ensemble_size, nobs, ens, Hk, obs_error_var, localize, CMat, obs[iassim])
            analy_mean[iassim] = np.mean(ens, axis=0)
            analy_spread[iassim] = np.std(ens, axis=0, ddof=1)

            # ensemble model integration
            if iassim < nobstime - 1:
                x1_ens, y1_ens, z1_ens, S1_ens = model.ensemble_forecast(obs_freq_timestep, dt, ens[:,0], ens[:,1], ens[:,2], S0_ens, ensemble_size)
                ens[:,0] = x1_ens[:, -1]
                ens[:,1] = y1_ens[:, -1]
                ens[:,2] = z1_ens[:, -1]
                S0_ens = S1_ens[:, -1]

        prior_rmse[:, iinf, iloc] = np.sqrt(np.mean((truth - prior_mean) ** 2, axis=1))
        analy_rmse[:, iinf, iloc] = np.sqrt(np.mean((truth - analy_mean) ** 2, axis=1))
        prior_err[iinf, iloc] = np.mean(prior_rmse[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err[iinf, iloc] = np.mean(analy_rmse[iobsbeg - 1: iobsend, iinf, iloc])
        t1 = time()
        print('time used: {:.2f} hours'.format((t1-t0)/3600))

save = {
    'analy_mean': analy_mean,
    'prior_mean': prior_mean,
    'spread_analy': analy_spread,
    'spread_prior': prior_spread,
    'prior_rmse': prior_rmse,
    'analy_rmse': analy_rmse,
    'inflation_values': np.array(inflation_values),
    'localization_values': np.array(localization_values),
}
np.savez('../data/enkf_L63.npz', **save)

prior_err = np.nan_to_num(prior_err, nan=999999)
analy_err = np.nan_to_num(analy_err, nan=999999)

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