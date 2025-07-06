import numpy as np
from enkf import eakf, construct_GC
from L63_noisy import L63RegimeModel
from sklearn.preprocessing import StandardScaler
from cluster import FCMEntropy, reorder_clusters_by_centers
from model_evaluation import evaluate_model
from scipy.linalg import expm
from scipy.optimize import minimize
from collections import deque
import pickle
from time import time
    
def allocate_ensemble(ensemble_size, weights):
    '''input total ensemble size and model weights, return ensemble size for each model'''
    allocations = np.round(weights * ensemble_size).astype(int)
    excess = allocations.sum() - ensemble_size
    if excess > 0:
        max_idx = np.argmax(allocations)
        allocations[max_idx] -= excess
    elif excess < 0:
        min_idx = np.argmin(allocations)
        allocations[min_idx] -= excess  

    return allocations.tolist()

def reallocate_ens(initial_allocation, target_counts):
    '''reallocate ensemble members with minimal movements (pen for ensemble member, drawer for model)'''
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
    n = len(mu0)
    def objective(Q_flat):
        Q = Q_flat.reshape((n, n))
        P = expm(T * Q)
        mu_pred = mu0 @ P
        return np.sum((mu_pred - muT) ** 2)
    ineq_constr = {
        'type': 'ineq',
        'fun': lambda Q_flat: Q_flat.reshape((n, n))[~np.eye(n, dtype=bool)]
    }
    eq_constr = {
        'type': 'eq',
        'fun': lambda Q_flat: Q_flat.reshape((n, n)).sum(axis=1)
    }
    Q0 = np.random.rand(n, n)
    np.fill_diagonal(Q0, 0)
    Q0 = Q0 / Q0.sum(axis=1, keepdims=True)
    np.fill_diagonal(Q0, -Q0.sum(axis=1))
    Q0_flat = Q0.flatten()
    result = minimize(objective, Q0_flat, constraints=[ineq_constr, eq_constr], method='SLSQP')
    success = result.success
    if success:
        Q_opt = result.x.reshape((n, n))
        h = -np.diag(Q_opt)
        R = np.zeros_like(Q_opt)
        for i in range(len(Q_opt)):
            if h[i] > 0:
                R[i, :] = Q_opt[i, :] / h[i]
                R[i, i] = 0.0
        return Q_opt, h, R
    else:
        raise RuntimeError("Optimization failed: could not find a valid generator matrix.")
        
def MAP(means, covariances, weights):
    import numpy as np
    from scipy.stats import multivariate_normal
    from scipy.optimize import minimize
    def negative_log_gmm_density(x, means, covariances, weights):
        x = np.array(x)
        density = 0.0
        for k in range(len(weights)):
            mvn = multivariate_normal(mean=means[k], cov=covariances[k])
            density += weights[k] * mvn.pdf(x)
        return -np.log(density + 1e-12)  # avoid log(0)
    # Initialize at the mean of the component with highest weight
    init_idx = np.argmax(weights)
    x0 = means[init_idx]
    # Minimize the negative log GMM density
    map_estimate = minimize(negative_log_gmm_density, x0, args=(means, covariances, weights), method='L-BFGS-B')
    
    return map_estimate.x # Reture MAP estimate


np.random.seed(0)

########################## load data ############################
data = np.load('../data/data_L63.npz')
N_gap = data['N_gap'].item()
truth_full = np.concatenate((data['x_truth'][:,None], data['y_truth'][:,None], data['z_truth'][:,None]), axis=1)[::N_gap]
obs_full = np.concatenate((data['x_obs'][:,None], data['y_obs'][:,None], data['z_obs'][:,None]), axis=1)
S_obs_full = data['S_obs'] # true regimes
train_size = 6400 # training data size
test_size = 1600 # test data size

######################### clustering ############################
K = 2 # number of clusters (regimes)
L = 4 # time delay steps
# scaler = StandardScaler()
# data = truth_full[:train_size]
# S_obs = S_obs_full[:train_size]
# Nt, _ = data.shape
# features = []
# features.append(np.mean(np.stack([data[i:Nt+i-L+1, 2][:,None] for i in range(L-1)], axis=2), axis=2)) # (z)
# features.append(np.mean(np.stack([np.abs(data[i:Nt-L+i+1, 2]-data[i-1:Nt-L+i, 2])[:,None] for i in range(1,L-1)], axis=2), axis=2)) # (|dz|)
# features.append(np.mean(np.stack([np.abs(data[i:Nt-L+i+1, 0]-data[i-1:Nt-L+i, 0])[:,None] for i in range(1,L)], axis=2), axis=2)) # (|dx|)
# # features.append(np.ptp(np.stack([data[i:Nt-L+i+1, 0][:,None] for i in range(L)], axis=2), axis=2)) # range(x0,x1,...)
# # features.append(np.ptp(np.stack([data[i:Nt-L+i+1, 2][:,None] for i in range(L)], axis=2), axis=2)) # range(z0,z1,...)
# data_embedded = np.concatenate(features, axis=1)

# # standardize features
# data_scaled = scaler.fit_transform(data_embedded)

# # clustering
# cluster_model = FCMEntropy(num_clusters=K, m=2.5, lambda_e=1, num_steps=500, seed=0)
# results = cluster_model.fit(data_scaled, optimizer='gradient_descent')

# # adjust the order of cluster centers to align with the prescibed regime order 
# reference_centers = np.stack([np.mean(data_scaled[S_obs[L-1:]==k], axis=0) for k in range(K)], axis=0)
# perm = reorder_clusters_by_centers(cluster_model.centers, reference_centers)
# cluster_model.centers = cluster_model.centers[perm]
# cluster_model.params = (cluster_model.params[0], cluster_model.centers, cluster_model.params[2])
# results['membership'] = results['membership'][:, perm]

# # save the clustering model
# fcm_model_data = {
#     'centers': cluster_model.centers,
#     'weights': cluster_model.weights,
#     'scaler': scaler,
#     'config': {
#         'num_clusters': cluster_model.num_clusters,
#         'm': cluster_model.m,
#         'lambda_e': cluster_model.lambda_e,
#     }}
# with open('../data/fcm_model_L63_partialobx.pkl', 'wb') as f:
#     pickle.dump(fcm_model_data, f)

# load the clustering model
with open('../data/fcm_model_L63_partialobx.pkl', 'rb') as f:
    fcm_model_data = pickle.load(f)
cluster_model = FCMEntropy(**fcm_model_data['config'])
cluster_model.centers = fcm_model_data['centers']
cluster_model.weights = fcm_model_data['weights']
scaler = fcm_model_data['scaler']

# """
######################### multi-model DA ###########################
# truth = truth_full[:train_size]
# obs = obs_full[:train_size]
# S_obs = S_obs_full[:train_size]
truth = truth_full[train_size:train_size+test_size]
obs = obs_full[train_size:train_size+test_size, 0][:, None]
S_obs = S_obs_full[train_size:train_size+test_size]
Nt, _ = obs.shape

# ---------------------- model parameters ---------------------
# Noise levels shared by all models
sigma_x = np.sqrt(2.0)
sigma_y = 1.0
sigma_z = 1.0
sigma_obs = 4
# Models
models = [
    {'sigma': 14, 'beta': 8/3, 'rho': 24},
    {'sigma': 16, 'beta': 5,   'rho': 14},
    # {'sigma': 15, 'beta': 4,   'rho': 35}
]
# Regimes
regimes = [
    {'sigma': 10, 'beta': 8/3, 'rho': 28},
    {'sigma': 20, 'beta': 5,   'rho': 10},
]
n_models = len(models) # number of models
n_regimes = len(regimes) # number of regimes
dt = 5e-3 # Time step size
Nx = 3 # Number of grid points
mlocs = np.array([ix for ix in range(Nx)])
nmod = mlocs.shape[0] # number of model variables

# model error evaluation
model_error = True
lead_time = 1 # one assimilation step
params = (models, sigma_x, sigma_y, sigma_z)
if model_error == True:
    # weight_matrix, score_matrix, _ = evaluate_model(L63RegimeModel, params, S_obs, truth, Nt, N_gap, dt, lead_time, n_models, n_regimes, rho=2, verbose=True)
    # np.savez('../data/model_evaluation_L63.npz', 
    #          weight_matrix=weight_matrix,
    #          score_matrix=score_matrix,
    #          models=models,
    #          regimes=regimes
    #         )
    model_error_data = np.load('../data/model_evaluation_L63.npz')
    weight_matrix = model_error_data['weight_matrix']

# ------------------- observation parameters ------------------
obs_error_var = sigma_obs**2
dt_obs = 0.25
obs_freq_timestep = int(round(dt_obs / dt))
ylocs = np.array([0])
nobs = ylocs.shape[0]
nobstime = obs.shape[0]
R = np.eye(nobs) * obs_error_var
Hk = np.zeros((nobs, nmod))
for iobs in range(nobs):
    Hk[iobs, ylocs[iobs]] = 1.0

# ------------------------ DA parameters ------------------------
# analysis period
iobsbeg = 40
iobsend = -1
# eakf parameters
ensemble_size = 100
inflation_values = [1] # provide multiple values if for tuning
localization_values = [1] # provide multiple values if for tuning
ninf = len(inflation_values)
nloc = len(localization_values)
localize = 0 # localization: 1 for on / 0 for off
inflate = 0 # inflatin: 1 for on / 0 for off

# ---------------------- initialization -----------------------
ics = truth_full[:train_size]
n_ics = ics.shape[0]
weights = np.array([1/n_models] * n_models, dtype=float) # uniform initial weights
ensemble_sizes = allocate_ensemble(ensemble_size, weights)
ensemble_indices = [np.arange(start, start + size).tolist() for start, size in zip(np.cumsum([0] + ensemble_sizes[:-1]), ensemble_sizes)]
S0_ens = np.zeros(ensemble_size, dtype=int) # initial model
for m in range(n_models):
    S0_ens[ensemble_indices[m]] = m
ens0 = ics[np.random.randint(n_ics, size=ensemble_size)]
prior_rmse = np.zeros((nobstime,ninf,nloc))
analy_rmse = np.zeros((nobstime,ninf,nloc))
prior_err = np.zeros((ninf,nloc))
analy_err = np.zeros((ninf,nloc))
pattern_corr = np.zeros((nmod, ninf, nloc))

# ----------------------- assimilation -------------------------
for iinf in range(ninf):
    inflation_value = inflation_values[iinf]
    for iloc in range(nloc):
        localization_value = localization_values[iloc]

        prior_mean_model = np.zeros((nobstime, nmod, n_models))
        analy_mean_model = np.zeros((nobstime, nmod, n_models))
        prior_spread_model = np.zeros((nobstime, nmod, n_models))
        analy_spread_model = np.zeros((nobstime, nmod, n_models))
        prior_mean_mixture = np.zeros((nobstime, nmod))
        analy_mean_mixture = np.zeros((nobstime, nmod))
        prior_map_mixture = np.zeros((nobstime, nmod))
        analy_map_mixture = np.zeros((nobstime, nmod))
        prior_spread_mixture = np.zeros((nobstime, nmod))
        analy_spread_mixture = np.zeros((nobstime, nmod))
        prior_weights = np.zeros((nobstime, n_models))
        posterior_weights = np.zeros((nobstime, n_models))
        S_ens = np.zeros((nobstime, ensemble_size))
        ens = ens0

        t0 = time()
        for iassim in range(L-2, nobstime):
            prior_weights[iassim] = weights
            S_ens[iassim] = S0_ens
            analy_cov_model = []
            
            for m in range(n_models):
                if ensemble_sizes[m] == 0:
                    weights[m] = 0 # posterior weight equals to zero
                    analy_cov_model.append(np.eye(nmod))
                elif ensemble_sizes[m] == 1:
                    # (no posterior updates for model weight and ensemble member)
                    ens_m = ens[ensemble_indices[m]]
                    prior_mean_model[iassim, :, m] = np.mean(ens_m, axis=0)
                    prior_spread_model[iassim, :, m] = prior_spread_model[iassim-1, :, m]
                    analy_mean_model[iassim, :, m] = np.mean(ens_m, axis=0)
                    analy_spread_model[iassim, :, m] = analy_spread_model[iassim-1, :, m]
                    weights[m] = 0 # posterior weight equals to zero
                    analy_cov_model.append(np.eye(nmod))
                else:
                    ens_m = ens[ensemble_indices[m]]
                    prior_mean_m = np.mean(ens_m, axis=0)

                    # inflation RTPP
                    ens_m = prior_mean_m + (ens_m - prior_mean_m) * inflation_value if inflate == 1 else ens_m
                    # localization matrix        
                    CMat = construct_GC(localization_value, mlocs, ylocs)

                    # posterior model weights
                    obs_inc = obs[iassim] - Hk @ prior_mean_m
                    cov = Hk @ np.cov(ens_m.T) @ Hk.T + R
                    likelihood = 1/(np.sqrt(np.linalg.det(cov))) * np.exp(-0.5 * obs_inc @ np.linalg.inv(cov) @ obs_inc)
                    weights[m] = weights[m] * likelihood

                    # EnKF serial update
                    prior_mean_model[iassim, :, m] = prior_mean_m
                    prior_spread_model[iassim, :, m] = np.std(ens_m, axis=0, ddof=1)
                    ens_m = eakf(ensemble_sizes[m], nobs, ens_m, Hk, obs_error_var, localize, CMat, obs[iassim])
                    ens[ensemble_indices[m]] = ens_m
                    analy_mean_model[iassim, :, m] = np.mean(ens_m, axis=0)
                    analy_spread_model[iassim, :, m] = np.std(ens_m, axis=0, ddof=1)
                    analy_cov_model.append(np.cov(ens_m.T, ddof=1))

            # normalize to get posterior weights
            weights = weights / np.sum(weights)
            posterior_weights[iassim] = weights

            # Gaussian mixture mean and covariance
            prior_mean_mixture[iassim] = np.sum(prior_weights[iassim] * prior_mean_model[iassim], axis=1)
            prior_spread_mixture[iassim] = np.sqrt(np.sum(prior_weights[iassim] * (prior_spread_model[iassim]**2 + (prior_mean_model[iassim] - prior_mean_mixture[iassim][:,None])**2), axis=1))
            analy_mean_mixture[iassim] = np.sum(weights * analy_mean_model[iassim], axis=1)
            analy_spread_mixture[iassim] = np.sqrt(np.sum(weights * (analy_spread_model[iassim]**2 + (analy_mean_model[iassim] - analy_mean_mixture[iassim][:,None])**2), axis=1))

            # MAP estimate
            analy_map_mixture[iassim] = MAP(analy_mean_model[iassim].T, analy_cov_model, weights)
            
            # allocate ensemble members according to posterior weights (distribution)
            ensemble_sizes = allocate_ensemble(ensemble_size, weights)
            ensemble_indices_new = reallocate_ens(ensemble_indices, ensemble_sizes) # greedy strategy with minimal movements
            for m in range(n_models):
                S0_ens[ensemble_indices_new[m]] = m
                # adjusting initial condtions as well in order to match the posterior distribution
                add_indices_m = [x for x in ensemble_indices_new[m] if x not in set(ensemble_indices[m])] # get indices of the additional ensemble members to be added to model m
                ens[add_indices_m] = analy_mean_model[iassim, :, m] + analy_spread_model[iassim, :, m] * np.random.randn(len(add_indices_m), nmod) # sample from the Gaussian posterior of model m
            ensemble_indices = ensemble_indices_new
            
            if iassim < nobstime - 1:
                # compute prior weights of the next assimilation step via clustering
                features = []
                analy_mean_mixture[iassim+1] = obs[iassim+1]
                features.append(np.mean([analy_mean_mixture[iassim-L+2+i, 2] for i in range(L-1)])) # (z)
                features.append(np.mean(np.abs(analy_mean_mixture[iassim-L+3:iassim+1, 2] - analy_mean_mixture[iassim-L+2:iassim, 2]))) # (|dz|)
                features.append(np.mean(np.abs(analy_mean_mixture[iassim-L+3:iassim+2, 0] - analy_mean_mixture[iassim-L+2:iassim+1, 0]))) # (|dx|)
                features = scaler.transform(np.array(features)[None,:])
                weights = cluster_model.predict(features)[0,:]
                # weights = np.array([1-S_obs[iassim+1], S_obs[iassim+1]], dtype=float) # true regime
                if model_error:
                    weights = weight_matrix @ weights # adjust prior weights according to model errors (weight_matrix shape: M x K)
                    
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

        prior_rmse[:, iinf, iloc] = np.sqrt(np.mean((truth - prior_mean_mixture) ** 2, axis=1))
        analy_rmse[:, iinf, iloc] = np.sqrt(np.mean((truth - analy_mean_mixture) ** 2, axis=1))
        prior_err[iinf, iloc] = np.mean(prior_rmse[iobsbeg - 1: iobsend, iinf, iloc])
        analy_err[iinf, iloc] = np.mean(analy_rmse[iobsbeg - 1: iobsend, iinf, iloc])
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
    'analy_map_mixture': analy_map_mixture,
    'prior_spread_mixture': prior_spread_mixture,
    'analy_spread_mixture': analy_spread_mixture,
    'prior_weights': prior_weights,
    'posterior_weights': posterior_weights,
    'S_ens': S_ens,
    'prior_rmse': prior_rmse,
    'analy_rmse': analy_rmse,
    'pattern_corr': pattern_corr,
}
# np.savez('../data/MultimodelEnKF_L63_modelerror_partialobsx.npz', **save)

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
print('analy pattern corr = ', pattern_corr[:,0,0])
# """