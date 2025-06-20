import numpy as np
from L63_noisy import L63RegimeModel

def compute_histogram(data, bins):
    hist, _ = np.histogramdd(data, bins=bins, density=False) 
    return hist / np.sum(hist)

def kl_divergence(p, q):
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))

def evaluate_model_error(A_data, A_model, bins=10):
    """
    Evaluate model error (KL divergence) in regime `k`.
    
    Parameters:
    - A_data: true values, shape (T, D)
    - A_model: model predictions, shape (T, D)
    - bins: number of bins per dimension (int or list of length D)
    
    Returns:
    - kl: scalar KL divergence
    - p_hat, q_hat: joint histograms
    - edges: bin edges
    """

    D = A_data.shape[1]
    combined = np.vstack([A_data, A_model])
    if isinstance(bins, int):
        bins = [bins] * D
    edges = [np.linspace(combined[:, d].min(), combined[:, d].max(), bins[d] + 1) for d in range(D)] # adaptive bins that functions as standardizing data

    p_hat = compute_histogram(A_data, bins=edges)
    q_hat = compute_histogram(A_model, bins=edges)

    kl = kl_divergence(p_hat, q_hat)
    return kl, p_hat, q_hat, edges

def evaluate_model(Model, params, S_obs, truth, Nt, N_gap, dt, lead_time, n_models, n_regimes, rho=2, verbose=False):
    """
    Compute model weights by evaluating KL-based scores for each model and regime.

    Parameters:
    - Model: model class (not instantiated)
    - params: tuple (models, sigma_x, sigma_y, sigma_z)
    - S_obs: array of regime IDs (length Nt)
    - truth: array of ground truth state (Nt, 3)
    - Nt: total number of time steps
    - N_gap: number of integration steps per forecast
    - dt: time step size
    - lead_time: assimilation step offset
    - n_models: number of models
    - n_regimes: number of regimes
    - rho: penalty factor in model score
    - verbose: print KL and score for each regime-model pair

    Returns:
    - weight_matrix: shape (n_models, n_regimes), normalized model scores
    - score_matrix: raw (unnormalized) scores
    - models: the model parameter list (as passed)
    """
    models, sigma_x, sigma_y, sigma_z = params
    holding_parameters = np.array([0.1, 0.1, 0.1]) # # arbitrary values since transition is forbidden from the routing matrix
    score_matrix = np.zeros((n_models, n_regimes))
    hist_data = {
    'x': [],
    'y': [],
    'z': [],
    'xyz': []
    }

    for regime_id in range(n_regimes):
        idx = np.where(S_obs == regime_id)[0] + lead_time
        idx = idx[idx < Nt]  # avoid index out of bounds
        X0 = truth[idx-lead_time]
        A_true = truth[idx]

        for model_id in range(n_models):    
            routing_matrix = np.zeros((n_models, n_models))
            routing_matrix[:, model_id] = 1
            model = Model(models, routing_matrix, holding_parameters, sigma_x, sigma_y, sigma_z)
            A_pred = np.zeros_like(A_true)
            Nt_pred = len(idx)
            for i in range(Nt_pred):
                x1, y1, z1, _ = model.forecast(N_gap, dt, X0[i,0], X0[i,1], X0[i,2], model_id)
                A_pred[i,0] = x1[-1]
                A_pred[i,1] = y1[-1]
                A_pred[i,2] = z1[-1]
                
            kl_x1, p_x1, q_x1, bins_x1 = evaluate_model_error(A_data=A_true[:,0][:,None], A_model=A_pred[:,0][:,None], bins=30)
            kl_y1, p_y1, q_y1, bins_y1 = evaluate_model_error(A_data=A_true[:,1][:,None], A_model=A_pred[:,1][:,None], bins=30)
            kl_z1, p_z1, q_z1, bins_z1 = evaluate_model_error(A_data=A_true[:,2][:,None], A_model=A_pred[:,2][:,None], bins=30)
            kl_xyz1, p_xyz1, q_xyz1, edges_xyz1 = evaluate_model_error(A_data=A_true, A_model=A_pred, bins=10)
            model_score = np.exp(-rho*kl_xyz1) # the constant parameter controls penalty to model errors
            score_matrix[model_id, regime_id] = model_score

            hist_data['x'].append((p_x1, q_x1, bins_x1, regime_id, model_id))
            hist_data['y'].append((p_y1, q_y1, bins_y1, regime_id, model_id))
            hist_data['z'].append((p_z1, q_z1, bins_z1, regime_id, model_id))
            hist_data['xyz'].append((p_xyz1, q_xyz1, edges_xyz1, regime_id, model_id))

            if verbose:
                print(f"KL divergence for x in regime {regime_id:d}, model {model_id:d}: {kl_x1:.4f}")
                print(f"KL divergence for y in regime {regime_id:d}, model {model_id:d}: {kl_y1:.4f}")
                print(f"KL divergence for z in regime {regime_id:d}, model {model_id:d}: {kl_z1:.4f}")
                print(f"KL divergence for (x,y,z) in regime {regime_id:d}, model {model_id:d}: {kl_xyz1:.4f}")
                print(f"model score in regime {regime_id:d}, model {model_id:d}: {model_score:.4e}")
    weight_matrix = score_matrix / np.sum(score_matrix, axis=0)

    return weight_matrix, score_matrix, models, hist_data


if __name__ == '__main__':
    np.random.seed(0)

    data = np.load('../data/data_L63.npz')
    dt = data['dt'].item()
    N_gap = data['N_gap'].item()
    dt_obs = data['dt_obs'].item()
    N_gap = data['N_gap'].item()
    truth_full = np.concatenate((data['x_truth'][:,None], data['y_truth'][:,None], data['z_truth'][:,None]), axis=1)
    truth_full = truth_full[::N_gap]
    S_obs = data['S_obs']
    T = len(S_obs)
    lead_time = 1
    sigma_x = np.sqrt(2.0)
    sigma_y = 1.0
    sigma_z = 1.0
    sigma_obs = 2 * np.sqrt(2)

    models = [
        {'sigma': 10, 'beta': 8/3, 'rho': 28},
        {'sigma': 20, 'beta': 5,   'rho': 10},
        {'sigma': 15, 'beta': 4,   'rho': 35}
    ]
    regimes = [
        {'sigma': 10, 'beta': 8/3, 'rho': 28},
        {'sigma': 20, 'beta': 5,   'rho': 10},
    ]
    holding_parameters = np.array([0.2, 0.3, 0.4])

    n_models = len(models)
    n_regimes = len(regimes)
    score_matrix = np.zeros((n_models, n_regimes))
    for model_id in range(n_models):    
        routing_matrix = np.zeros((n_models, n_models))
        routing_matrix[:, model_id] = 1
        model = L63RegimeModel(models, routing_matrix, holding_parameters, sigma_x, sigma_y, sigma_z)

        for regime_id in range(n_regimes):
            idx = np.where(S_obs == regime_id)[0]
            idx = idx + lead_time
            idx = idx[idx < T]  # avoid index out of bounds
            X0 = truth_full[idx-lead_time]
            A_true = truth_full[idx]
            A_pred = np.zeros_like(A_true)
            T_pred = len(idx)
            
            for i in range(T_pred):
                x1, y1, z1, _ = model.forecast(N_gap, dt, X0[i,0], X0[i,1], X0[i,2], model_id)
                A_pred[i,0] = x1[-1]
                A_pred[i,1] = y1[-1]
                A_pred[i,2] = z1[-1]

            kl_xyz_1, p_xyz_1, q_xyz_1, edges_xyz_1 = evaluate_model_error(A_data=A_true, A_model=A_pred, bins=10)
            model_score = np.exp(-2*kl_xyz_1)
            score_matrix[model_id, regime_id] = model_score

    weight_matrix = score_matrix / np.sum(score_matrix, axis=0)

    np.savez('../data/model_evaluation.npz', 
             weight_matrix=weight_matrix,
             score_matrix=score_matrix,
             models=models,
             regimes=regimes
            )