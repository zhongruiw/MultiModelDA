import numpy as np
# from L63_noisy import L63RegimeModel

# def compute_histogram(data, bins):
#     hist, _ = np.histogramdd(data, bins=bins, density=False) 
#     return hist / np.sum(hist)

# def kl_divergence(p, q):
#     eps = 1e-12
#     p = np.clip(p, eps, 1.0)
#     q = np.clip(q, eps, 1.0)
#     return np.sum(p * np.log(p / q))

# def evaluate_model_error(A_data, A_model, bins=10):
#     """
#     Evaluate model error (KL divergence) in regime `k`.
    
#     Parameters:
#     - A_data: true values, shape (T, D)
#     - A_model: model predictions, shape (T, D)
#     - bins: number of bins per dimension (int or list of length D)
    
#     Returns:
#     - kl: scalar KL divergence
#     - p_hat, q_hat: joint histograms
#     - edges: bin edges
#     """

#     D = A_data.shape[1]
#     combined = np.vstack([A_data, A_model])
#     if isinstance(bins, int):
#         bins = [bins] * D
#     edges = [np.linspace(combined[:, d].min(), combined[:, d].max(), bins[d] + 1) for d in range(D)] # adaptive bins that functions as standardizing data

#     p_hat = compute_histogram(A_data, bins=edges)
#     q_hat = compute_histogram(A_model, bins=edges)

#     kl = kl_divergence(p_hat, q_hat)
#     return kl, p_hat, q_hat, edges

# def evaluate_model(Model, params, S_obs, truth, Nt, N_gap, dt, lead_time, n_models, n_regimes, rho=2, verbose=False):
#     """
#     Compute model weights by evaluating KL-based scores for each model and regime.

#     Parameters:
#     - Model: model class (not instantiated)
#     - params: tuple (models, sigma_x, sigma_y, sigma_z)
#     - S_obs: array of regime IDs (length Nt)
#     - truth: array of ground truth state (Nt, 3)
#     - Nt: total number of time steps
#     - N_gap: number of integration steps per forecast
#     - dt: time step size
#     - lead_time: assimilation step offset
#     - n_models: number of models
#     - n_regimes: number of regimes
#     - rho: penalty factor in model score
#     - verbose: print KL and score for each regime-model pair

#     Returns:
#     - weight_matrix: shape (n_models, n_regimes), normalized model scores
#     - score_matrix: raw (unnormalized) scores
#     - models: the model parameter list (as passed)
#     """
#     models, sigma_x, sigma_y, sigma_z = params
#     holding_parameters = np.array([0.1, 0.1, 0.1]) # # arbitrary values since transition is forbidden from the routing matrix
#     score_matrix = np.zeros((n_models, n_regimes))
#     hist_data = {
#     'x': [],
#     'y': [],
#     'z': [],
#     'xyz': []
#     }

#     for regime_id in range(n_regimes):
#         idx = np.where(S_obs == regime_id)[0] + lead_time
#         idx = idx[idx < Nt]  # avoid index out of bounds
#         X0 = truth[idx-lead_time]
#         A_true = truth[idx]

#         for model_id in range(n_models):    
#             routing_matrix = np.zeros((n_models, n_models))
#             routing_matrix[:, model_id] = 1
#             model = Model(models, routing_matrix, holding_parameters, sigma_x, sigma_y, sigma_z)
#             A_pred = np.zeros_like(A_true)
#             Nt_pred = len(idx)
#             for i in range(Nt_pred):
#                 x1, y1, z1, _ = model.forecast(N_gap, dt, X0[i,0], X0[i,1], X0[i,2], model_id)
#                 A_pred[i,0] = x1[-1]
#                 A_pred[i,1] = y1[-1]
#                 A_pred[i,2] = z1[-1]
                
#             kl_x1, p_x1, q_x1, bins_x1 = evaluate_model_error(A_data=A_true[:,0][:,None], A_model=A_pred[:,0][:,None], bins=30)
#             kl_y1, p_y1, q_y1, bins_y1 = evaluate_model_error(A_data=A_true[:,1][:,None], A_model=A_pred[:,1][:,None], bins=30)
#             kl_z1, p_z1, q_z1, bins_z1 = evaluate_model_error(A_data=A_true[:,2][:,None], A_model=A_pred[:,2][:,None], bins=30)
#             kl_xyz1, p_xyz1, q_xyz1, edges_xyz1 = evaluate_model_error(A_data=A_true, A_model=A_pred, bins=10)
#             model_score = np.exp(-rho*kl_xyz1) # the constant parameter controls penalty to model errors
#             score_matrix[model_id, regime_id] = model_score

#             hist_data['x'].append((p_x1, q_x1, bins_x1, regime_id, model_id))
#             hist_data['y'].append((p_y1, q_y1, bins_y1, regime_id, model_id))
#             hist_data['z'].append((p_z1, q_z1, bins_z1, regime_id, model_id))
#             hist_data['xyz'].append((p_xyz1, q_xyz1, edges_xyz1, regime_id, model_id))

#             if verbose:
#                 print(f"KL divergence for x in regime {regime_id:d}, model {model_id:d}: {kl_x1:.4f}")
#                 print(f"KL divergence for y in regime {regime_id:d}, model {model_id:d}: {kl_y1:.4f}")
#                 print(f"KL divergence for z in regime {regime_id:d}, model {model_id:d}: {kl_z1:.4f}")
#                 print(f"KL divergence for (x,y,z) in regime {regime_id:d}, model {model_id:d}: {kl_xyz1:.4f}")
#                 print(f"model score in regime {regime_id:d}, model {model_id:d}: {model_score:.4e}")
#     weight_matrix = score_matrix / np.sum(score_matrix, axis=0)

#     return weight_matrix, score_matrix, models, hist_data

# -------------------------
# Helpers
# -------------------------
def _univariate_kl(x_true, x_pred, bins=30):
    """
    Estimate KL(p || q) for 1D variables using a common histogram.
    x_true, x_pred: (N,) arrays.
    """
    x_true = np.asarray(x_true, dtype=float).ravel()
    x_pred = np.asarray(x_pred, dtype=float).ravel()
    xmin = min(x_true.min(), x_pred.min())
    xmax = max(x_true.max(), x_pred.max())
    edges = np.linspace(xmin, xmax, bins + 1)
    H_true, _ = np.histogram(x_true, bins=edges)
    H_pred, _ = np.histogram(x_pred, bins=edges)
    P = H_true / np.sum(H_true)
    Q = H_pred / np.sum(H_pred)
    mask = (P > 0) & (Q > 0) # support restriction on KL(P || Q)
    kl = np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask])))
    return kl

def _joint_kl(A_true, A_pred, bins=10, max_dim=3):
    """
    Estimate KL(p || q) between truth and model using a joint histogram (up to `max_dim` dimensions).

    A_true, A_pred: (N, D) arrays (flattened variables).
    If D > max_dim, only the first max_dim dims are used.
    """
    A_true = np.asarray(A_true)
    A_pred = np.asarray(A_pred)
    N_true, D = A_true.shape
    D_use = min(D, max_dim)
    X = A_true[:, :D_use]
    Y = A_pred[:, :D_use]
    Z = np.vstack([X, Y])  # combined for common bin edges
    edges = [np.linspace(Z[:, d].min(), Z[:, d].max(), bins + 1) for d in range(D_use)] # bin edges per dimension
    H_true, _ = np.histogramdd(X, bins=edges)
    H_pred, _ = np.histogramdd(Y, bins=edges)
    P = H_true / np.sum(H_true)
    Q = H_pred / np.sum(H_pred)
    mask = (P > 0) & (Q > 0)  # support restriction on KL(P || Q)
    kl = np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask])))
    return kl

def _energy_distance(A_true, A_pred):
    """
    Energy distance between two empirical distributions:
        ED^2 = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||
    Returns ED (square root of ED^2, clipped to >=0).
    A_true, A_pred: (N, D) arrays.
    """
    X = np.asarray(A_true, dtype=float)
    Y = np.asarray(A_pred, dtype=float)
    # pairwise distances
    XX = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    YY = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=-1)
    XY = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
    ed2 = 2.0 * XY.mean() - XX.mean() - YY.mean()
    ed2 = max(ed2, 0.0)   # numerical safeguard
    return np.sqrt(ed2)


# -------------------------
# Main evaluator
# -------------------------
def evaluate_model(models, S_obs, truth, N_gap, dt, lead_time, n_regimes,
                   rho_mse=2.0, rho_kl=2.0, rho_ed=2.0, verbose=False, 
                   seq_len=None, bins_kl_joint=10, max_joint_dim_kl=3, bins_kl_pervar=None):
    """
    Evaluate a list of models per regime using:
      - Pointwise error: MSE at lead time (paired forecast vs truth)
      - Probabilistic error (KL): joint histogram KL between p(A_t | S_0=k) and p^M(A_t | S_0=k)
      - Probabilistic error (Energy Distance): ED between same
      - Probabilistic error (KL, per-variable): 1D KL per variable (optional)

    Parameters
    ----------
    models : list, Each model must implement: forecast(N_gap, dt, x0)
    S_obs : array-like, shape (Nt,) Regime IDs at each time t (for the initial-time index).
    truth : array-like, shape (Nt, n_vars) or (Nt, C, Nx). Ground truth trajectory.
    N_gap : int, Number of integration steps per forecast (lead time in model steps).
    dt : float, Time step (passed to the model; not used directly here).
    lead_time : int, Forecast lead time in observation steps: target index is t0 + lead_time.
    n_regimes : int, Number of regimes.
    rho_mse, rho_kl, rho_ed : float, Penalty factors for turning errors into scores via exp(-rho * error).
    verbose : bool, If True, print per-regime, per-model errors.
    seq_len=1 : int or None, History length used as input to model, default: None (seq_len=1, no history dim). 
    bins_kl_joint : int, Number of bins per dimension for joint histogram KL.
    max_joint_dim_kl : int, Maximum number of dimensions used for joint KL (for D > this, KL is set to NaN).
    bins_kl_pervar: int or None, Number of bins for per-variable histogram KL. Default: None (not computed).
    
    Returns
    -------
    results : dict with keys
        'mse'         : (n_models, n_regimes) MSE per model and regime
        'kl'          : (n_models, n_regimes) KL per model and regime (NaN if not computed)
        'ed'          : (n_models, n_regimes) Energy distance per model and regime
        'kl_pervar'   : (n_models, n_regimes, n_vars) KL per model, regime, and variable
        'weights_mse' : (n_models, n_regimes) weights from exp(-rho_mse * MSE)
        'weights_kl'  : (n_models, n_regimes) weights from exp(-rho_kl  * KL)
        'weights_ed'  : (n_models, n_regimes) weights from exp(-rho_ed  * ED)
    """
    S_obs = np.asarray(S_obs)
    truth = np.asarray(truth)
    if truth.ndim == 2:
        Nt, n_vars = truth.shape
    elif truth.ndim == 3:
        Nt, C, Nx = truth.shape
        n_vars = C * Nx
    n_models = len(models)

    # Error matrices
    mse_matrix = np.full((n_models, n_regimes), np.nan, dtype=float)
    kl_matrix  = np.full((n_models, n_regimes), np.nan, dtype=float)
    ed_matrix  = np.full((n_models, n_regimes), np.nan, dtype=float)
    kl_pervar_matrix = np.full((n_models, n_regimes, n_vars), np.nan, dtype=float)

    for regime_id in range(n_regimes):
        idx_init = np.where(S_obs == regime_id)[0] # indices of initial times
        idx_target = idx_init + lead_time  # indices of target times
        # avoid out-of-bounds
        mask_valid = (idx_target < Nt)
        idx_init   = idx_init[mask_valid]
        idx_target = idx_target[mask_valid]
        # ensure enough history for seq_len
        mask_hist = (idx_init - (seq_len - 1)) >= 0
        idx_init   = idx_init[mask_hist]
        idx_target = idx_target[mask_hist]
        if idx_init.size == 0:
            continue
        N_pred = idx_init.size
        A_true_flat = truth[idx_target].reshape(N_pred, n_vars)  # (N_pred, n_vars)

        for model_id, model in enumerate(models):
            # Forecasts
            A_pred_flat = np.zeros_like(A_true_flat)
            for i in range(N_pred):
                t0 = idx_init[i]
                if seq_len >= 1:
                    x0 = truth[t0 - (seq_len - 1): t0 + 1]  # (seq_len, *trailing_shape)
                elif seq_len == None:
                    x0 = truth[t0]                          # (*trailing_shape,)
                out = model.forecast(N_gap, dt, x0)         # (N_gap+1, *trailing_shape)
                A_pred_flat[i] = out[-1].reshape(n_vars)
            
            # ---------- 1) Pointwise MSE ----------
            mse_regime = np.mean((A_pred_flat - A_true_flat) ** 2)
            mse_matrix[model_id, regime_id] = mse_regime

            # ---------- 2) KL-based probabilistic error ----------
            kl_regime = _joint_kl(A_true_flat, A_pred_flat,
                                  bins=bins_kl_joint,
                                  max_dim=max_joint_dim_kl)
            kl_matrix[model_id, regime_id] = kl_regime

            # ---------- 3) Energy-distance probabilistic error ----------
            ed_regime = _energy_distance(A_true_flat, A_pred_flat)
            ed_matrix[model_id, regime_id] = ed_regime

            # ---------- 4) Per-variable KL (optional) ----------
            if bins_kl_pervar is not None:
                kl_pervar = np.zeros(n_vars, dtype=float)
                for v in range(n_vars):
                    kl_v = _univariate_kl(A_true_flat[:, v], A_pred_flat[:, v], bins=bins_kl_pervar)
                    kl_pervar[v] = kl_v
                kl_pervar_matrix[model_id, regime_id, :] = kl_pervar

            if verbose:
                print(f"[Regime {regime_id:d}, Model {model_id:d}] "
                      f"MSE={mse_regime:.4e}, KL={kl_regime:.4e}, ED={ed_regime:.4e}")

    # Turn errors into scores via exp(-rho * error), then normalize to weights
    def _errors_to_scores_weights(err_matrix, rho):
        n_models, n_regimes = err_matrix.shape
        scores  = np.zeros_like(err_matrix, dtype=float)
        weights = np.zeros_like(err_matrix, dtype=float)
        for k in range(n_regimes):
            col = err_matrix[:, k]
            valid = np.isfinite(col)
            if not np.any(valid):
                # no valid errors: assign uniform weights, zero scores
                if n_models > 0:
                    weights[:, k] = 1.0 / n_models
                continue
            # compute scores only for valid entries
            scores_k = np.zeros(n_models, dtype=float)
            scores_k[valid] = np.exp(-rho * col[valid])
            s = scores_k.sum()
            scores[:, k] = scores_k  # save raw scores
            if s > 0:
                weights[:, k] = scores_k / s
            else:
                # all scores ~0 -> uniform over valid
                weights[valid, k] = 1.0 / valid.sum()
        return scores, weights

    scores_mse, weights_mse = _errors_to_scores_weights(mse_matrix, rho_mse)
    scores_kl,  weights_kl  = _errors_to_scores_weights(kl_matrix,  rho_kl)
    scores_ed,  weights_ed  = _errors_to_scores_weights(ed_matrix,  rho_ed)

    results = {
        "mse": mse_matrix,
        "kl": kl_matrix,
        "ed": ed_matrix,
        "kl_pervar": kl_pervar_matrix,
        "scores_mse": scores_mse,
        "scores_kl": scores_kl,  # based on joint KL
        "scores_ed": scores_ed,
        "weights_mse": weights_mse,
        "weights_kl": weights_kl,
        "weights_ed": weights_ed,
    }
    return results



if __name__ == '__main__':
    from L63_noisy import L63RegimeModel
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