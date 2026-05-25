import numpy as np

# -------------------------
# Helpers
# -------------------------
def _univariate_kl(x_true, x_pred, bins=30, return_hist=False):
    """
    Estimate KL(p || q) for 1D variables using a common histogram.
    x_true, x_pred: (N,) arrays.
    If return_hist=True : return (kl, P, Q, edges)
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
    if return_hist:
        return kl, P, Q, edges
    return kl

def _joint_kl(A_true, A_pred, bins=10, max_dim=3, eps=1e-12):
    A_true = np.asarray(A_true)
    A_pred = np.asarray(A_pred)
    N_true, D = A_true.shape
    D_use = min(D, max_dim)

    X = A_true[:, :D_use]
    Y = A_pred[:, :D_use]
    Z = np.vstack([X, Y])

    edges = [np.linspace(Z[:, d].min(), Z[:, d].max(), bins + 1) for d in range(D_use)]
    H_true, _ = np.histogramdd(X, bins=edges)
    H_pred, _ = np.histogramdd(Y, bins=edges)

    P = H_true / np.sum(H_true)
    Q = H_pred / np.sum(H_pred)

    # epsilon smoothing exactly like your old implementation
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    return np.sum(P * (np.log(P) - np.log(Q)))


def _joint_kl(A_true, A_pred, bins=10, max_dim=3, eps=1e-12):
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
    # # support restriction on KL(P || Q)
    # mask = (P > 0) & (Q > 0)  
    # kl = np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask])))
    # punish missing support
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    kl = np.sum(P * (np.log(P) - np.log(Q)))
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
                   seq_len=None, bins_kl_joint=10, max_joint_dim_kl=3,
                   bins_kl_pervar=None, save_hist_pervar=False, scales=None):
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
    scales : array-like, shape (n_vars,)

    Returns
    -------
    If save_hist_pervar=True, returns 'hist_pervar': hist_pervar[v]=(p_v, q_v, edges_v, regime_id, model_id, v)
    results : dict with keys
        'mse'         : (n_models, n_regimes) MSE per model and regime
        'kl'          : (n_models, n_regimes) KL per model and regime (NaN if not computed)
        'ed'          : (n_models, n_regimes) Energy distance per model and regime
        'kl_pervar'   : (n_models, n_regimes, n_vars) KL per model, regime, and variable
        'weights_mse' : (n_models, n_regimes) weights from exp(-rho_mse * MSE)
        'weights_kl'  : (n_models, n_regimes) weights from exp(-rho_kl  * KL)
        'weights_ed'  : (n_models, n_regimes) weights from exp(-rho_ed  * ED)
        'hist_pervar'  : list of length n_vars (or None if save_hist_pervar=False)
    """
    S_obs = np.asarray(S_obs)
    truth = np.asarray(truth)
    if truth.ndim == 2:
        Nt, n_vars = truth.shape
    elif truth.ndim == 3:
        Nt, C, Nx = truth.shape
        n_vars = C * Nx
    n_models = len(models)
    if scales is None:
        scales = np.ones((n_vars))
    else:
        scales = np.asarray(scales)

    # Error matrices
    mse_matrix = np.full((n_models, n_regimes), np.nan, dtype=float)
    kl_matrix  = np.full((n_models, n_regimes), np.nan, dtype=float)
    ed_matrix  = np.full((n_models, n_regimes), np.nan, dtype=float)
    kl_pervar_matrix = np.full((n_models, n_regimes, n_vars), np.nan, dtype=float)
    me_matrix  = np.full((n_models, n_regimes), np.nan, dtype=float)

    # Histogram storage for per-variable KL (optional)
    hist_pervar = None
    if save_hist_pervar:
        hist_pervar = [[] for _ in range(n_vars)] # list of (p_v, q_v, edges_v, regime_id, model_id, v)
    
    for regime_id in range(n_regimes):
        idx_init = np.where(S_obs == regime_id)[0] # indices of initial times
        idx_target = idx_init + lead_time  # indices of target times
        # avoid out-of-bounds
        mask_valid = (idx_target < Nt)
        idx_init   = idx_init[mask_valid]
        idx_target = idx_target[mask_valid]
        # ensure enough history for seq_len
        if seq_len == None:
            mask_hist = idx_init >= 0
        elif seq_len >= 1:
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
                if seq_len == None:
                    x0 = truth[t0]                          # (*trailing_shape,)
                elif seq_len >= 1:
                    x0 = truth[t0 - (seq_len - 1): t0 + 1]  # (seq_len, *trailing_shape)
                out = model.forecast(N_gap, dt, x0)         # (N_gap+1, *trailing_shape)
                A_pred_flat[i] = out[-1].reshape(n_vars)
            
            # ---------- 1) Pointwise MSE ----------
            mse_regime = np.mean(((A_pred_flat - A_true_flat) / scales[None, :]) ** 2)
            mse_matrix[model_id, regime_id] = mse_regime

            # ---------- 2) KL-based probabilistic error ----------
            kl_regime = _joint_kl(A_true_flat/scales[None, :], A_pred_flat/scales[None, :], bins=bins_kl_joint, max_dim=max_joint_dim_kl)
            kl_matrix[model_id, regime_id] = kl_regime

            # ---------- 3) Energy-distance probabilistic error ----------
            ed_regime = _energy_distance(A_true_flat/scales[None, :], A_pred_flat/scales[None, :])
            ed_matrix[model_id, regime_id] = ed_regime

            # ---------- 4) Per-variable KL (optional) ----------
            if bins_kl_pervar is not None:
                kl_pervar = np.zeros(n_vars, dtype=float)
                for v in range(n_vars):
                    if save_hist_pervar:
                        kl_v, p_v, q_v, edges_v = _univariate_kl(A_true_flat[:, v], A_pred_flat[:, v],bins=bins_kl_pervar, return_hist=True)
                        hist_pervar[v].append((p_v, q_v, edges_v, regime_id, model_id, v))
                    else:
                        kl_v = _univariate_kl(A_true_flat[:, v], A_pred_flat[:, v], bins=bins_kl_pervar)
                    kl_pervar[v] = kl_v
                kl_pervar_matrix[model_id, regime_id, :] = kl_pervar

            # ---------- 5) Pointwise Mean bias ----------
            me_regime = np.mean(((A_pred_flat - A_true_flat) / scales[None, :]))
            me_matrix[model_id, regime_id] = me_regime

            if verbose:
                print(f"[Regime {regime_id:d}, Model {model_id:d}] "
                      f"MSE={mse_regime:.4e}, KL={kl_regime:.4e}, ED={ed_regime:.4e}, ME={me_regime:.4e}")

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
        "me": me_matrix,
        "scores_mse": scores_mse,
        "scores_kl": scores_kl,  # based on joint KL
        "scores_ed": scores_ed,
        "weights_mse": weights_mse,
        "weights_kl": weights_kl,
        "weights_ed": weights_ed,
        "hist_pervar": hist_pervar,  # None if save_hist_pervar=False
    }
    return results


if __name__ == '__main__':
    import numpy as np
    import torch
    import xarray as xr
    from ENSO import CNNLSTM1D, ChannelZScoreScaler, AutoRegressiveModelSingle
    # from model_evaluation import evaluate_model
    import pickle

    with open('../data/ENSO_FCM_Obs_anomalies_4regimes.pkl', 'rb') as f:
        cluster_file = pickle.load(f)
    S_obs = cluster_file['labels']
    regime_weights = cluster_file['membership']
    ds = xr.open_dataset('../data/ENSO_Obs_anomalies.nc').sel(time=slice("1981-01", "2025-07"))
    arrays = [ds["sst_eq_anom"], ds["ssh_eq_anom"]]  # (time, lon) each
    truth_full = xr.concat(arrays, dim='var').transpose('time', 'var', 'lon').values # take real obs as truth
    L = cluster_file['t_window']           # time delay steps
    n_regimes = cluster_file['n_cluster']  # number of regimes
    truth = truth_full[L-1:, :2]
    Nt, Nv, Nx = truth.shape
    S_obs = S_obs[:Nt]
    regime_weights = regime_weights[:Nt]
    scales_var = np.array([1, .2]) # SST, SSH
    scales = np.repeat(scales_var, Nx)

    device = "cuda:1"
    model_names = [
                   # 'ACCESS-CM2_historical_r1i1p1f1',    # fair
                   # 'GFDL-CM4_historical_r1i1p1f1',      # best
                   # 'UKESM1-0-LL_historical_r1i1p1f2',   # good
                   'MIROC-ES2L_historical_r1i1p1f2',    # bad
                   # 'CMCC-ESM2_historical_r1i1p1f1',     # bad
                   'MPI-ESM1-2-LR_historical_r1i1p1f1', # bad
                   # 'CanESM5_historical_r1i1p1f1',
                  ]
    n_models = len(model_names)
    results_eval_by_regime = {}
    for reg in range(n_regimes):
        print(f"\n=== Evaluating models for regime {reg} ===")
        models = []
        for idx_model, name in enumerate(model_names):
            short_name = name.split("_")[0]
            print(f"model {idx_model}: {short_name}")
            model = CNNLSTM1D(
                in_channels=2,
                out_channels=2,
                latent_channels=40,
                hidden_channels=10,
                Nx=Nx,
                latent_dim=20,
                lstm_hidden_dim=64,
                lstm_layers=1,
            ).to(device)        
            scaler = ChannelZScoreScaler()
            checkpoint = torch.load(f"../model/ENSO_NNs4CMIP6_{name}_SingleModel_anomalies.pt", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            ar_model = AutoRegressiveModelSingle(model=model, scaler=scaler, device=device)
            models.append(ar_model)
        results_eval = evaluate_model(models, S_obs, truth, N_gap=3, 
                           dt=1/3, lead_time=1, n_regimes=n_regimes,
                           rho_mse=4.0, rho_kl=2.0, rho_ed=2.0, verbose=False, 
                           seq_len=1, bins_kl_joint=10, max_joint_dim_kl=3,
                           bins_kl_pervar=30, save_hist_pervar=True, scales=scales)
        results_eval_by_regime[reg] = results_eval

    # Build a hist_pervar_same that ONLY keeps same-regime pairs: e.g., for regime 0: entries with regime_id == 0
    hist_pervar_example = results_eval_by_regime[0]["hist_pervar"]
    n_vars_total = len(hist_pervar_example)
    hist_pervar_same = [[] for _ in range(n_vars_total)]
    for reg, res in results_eval_by_regime.items():
        hist_reg = res["hist_pervar"] 
        for v in range(n_vars_total):
            for (p_v, q_v, edges_v, regime_id, model_id, v_idx) in hist_reg[v]:
                # Keep ONLY same-regime entries: models of "reg" evaluated on regime "reg"
                if regime_id == reg:
                    hist_pervar_same[v].append((p_v, q_v, edges_v, regime_id, model_id, v_idx))

    # Save the same-regime weights
    weights_mse_same_reg = np.zeros((n_models, n_regimes))
    weights_kl_same_reg  = np.zeros((n_models, n_regimes))
    weights_ed_same_reg  = np.zeros((n_models, n_regimes))
    for reg, results_eval in results_eval_by_regime.items():
        weights_mse = results_eval["weights_mse"]
        weights_kl  = results_eval["weights_kl"]
        weights_ed  = results_eval["weights_ed"]
        weights_mse_same_reg[:, reg] = weights_mse[:, reg]
        weights_kl_same_reg[:, reg]  = weights_kl[:, reg]
        weights_ed_same_reg[:, reg]  = weights_ed[:, reg]
    weights_same_reg = {
        "mse": weights_mse_same_reg,
        "kl":  weights_kl_same_reg,
        "ed":  weights_ed_same_reg,
    }
    bundle = {
        "model_names": model_names,
        "n_regimes": n_regimes,
        "results_eval_by_regime": results_eval_by_regime,
        "hist_pervar_same": hist_pervar_same,
        "weights_same_reg": weights_same_reg,
    }
    with open("../data/ENSO_ModelEval_Results_4regimes2modelsweighted_anomalies.pkl", "wb") as f:
        pickle.dump(bundle, f)
