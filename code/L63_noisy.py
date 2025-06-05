import numpy as np


class L63RegimeModel:
    def __init__(self, regimes, routing_matrix, holding_parameters,
                 sigma_x=0.0, sigma_y=0.0, sigma_z=0.0):
        """
        Initialize the Lorenz-63 model with regime switching.
        """
        self.regimes = regimes
        self.routing_matrix = routing_matrix
        self.holding_parameters = holding_parameters
        self.n_regimes = len(regimes)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z

    def forecast(self, N, dt, x0, y0, z0, S0):
        """
        Simulate a single trajectory with regime switching.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        z = np.zeros(N)
        S = np.zeros(N, dtype=int)

        x[0], y[0], z[0], S[0] = x0, y0, z0, S0

        for i in range(1, N):
            current_regime = S[i - 1]
            holding_param = self.holding_parameters[current_regime]

            # Regime switching
            if np.random.rand() < holding_param * dt:
                S[i] = np.random.choice(self.n_regimes, p=self.routing_matrix[current_regime])
            else:
                S[i] = current_regime

            # Get regime parameters
            sigma, beta, rho = self._get_params(S[i])

            # Euler-Maruyama step with noise
            x[i] = x[i - 1] + sigma * (y[i - 1] - x[i - 1]) * dt + self.sigma_x * np.sqrt(dt) * np.random.randn()
            y[i] = y[i - 1] + (x[i - 1] * (rho - z[i - 1]) - y[i - 1]) * dt + self.sigma_y * np.sqrt(dt) * np.random.randn()
            z[i] = z[i - 1] + (x[i - 1] * y[i - 1] - beta * z[i - 1]) * dt + self.sigma_z * np.sqrt(dt) * np.random.randn()

        return x, y, z, S

    def ensemble_forecast(self, N, dt, x0, y0, z0, S0, ensemble_size):
        """
        Run an ensemble forecast of size `ensemble_size`.
        Returns arrays of shape (ensemble_size, N)
        """
        X = np.zeros((ensemble_size, N))
        Y = np.zeros((ensemble_size, N))
        Z = np.zeros((ensemble_size, N))
        S = np.zeros((ensemble_size, N), dtype=int)

        for i in range(ensemble_size):
            xi, yi, zi, Si = self.forecast(N, dt, x0[i], y0[i], z0[i], S0[i])
            X[i, :], Y[i, :], Z[i, :], S[i, :] = xi, yi, zi, Si

        return X, Y, Z, S

    def _get_params(self, regime_index):
        """
        Extract parameters for a given regime index.
        """
        regime = self.regimes[regime_index]
        return regime['sigma'], regime['beta'], regime['rho']


if __name__ == '__main__':
    # Parameters
    T = 1e3
    dt = 0.005
    dt_obs = 0.5
    N = int(round(T / dt))
    N_gap = int(round(dt_obs / dt))
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
        [0, 1, 0],
        [1, 0, 0],
        [0.8, 0.2, 0]
    ])
    holding_parameters = np.array([0.2, 0.3, 0.4])
    rate_matrix = np.zeros_like(routing_matrix)
    for i in range(n_regimes):
        rate_matrix[i, :] = routing_matrix[i, :] * holding_parameters[i]
        rate_matrix[i, i] = -holding_parameters[i]

    # Initialize states
    S0 = 0 # initial regime
    x0= 1.508870
    y0 = -1.531271
    z0 = 25.46091

    model = L63RegimeModel(regimes, routing_matrix, holding_parameters, sigma_x, sigma_y, sigma_z)
    x_truth, y_truth, z_truth, S = model.forecast(N, dt, x0, y0, z0, S0)

    # Generate observations
    x_obs = x_truth[::N_gap] + sigma_obs * np.random.randn(N // N_gap)
    y_obs = y_truth[::N_gap] + sigma_obs * np.random.randn(N // N_gap)
    z_obs = z_truth[::N_gap] + sigma_obs * np.random.randn(N // N_gap)
    S_obs = S[::N_gap]

    np.savez('../data/data_L63.npz',
             x_truth=x_truth,
             y_truth=y_truth,
             z_truth=z_truth,
             S=S,
             x_obs=x_obs,
             y_obs=y_obs,
             z_obs=z_obs,
             N=N,
             N_gap=N_gap,
             dt=dt,
             dt_obs=dt_obs,
             S_obs=S_obs,
             sigma_obs=sigma_obs,
             sigma_x=sigma_x,
             sigma_y=sigma_y,
         sigma_z=sigma_z)