import numpy as np


class OUNonlinearModel:
    def __init__(self, omega=2, f_u=0, a=2.8, b=8, c=4, f_v=-0.4,
                 sigma_u=0.2, sigma_v=0.7):
        """
        Initialize the model:
        a nonlinear Ornstein–Uhlenbeck system with 
        cubic damping in the auxiliary variable `v`:
        
        du_r = [ -v * u_r - omega * u_i + f_u ] dt + sigma_u * dW_1
        du_i = [ -v * u_i + omega * u_r + f_u ] dt + sigma_u * dW_2
        dv   = [ -a*v + b*v^2 - c*v^3 + f_v ] dt + sigma_v * dW_3
        """
        self.omega = omega
        self.f_u = f_u
        self.a = a
        self.b = b
        self.c = c
        self.f_v = f_v
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v

    def forecast(self, N, dt, ur0, ui0, v0):
        """
        Simulate a single trajectory using the Euler-Maruyama method.
        """
        omega = self.omega
        f_u = self.f_u
        a = self.a
        b = self.b
        c = self.c
        f_v = self.f_v
        sigma_u = self.sigma_u
        sigma_v = self.sigma_v

        ur = np.zeros(N)
        ui = np.zeros(N)
        v = np.zeros(N)
        ur[0], ui[0], v[0] = ur0, ui0, v0

        for i in range(1, N):
            ur[i] = ur[i - 1] + (-v[i-1] * ur[i - 1] - omega * ui[i - 1] + f_u) * dt \
                    + sigma_u * np.sqrt(dt) * np.random.randn()
            ui[i] = ui[i - 1] + (-v[i-1] * ui[i - 1] + omega * ur[i - 1] + f_u) * dt \
                    + sigma_u * np.sqrt(dt) * np.random.randn()
            v[i] = v[i - 1] + (-a * v[i - 1] + b * v[i - 1]**2 - c * v[i - 1]**3 + f_v) * dt \
                   + sigma_v * np.sqrt(dt) * np.random.randn()

        return ur, ui, v

    def ensemble_forecast(self, N, dt, ur0, ui0, v0, ensemble_size):
        """
        Run an ensemble forecast of size `ensemble_size`.
        Returns arrays of shape (ensemble_size, N)
        """
        UR = np.zeros((ensemble_size, N))
        UI = np.zeros((ensemble_size, N))
        V = np.zeros((ensemble_size, N))

        for i in range(ensemble_size):
            uri, uii, vi = self.forecast(N, dt, ur0[i], ui0[i], v0[i])
            UR[i, :], UI[i, :], V[i, :] = uri, uii, vi

        return UR, UI, V

class MeanStochasticModel:
    def __init__(self, regimes=[1], routing_matrix=None, holding_parameters=None, sigma_u=0.2):
        """
        Initialize the model:
        a liner Ornstein–Uhlenbeck system with 
        damping term 'v' calibrated by the equilibrium statistics,
        regime switching descibed by a Markov jump process.
        
        du_r = [ -v * u_r - omega * u_i + f_u ] dt + sigma_u * dW_1
        du_i = [ -v * u_i + omega * u_r + f_u ] dt + sigma_u * dW_2
        """
        # self.omega = omega
        # self.v = v
        # self.f_u = f_u
        self.regimes = regimes
        self.routing_matrix = routing_matrix
        self.holding_parameters = holding_parameters
        self.n_regimes = len(regimes)
        self.sigma_u = sigma_u

    def forecast(self, N, dt, ur0, ui0, S0):
        """
        Simulate a single trajectory using the Euler-Maruyama method.
        """
        sigma_u = self.sigma_u

        ur = np.zeros(N)
        ui = np.zeros(N)
        S = np.zeros(N, dtype=int)
        ur[0], ui[0], S[0] = ur0, ui0, S0

        for i in range(1, N):
            current_regime = S[i - 1]
            holding_param = self.holding_parameters[current_regime]

            # Regime switching
            if np.random.rand() < holding_param * dt:
                S[i] = np.random.choice(self.n_regimes, p=self.routing_matrix[current_regime])
            else:
                S[i] = current_regime

            # Get regime parameters
            regime = self.regimes[S[i]]
            v, omega, f_u = regime['v'], regime['omega'], regime['f_u']

            # Euler-Maruyama step with noise
            ur[i] = ur[i - 1] + (-v * ur[i - 1] - omega * ui[i - 1] + f_u) * dt \
                    + sigma_u * np.sqrt(dt) * np.random.randn()
            ui[i] = ui[i - 1] + (-v * ui[i - 1] + omega * ur[i - 1] + f_u) * dt \
                    + sigma_u * np.sqrt(dt) * np.random.randn()

        return ur, ui, S

    def ensemble_forecast(self, N, dt, ur0, ui0, S0, ensemble_size):
        """
        Run an ensemble forecast of size `ensemble_size`.
        Returns arrays of shape (ensemble_size, N)
        """
        UR = np.zeros((ensemble_size, N))
        UI = np.zeros((ensemble_size, N))
        S = np.zeros((ensemble_size, N), dtype=int)

        for i in range(ensemble_size):
            uri, uii, Si = self.forecast(N, dt, ur0[i], ui0[i], S0[i])
            UR[i, :], UI[i, :], S[i, :] = uri, uii, Si

        return UR, UI, S

    def calibrate(self, u_t, dt, Lag):
        """
        calibration of complex OU process
        - Lag: int, lag for computing the ACF
        """
        from scipy.optimize import curve_fit
        from statsmodels.tsa.stattools import acf, ccf

        def CCF(data, gamma, omega):
            '''
            Ansatz of cross-correlation between real and imaginary parts 
            '''
            return np.exp(-gamma*data) * np.sin(omega*data)

        def ACF(data, gamma, omega):
            '''
            Ansatz of auto-correlation of real part
            '''
            return np.exp(-gamma*data) * np.cos(omega*data)

        tt = np.linspace(0, Lag*dt, num=Lag+1, endpoint=True) # time interval to plot the ACF or cross-correlation function
        
        acf_u = acf(np.real(u_t), nlags=Lag, fft=True) 
        ccf_u = -ccf(np.real(u_t), np.imag(u_t), adjusted=False)[:Lag+1]

        x0 = [0.5, 0.5]
        x1, _ = curve_fit(ACF, tt, acf_u, p0=x0, check_finite=True, maxfev=2000)
        x1_, _ = curve_fit(CCF, tt, ccf_u, p0=x0, check_finite=True, maxfev=2000)
        gamma_est = x1[0]
        omega_est_acf = x1[1]
        omega_est_ccf = x1_[1]
        omega_est_ca = np.abs(omega_est_acf) * np.sign(omega_est_ccf)

        m1 = np.mean(u_t)
        E1 = np.var(u_t)
        f_est = m1 * (gamma_est - 1j * omega_est_ca) 
        sigma_est = np.sqrt(2*E1*gamma_est)

        est_params = {
            'gamma': gamma_est,
            'omega_ccf': omega_est_ccf,
            'omega_acf': omega_est_acf,
            'omega': omega_est_ca,
            'f': f_est,
            'sigma': sigma_est,
        }
        return est_params

    def calibrate_v(self, u_t, dt, Lag):
        """
        calibration of real OU process
        - Lag: int, lag for computing the ACF
        """
        from scipy.optimize import curve_fit
        from statsmodels.tsa.stattools import acf

        def ACF(data, gamma):
            '''
            Ansatz of auto-correlation of real part
            '''
            return np.exp(-gamma*data)

        tt = np.linspace(0, Lag*dt, num=Lag+1, endpoint=True) # time interval to plot the ACF or cross-correlation function
        
        acf_u = acf(u_t, nlags=Lag, fft=True) 

        x0 = [0.5]
        x1, _ = curve_fit(ACF, tt, acf_u, p0=x0, check_finite=True, maxfev=2000)
        gamma_est = x1[0]

        m1 = np.mean(u_t)
        E1 = np.var(u_t)
        f_est = m1 * gamma_est 
        sigma_est = np.sqrt(2*E1*gamma_est)

        est_params = {
            'gamma': gamma_est,
            'f': f_est,
            'sigma': sigma_est,
        }
        return est_params


if __name__ == '__main__':
    np.random.seed(0)

    # Parameters
    T = 2e3
    dt = 0.005
    dt_obs = 0.25
    N = int(round(T / dt))
    N_gap = int(round(dt_obs / dt))
    sigma_obs = 0.5

    # Initialize states
    ur0 = 0
    ui0 = 0
    v0 = 0

    model = OUNonlinearModel()
    ur_truth, ui_truth, v_truth = model.forecast(N, dt, ur0, ui0, v0)

    # Generate observations
    ur_obs = ur_truth[::N_gap] + sigma_obs * np.random.randn(N // N_gap)
    ui_obs = ui_truth[::N_gap] + sigma_obs * np.random.randn(N // N_gap)
    v_obs = v_truth[::N_gap] + sigma_obs * np.random.randn(N // N_gap)

    # Save data
    np.savez('../data/data_ou_nonlinear.npz',
             ur_truth=ur_truth,
             ui_truth=ui_truth,
             v_truth=v_truth,
             ur_obs=ur_obs,
             ui_obs=ui_obs,
             v_obs=v_obs,
             N=N,
             N_gap=N_gap,
             dt=dt,
             dt_obs=dt_obs,
             sigma_obs=sigma_obs)