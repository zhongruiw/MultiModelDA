import numpy as np


class TopoBaroModel:
    def __init__(self, K=10, lx=1.0, beta=1.0,
                 d_U=0.0125, d_v=0.0125, d_T=0.1,
                 kappa_T=0.001, alpha=1.0,
                 sigma_U=1/np.sqrt(2), sigma_v=1/(20*np.sqrt(2)),
                 H1=1.0, H2=0.5, theta0=-np.pi / 4):
        """
        Initialize the spectral barotropic model with topography.
        """
        self.K = K
        self.Kx = np.fft.fftfreq(2*K+1) * (2*K+1)
        self.lx = lx
        self.beta = beta
        self.d_U = d_U
        self.d_v = d_v
        self.d_T = d_T
        self.kappa_T = kappa_T
        self.alpha = alpha
        self.sigma_U = sigma_U
        self.sigma_vk = np.zeros(2*K+1, dtype=complex)
        self.sigma_vk[1:] = -1j / self.Kx[1:] * sigma_v  # avoid div-by-zero at k=0
        self.sigma_vk[0] = 0.0
        self.gamma_vk = np.zeros_like(self.Kx)
        self.gamma_Tk = d_T + kappa_T * self.Kx**2
        self.H1 = H1
        self.H2 = H2
        self.theta0 = theta0
        self.h_hat = self._generate_topography()

    def _generate_topography(self):
        h_hat = np.zeros(self.K*2+1, dtype=complex) # wavenumber order (0,1,...,K,-K,...,-1)
        h_hat[0] = 0 # constant mode set to zero
        h_hat[1] = 0.5 * self.H1 * (1 - 1j)  # k=1
        h_hat[2] = 0.5 * self.H2 * (1 - 1j)  # k=2
        for k in range(3, self.K+1):
            h_hat[k] = k ** (-2) * np.exp(1j * self.theta0) / np.sqrt(2) # (normalization)
        h_hat[self.K+1:] = np.conj(h_hat[1:self.K+1][::-1])
        return h_hat

    # def forecast(self, N, dt, U0, v_hat0, T_hat0):
    #     """
    #     Forecast a single trajectory using Euler-Maruyama.
    #     Parameters:
    #     - N: total number of simulation steps
    #     - dt: time step
    #     - U0: scalar initial zonal flow
    #     - v_hat0, T_hat0: complex arrays of shape (2K+1,)
    #     - N_gap: save data every N_gap steps1)
    #     Returns:
    #     - U, v_hat, T_hat: downsampled arrays (length ≈ N//N_gap)
    #     """
    #     K, Kx = self.K, self.Kx
    #     h_hat, d_U, sigma_U, lx, d_v = self.h_hat, self.d_U, self.sigma_U, self.lx, self.d_v
    #     gamma_vk, gamma_Tk, sigma_vk = self.gamma_vk, self.gamma_Tk, self.sigma_vk
    #     alpha, beta, d_T = self.alpha, self.beta, self.d_T

    #     U = np.zeros(N)
    #     v_hat = np.zeros((N, 2*K+1), dtype=complex)
    #     T_hat = np.zeros((N, 2*K+1), dtype=complex)
    #     U[0] = U0
    #     v_hat[0] = v_hat0
    #     T_hat[0] = T_hat0

    #     save_idx = 1
    #     for n in range(1, N):
    #         omega_vk = lx * (1.0 / Kx * beta - Kx * U[n-1])
    #         omega_vk[0] = 0.0  # avoid div-by-zero
    #         omega_Tk = -Kx * U[n-1]

    #         # U update
    #         U[n] = U[n-1] + (np.sum(np.conj(h_hat) * v_hat[n-1]).real - d_U * U[n-1]) * dt \
    #                + sigma_U * np.sqrt(dt) * np.random.randn()

    #         # v_hat update for k ≠ 0
    #         noise = sigma_vk[1:K+1] / np.sqrt(2) * np.sqrt(dt) * (np.random.randn(K) + 1j * np.random.randn(K))
    #         v_hat[n, 1:K+1] = v_hat[n-1, 1:K+1] \
    #             + ((-gamma_vk[1:K+1] + 1j * omega_vk[1:K+1]) * v_hat[n-1, 1:K+1]
    #                - lx ** 2 * h_hat[1:K+1] * U[n-1]
    #                - d_v * v_hat[n-1, 1:K+1]) * dt + noise

    #         # T_hat update for k ≠ 0
    #         T_hat[n, 1:K+1] = T_hat[n-1, 1:K+1] \
    #             + ((-gamma_Tk[1:K+1] + 1j * omega_Tk[1:K+1]) * T_hat[n-1, 1:K+1]
    #                - d_T * T_hat[n-1, 1:K+1] - alpha * v_hat[n-1, 1:K+1]) * dt

    #         # symmetry: v(-k) = conj(v(k))
    #         v_hat[n, K+1:] = np.conj(v_hat[n, 1:K+1][::-1])
    #         T_hat[n, K+1:] = np.conj(T_hat[n, 1:K+1][::-1])

    #     return U, v_hat, T_hat

    # def forecast(self, N, dt, U0, v_hat0, T_hat0, N_gap=10):
    #     """
    #     Forecast a single trajectory using Euler-Maruyama.
        
    #     Parameters:
    #     - N: total number of simulation steps
    #     - dt: time step
    #     - U0: scalar initial zonal flow
    #     - v_hat0, T_hat0: complex arrays of shape (2K+1,)
    #     - N_gap: save data every N_gap steps
        
    #     Returns:
    #     - U, v_hat, T_hat: downsampled arrays (length ≈ N//N_gap)
    #     """
    #     K, Kx = self.K, self.Kx
    #     h_hat, d_U, sigma_U, lx, d_v = self.h_hat, self.d_U, self.sigma_U, self.lx, self.d_v
    #     gamma_vk, gamma_Tk, sigma_vk = self.gamma_vk, self.gamma_Tk, self.sigma_vk
    #     alpha, beta, d_T = self.alpha, self.beta, self.d_T

    #     # Preallocate full internal state
    #     u_n = U0
    #     v_n = v_hat0.copy()
    #     t_n = T_hat0.copy()

    #     num_save = int(N // N_gap)
    #     U = np.zeros(num_save)
    #     v_hat = np.zeros((num_save, 2 * K + 1), dtype=complex)
    #     T_hat = np.zeros((num_save, 2 * K + 1), dtype=complex)

    #     # Save initial state
    #     U[0] = u_n
    #     v_hat[0] = v_n
    #     T_hat[0] = t_n

    #     save_idx = 1
    #     for n in range(1, N):
    #         omega_vk = lx * (1.0 / Kx * beta - Kx * u_n)
    #         omega_vk[0] = 0.0  # avoid div-by-zero
    #         omega_Tk = -Kx * u_n

    #         # U update
    #         u_n = u_n + (np.sum(np.conj(h_hat) * v_n).real - d_U * u_n) * dt \
    #               + sigma_U * np.sqrt(dt) * np.random.randn()

    #         # v_hat update for k ≠ 0
    #         noise = sigma_vk[1:K + 1] / np.sqrt(2) * np.sqrt(dt) * (np.random.randn(K) + 1j * np.random.randn(K))
    #         v_new = v_n.copy()
    #         v_new[1:K + 1] = v_n[1:K + 1] + ((-gamma_vk[1:K + 1] + 1j * omega_vk[1:K + 1]) * v_n[1:K + 1]
    #                                         - lx ** 2 * h_hat[1:K + 1] * u_n
    #                                         - d_v * v_n[1:K + 1]) * dt + noise
    #         v_new[K + 1:] = np.conj(v_new[1:K + 1][::-1])

    #         # T_hat update for k ≠ 0
    #         T_new = t_n.copy()
    #         T_new[1:K + 1] = t_n[1:K + 1] + ((-gamma_Tk[1:K + 1] + 1j * omega_Tk[1:K + 1]) * t_n[1:K + 1]
    #                                         - d_T * t_n[1:K + 1] - alpha * v_n[1:K + 1]) * dt
    #         T_new[K + 1:] = np.conj(T_new[1:K + 1][::-1])

    #         # Update state
    #         v_n = v_new
    #         t_n = T_new

    #         if n % N_gap == 0:
    #             U[save_idx] = u_n
    #             v_hat[save_idx] = v_n
    #             T_hat[save_idx] = t_n
    #             save_idx += 1

    #     return U, v_hat, T_hat


    def forecast(self, N, dt, U0, v_hat0, T_hat0, N_gap=10):
        """
        Forecast using RK4 for deterministic terms + additive noise (Euler–Maruyama).
        """
        K, Kx = self.K, self.Kx
        h_hat, d_U, lx, d_v = self.h_hat, self.d_U, self.lx, self.d_v
        gamma_vk, gamma_Tk, sigma_vk = self.gamma_vk, self.gamma_Tk, self.sigma_vk
        alpha, beta, d_T, sigma_U = self.alpha, self.beta, self.d_T, self.sigma_U

        def rhs(U, v_hat, T_hat):
            omega_vk = lx * (1.0 / Kx * beta - Kx * U)
            omega_vk[0] = 0.0
            omega_Tk = -Kx * U

            dv = np.zeros_like(v_hat)
            dT = np.zeros_like(T_hat)

            dv[1:K + 1] = (-gamma_vk[1:K + 1] + 1j * omega_vk[1:K + 1]) * v_hat[1:K + 1] \
                          - lx ** 2 * h_hat[1:K + 1] * U - d_v * v_hat[1:K + 1]

            dT[1:K + 1] = (-gamma_Tk[1:K + 1] + 1j * omega_Tk[1:K + 1]) * T_hat[1:K + 1] \
                          - d_T * T_hat[1:K + 1] - alpha * v_hat[1:K + 1]

            dv[K + 1:] = np.conj(dv[1:K + 1][::-1])
            dT[K + 1:] = np.conj(dT[1:K + 1][::-1])

            dU = np.sum(np.conj(h_hat) * v_hat).real - d_U * U
            return dU, dv, dT

        # Initialize state
        u_n = U0
        v_n = v_hat0.copy()
        t_n = T_hat0.copy()

        num_save = N // N_gap 
        U = np.zeros(num_save)
        v_hat = np.zeros((num_save, 2 * K + 1), dtype=complex)
        T_hat = np.zeros((num_save, 2 * K + 1), dtype=complex)

        U[0] = u_n
        v_hat[0] = v_n
        T_hat[0] = t_n

        save_idx = 1
        for n in range(1, N):
            # RK4 steps for deterministic part
            k1_u, k1_v, k1_t = rhs(u_n, v_n, t_n)
            k2_u, k2_v, k2_t = rhs(u_n + 0.5 * dt * k1_u,
                                   v_n + 0.5 * dt * k1_v,
                                   t_n + 0.5 * dt * k1_t)
            k3_u, k3_v, k3_t = rhs(u_n + 0.5 * dt * k2_u,
                                   v_n + 0.5 * dt * k2_v,
                                   t_n + 0.5 * dt * k2_t)
            k4_u, k4_v, k4_t = rhs(u_n + dt * k3_u,
                                   v_n + dt * k3_v,
                                   t_n + dt * k3_t)

            # Deterministic update
            u_n += dt / 6 * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
            v_n += dt / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            t_n += dt / 6 * (k1_t + 2 * k2_t + 2 * k3_t + k4_t)

            # Additive noise update (Euler–Maruyama)
            u_n += sigma_U * np.sqrt(dt) * np.random.randn()
            noise = sigma_vk[1:K + 1] / np.sqrt(2) * np.sqrt(dt) * (np.random.randn(K) + 1j * np.random.randn(K))
            v_n[1:K + 1] += noise
            v_n[K + 1:] = np.conj(v_n[1:K + 1][::-1])

            if n % N_gap == 0:
                U[save_idx] = u_n
                v_hat[save_idx] = v_n
                T_hat[save_idx] = t_n
                save_idx += 1

        return U, v_hat, T_hat

    def ensemble_forecast(self, N, dt, U0, v_hat0, T_hat0, ensemble_size, N_gap=10):
        """
        Ensemble forecast. Initial states must be arrays of shape (ensemble_size,)
        or (ensemble_size, 2K+1) for v_hat0, T_hat0.
        """
        num_save = N // N_gap 
        U_ens = np.zeros((ensemble_size, num_save))
        v_hat_ens = np.zeros((ensemble_size, num_save, self.K*2+1), dtype=complex)
        T_hat_ens = np.zeros((ensemble_size, num_save, self.K*2+1), dtype=complex)

        for i in range(ensemble_size):
        # for i in prange(ensemble_size):
            U_path, v_hat_path, T_hat_path = self.forecast(N, dt, U0[i], v_hat0[i], T_hat0[i], N_gap)
            U_ens[i] = U_path
            v_hat_ens[i] = v_hat_path
            T_hat_ens[i] = T_hat_path

        return U_ens, v_hat_ens, T_hat_ens


if __name__ == '__main__':
    np.random.seed(0)

    # Parameters
    T = 2e3
    dt = 5e-5
    dt_obs = 0.1
    N = int(round(T / dt))
    N_gap = int(round(dt_obs / dt))
    sigma_obs = 0.5
    K = 10

    # Initialize states
    U0 = .1 * np.random.randn()
    v_hat0 = 0.1 / np.sqrt(2) * (np.random.randn(2*K+1) + 1j * np.random.randn(2*K+1))
    T_hat0 = 0.1 / np.sqrt(2) * (np.random.randn(2*K+1) + 1j * np.random.randn(2*K+1))
    v_hat0[0] = 0
    T_hat0[0] = 0

    model = TopoBaroModel(K=K)
    U_truth, v_hat_truth, T_hat_truth = model.forecast(N, dt,  U0, v_hat0, T_hat0, N_gap=20)

    # Generate observations
    U_obs = U_truth[::N_gap] + sigma_obs * np.random.randn(N // N_gap)
    # v_obs = v_truth[::N_gap] + sigma_obs * np.random.randn(N // N_gap)
    # T_obs = T_truth[::N_gap] + sigma_obs * np.random.randn(N // N_gap)

    # Save data
    np.savez('../data/data_topobaro.npz',
             U_truth=U_truth,
             v_hat_truth=v_hat_truth,
             T_hat_truth=T_hat_truth,
             U_obs=U_obs,
             # v_obs=v_obs,
             # T_obs=T_obs,
             N=N,
             N_gap=N_gap,
             dt=dt,
             dt_obs=dt_obs,
             sigma_obs=sigma_obs)