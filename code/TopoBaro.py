import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os


############################################## TopoBaro Model for generating truth #############################################    
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
            h_hat[k] = k ** (-2) * np.exp(1j * self.theta0) #/ np.sqrt(2) # (normalization)
        h_hat[self.K+1:] = np.conj(h_hat[1:self.K+1][::-1])
        return h_hat

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

        num_save = N // N_gap + 1
        U = np.zeros(num_save)
        v_hat = np.zeros((num_save, 2 * K + 1), dtype=complex)
        T_hat = np.zeros((num_save, 2 * K + 1), dtype=complex)

        U[0] = u_n
        v_hat[0] = v_n
        T_hat[0] = t_n

        save_idx = 1
        for n in range(1, N+1):
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
        num_save = N // N_gap + 1
        U_ens = np.zeros((ensemble_size, num_save))
        v_hat_ens = np.zeros((ensemble_size, num_save, self.K*2+1), dtype=complex)
        T_hat_ens = np.zeros((ensemble_size, num_save, self.K*2+1), dtype=complex)

        for i in range(ensemble_size):
            U_path, v_hat_path, T_hat_path = self.forecast(N, dt, U0[i], v_hat0[i], T_hat0[i], N_gap)
            U_ens[i] = U_path
            v_hat_ens[i] = v_hat_path
            T_hat_ens[i] = T_hat_path

        return U_ens, v_hat_ens, T_hat_ens

    def ifft2phy(self, X, v_hat, T_hat):
        K, Kx = self.K, self.Kx
        v = (v_hat @ np.exp(1j * Kx[:, None] * X)).real
        T = (T_hat @ np.exp(1j * Kx[:, None] * X)).real

        return v, T


############################################# LSTM Surrogate Model #############################################    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

class BarotropicDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y[0], dtype=torch.float32)

class PiecewiseSeriesDataset(Dataset):
    def __init__(self, segments, seq_len, pred_len):
        self.samples = []
        for seg in segments:
            if len(seg) < seq_len + pred_len:
                continue
            for i in range(len(seg) - seq_len - pred_len + 1):
                x = seg[i:i + seq_len]
                y = seg[i + seq_len:i + seq_len + pred_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y[0], dtype=torch.float32)

class LSTMTrainer:
    def __init__(self, data_path=None, piecewise=False, data_segments=None, seq_len=10, pred_len=1,
                 hidden_dim=64, batch_size=200, num_epochs=20,
                 model_dir="../model", seed=0, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_dir = model_dir
        self._load_data(data_path, piecewise, data_segments)
        self.model = LSTMModel(self.input_dim, hidden_dim, self.input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.train_losses = []

    def _load_data(self, path=None, piecewise=False, data_segments=None):
        if piecewise:
            total_size = len(data_segments)
            train_size = int(0.8 * total_size)
            test_size = total_size - train_size
            print(f'train size: {train_size}, test size: {test_size}')
            self.train_dataset = PiecewiseSeriesDataset(data_segments[:train_size], self.seq_len, self.pred_len)
            self.test_dataset = PiecewiseSeriesDataset(data_segments[train_size:], self.seq_len, self.pred_len)
            self.input_dim = data_segments[0].shape[1]
        else:
            data = np.load(path)
            U_truth = data['U_truth']
            v_truth = data['v_truth']
            T_truth = data['T_truth']
            data_all = np.concatenate([U_truth[:, None], v_truth, T_truth], axis=1).astype(np.float32)
            dataset = BarotropicDataset(data_all, self.seq_len, self.pred_len)
            total_size = len(dataset)
            train_size = int(0.8 * total_size)
            test_size = total_size - train_size
            print(f'train size: {train_size}, test size: {test_size}')
            self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
            self.test_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))
            self.input_dim = data_all.shape[1]

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, plot=True, save_dir_model=None, save_dir_loss=None):
        os.makedirs(self.model_dir, exist_ok=True)
        start_time = time.time()
        for epoch in range(0, self.num_epochs + 1):
            self.model.train()
            total_loss = 0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(self.train_dataset)
            self.train_losses.append(avg_loss)
            if epoch % 10 == 0:
                time_used = time.time() - start_time
                print(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}, Time = {time_used / 60:.4f} mins")
                start_time = time.time()

        if save_dir_loss is not None:
            np.save(f"{save_dir_loss}", np.array(self.train_losses))
        if save_dir_model is not None:
            torch.save(self.model.state_dict(), f"{save_dir_model}")    
        if plot:
            plt.figure(figsize=(4, 3))
            plt.plot(self.train_losses, label='Train Loss')
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("Training Loss Curve")
            plt.grid(True)
            plt.legend()
            plt.show()

    def test(self, pretrain=False, pretrain_dir=None):
        if pretrain:
            self.model.load_state_dict(torch.load(f"{pretrain_dir}"))
        self.model.eval()
        test_loss = 0
        test_preds = []
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                test_loss += loss.item() * xb.size(0)
                test_preds.append(pred)

        test_pred = torch.cat(test_preds, dim=0)
        test_loss /= len(self.test_dataset)
        print(f"Test Loss = {test_loss:.6f}")

        return test_pred.to("cpu")

class AutoRegressiveLSTMModel:
    def __init__(self, regime_models=[1], routing_matrix=None, holding_parameters=None, device=None):
        """
        Initialize autoregressive LSTM model with CTMC-based regime switching.
        
        Args:
            regime_models (list of nn.Module): List of LSTM models, one per regime.
            routing_matrix (np.ndarray): CTMC routing_matrix matrix of shape (n_regimes, n_regimes).
            holding_parameters (list or np.ndarray): Holding rates (lambda) for each regime.
            device (str or torch.device): Device to run the simulation on.
        """
        self.regime_models = regime_models
        self.routing_matrix = routing_matrix
        self.holding_parameters = holding_parameters
        self.n_regimes = len(regime_models)
        self.device = device

    def forecast(self, N, dt, x0, S0):
        """
        Simulate a single trajectory with autoregressive regime-switching LSTM.
        
        Args:
            N (int): Number of forecast steps.
            dt (float): Time step size.
            x0 (torch.Tensor): Initial input of shape (seq_len, input_dim).
            S0 (int): Initial regime index.
        
        Returns:
            x (np.ndarray): Forecasted trajectory, shape (N+1, input_dim).
            S (np.ndarray): Regime path, shape (N+1,)
        """
        input_dim = x0.shape[-1]
        x0 = x0.to(self.device)
        x0 = x0.unsqueeze(0) # (1, seq_len, input_dim)
        x = torch.zeros((1, N+1, input_dim), device=self.device)
        S = np.zeros(N+1, dtype=int)
        x[:, 0], S[0] = x0[:, -1], S0
        preds = []

        with torch.no_grad():
            for n in range(1, N+1):
                current_regime = S[n - 1]
                holding_param = self.holding_parameters[current_regime]

                # Regime switching
                if np.random.rand() < holding_param * dt:
                    S[n] = np.random.choice(self.n_regimes, p=self.routing_matrix[current_regime])
                else:
                    S[n] = current_regime

                model = self.regime_models[current_regime]
                model.eval()
                x[:, n] = model(x0)
                x0 = torch.cat((x0[:, 1:], x[:, n].unsqueeze(1)), dim=1)

        return x[0].cpu().numpy(), S

    def ensemble_forecast(self, N, dt, x0, S0, ensemble_size):
        """
        Simulate an ensemble of regime-switching forecasts.
        
        Args:
            N (int): Number of forecast steps.
            dt (float): Time step size.
            x0 (torch.Tensor): Initial inputs, shape (ensemble_size, seq_len, input_dim).
            S0 (array-like): Initial regimes, shape (ensemble_size,)
            ensemble_size (int): Number of ensemble members.
        
        Returns:
            X (np.ndarray): Forecast trajectories, shape (ensemble_size, N+1, input_dim)
            S (np.ndarray): Regime paths, shape (ensemble_size, N+1)
        """
        input_dim = x0.shape[-1]
        X = np.zeros((ensemble_size, N+1, input_dim))
        S = np.zeros((ensemble_size, N+1), dtype=int)
        X[:, 0, :] = x0[:, -1, :].cpu().numpy()
        S[:, 0] = S0
        x0 = x0.to(self.device)

        for i in range(ensemble_size):
            xi, Si = self.forecast(N, dt, x0[i], S0[i])
            X[i], S[i] = xi, Si

        return X, S


if __name__ == '__main__':
    np.random.seed(0)

    # Parameters
    T = 2e3
    dt = 1e-3
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

    # U0 = np.tile(.1 * np.random.randn(1), Ns)
    # v_hat0 = 0.1 / np.sqrt(2) * np.tile((np.random.randn(2*K+1) + 1j * np.random.randn(2*K+1))[None, :], (Ns, 1))
    # T_hat0 = 0.1 / np.sqrt(2) * np.tile((np.random.randn(2*K+1) + 1j * np.random.randn(2*K+1))[None, :], (Ns, 1))
    # v_hat0[:,0] = 0
    # T_hat0[:,0] = 0

    model = TopoBaroModel(K=K)
    U_truth, v_hat_truth, T_hat_truth = model.forecast(N, dt,  U0, v_hat0, T_hat0, N_gap)

    # get physical variables
    X = np.linspace(0, np.pi*2, 2*K+1, endpoint=False)
    v_truth, T_truth = model.ifft2phy(X, v_hat_truth, T_hat_truth)

    # Generate observations
    U_obs = U_truth + sigma_obs * np.random.randn(N // N_gap + 1)
    v_obs = v_truth + sigma_obs * np.random.randn(N // N_gap + 1, 2*K+1)
    T_obs = T_truth + sigma_obs * np.random.randn(N // N_gap + 1, 2*K+1)

    # Save data
    np.savez('../data/data_topobaro.npz',
             U_truth=U_truth,
             v_hat_truth=v_hat_truth,
             T_hat_truth=T_hat_truth,
             v_truth = v_truth,
             T_truth = T_truth,
             U_obs=U_obs,
             v_obs=v_obs,
             T_obs=T_obs,
             N=N,
             N_gap=N_gap,
             dt=dt,
             dt_obs=dt_obs,
             sigma_obs=sigma_obs)