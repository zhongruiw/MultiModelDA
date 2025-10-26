import numpy as np

################################# Utils ###################################
class Polynomial_Detrend:
    def __init__(self, degree=1):
        """
        Fit polynomial trends along time for each spatial point.
        """
        self.degree = degree # degree of the polynomial. 1 = linear, 2 = quadratic, etc.
        self.coefs = None   # shape (Nx, degree+1)

    def fit(self, data):
        """
        data : np.ndarray, shape (T, Nx) Training data.
        """
        T, Nx = data.shape
        t = np.arange(T)  # time index
        self.coefs = np.empty((Nx, self.degree+1))
        for i in range(Nx):
            # polyfit handles NaNs poorly; skip or mask if needed
            y = data[:, i]
            mask = np.isfinite(y)
            if np.sum(mask) <= self.degree:
                self.coefs[i] = np.zeros(self.degree+1)
            else:
                self.coefs[i] = np.polyfit(t[mask], y[mask], deg=self.degree)
        return self
    
    def predict(self, t_points):
        """
        Predict trend on given time points (relative to training data).
        """
        t_points = np.asarray(t_points) # 1D array of time indices.
        Nx = self.coefs.shape[0]
        trend_pred = np.empty((len(t_points), Nx))
        for i in range(Nx):
            trend_pred[:, i] = np.polyval(self.coefs[i], t_points)
        return trend_pred

    def detrend(self, t_points, data):
        trend = self.predict(t_points)
        return data - trend

    def save(self, path: str):
        np.savez_compressed(path, degree=np.int64(self.degree), coefs=self.coefs)

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=False)
        degree = int(data["degree"])
        coefs = np.asarray(data["coefs"])
        obj = cls(degree=degree)
        obj.coefs = coefs
        return obj


############################ CNNLSTM Surrogate Model ###########################
import os
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class SeriesDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int, pred_len: int = 1):
        self.data = data.to(torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        if self.pred_len == 1:
            y = y.squeeze(0)
        return x, y


class PiecewiseSeriesDataset(Dataset):
    def __init__(self, segments: list[torch.Tensor], seq_len: int, pred_len: int):
        self.samples = []
        for seg in segments:
            seg = seg.to(torch.float32)
            if seg.shape[0] < seq_len + pred_len:
                continue
            for i in range(seg.shape[0] - seq_len - pred_len + 1):
                x = seg[i : i + seq_len]
                y = seg[i + seq_len : i + seq_len + pred_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if y.shape[0] == 1:
            y = y.squeeze(0)
        return x, y

# Model
class ConvBlock1D(nn.Module):
    """
    Length-preserving 1D conv block with residual.
    """
    def __init__(self, in_channels, out_channels, k=5, dilation=1):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(1, out_channels)
        self.skip = (in_channels == out_channels)
        if not self.skip:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.conv(x)
        y = self.act(y)
        y = self.norm(y)
        if self.skip:
            return x + y
        else:
            return self.proj(x) + y
            
class SE1D(nn.Module):
    """Squeeze-and-Excitation for 1D feature maps.
    Let the network learn which channels are important given the current feature map. 
    Computes a per-channel gate in [0,1] and rescales channels accordingly (channel attention). 
    Helpful when different physics dominate at different times/locations (e.g., wind-driven vs heat-flux-driven regimes).
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, 1)

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)           # (B, C, 1)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class Encoder1D(nn.Module):
    """
    Spatial encoder with pyramid downsampling to latent_dim.
    Input : (B, in_channels,  N_x)
    Output: (B, out_channels, latent_dim)
    """
    def __init__(self, in_channels, out_channels, hidden_channels, Nx, latent_dim, k=5, use_se=True):
        super().__init__()
        self.Nx = Nx
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.pre = ConvBlock1D(in_channels, hidden_channels, k, dilation=1)
        # progressive downsampling
        n_down = max(0, int(math.floor(math.log2(max(1, Nx // max(1, latent_dim))))))
        ch_in = hidden_channels
        downs = []
        for i in range(n_down):
            ch_out = min(out_channels, ch_in * 2)  # gradually increase channels
            downs += [
                ConvBlock1D(ch_in, ch_out, k, dilation=1),
                nn.Conv1d(ch_out, ch_out, kernel_size=4, stride=2, padding=1),  # length // 2
                nn.GroupNorm(1, ch_out),
                nn.GELU(),
            ]
            ch_in = ch_out
        self.down = nn.Sequential(*downs) if downs else nn.Identity()
        self.proj = nn.Conv1d(ch_in, out_channels, kernel_size=1) if ch_in != out_channels else nn.Identity()
        self.block = nn.Sequential(
            ConvBlock1D(out_channels, out_channels, k, dilation=1),
            ConvBlock1D(out_channels, out_channels, k, dilation=2),
            ConvBlock1D(out_channels, out_channels, k, dilation=4),
        )
        self.se = SE1D(out_channels) if use_se else nn.Identity()
        self.post = ConvBlock1D(out_channels, out_channels, k=3, dilation=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # (B, C, Nx)
        y = self.pre(x)            # (B, C_h, Nx)
        y = self.down(y)           # (B, C_h*2^k, ~Nx/2^k)
        y = self.proj(y)           # (B, C_o, ~)
        y = self.block(y)          # (B, C_o, ~)
        y = self.se(y)             # (B, C_o, ~)
        y = self.post(y)           # (B, C_o, ~)
        if y.shape[-1] != self.latent_dim:
            # y = F.interpolate(y, size=self.latent_dim, mode='linear', align_corners=False)
            y = F.adaptive_avg_pool1d(y, self.latent_dim)
        return y                    # (B, C_o, latent_dim)

class Decoder1D(nn.Module):
    """
    Mirrors Encoder1D: progressive upsampling back to Nx.
    Input:  (B, in_channels, latent_dim)
    Output: (B, out_channels, Nx)
    """
    def __init__(self, in_channels, out_channels, hidden_channels, Nx, latent_dim, k=5, use_se=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.Nx = Nx

        self.pre = ConvBlock1D(in_channels, in_channels, k=3, dilation=1)
        self.se = SE1D(in_channels) if use_se else nn.Identity()
        self.block = nn.Sequential(
            ConvBlock1D(in_channels, in_channels, k, dilation=1),
            ConvBlock1D(in_channels, in_channels, k, dilation=2),
            ConvBlock1D(in_channels, in_channels, k, dilation=4),
        )
        # progressive upsampling
        n_up = max(0, int(math.floor(math.log2(max(1, Nx // max(1, latent_dim))))))
        ch_out_proj = min(in_channels, hidden_channels * 2**n_up)
        self.proj = nn.Conv1d(in_channels, ch_out_proj, kernel_size=1) if ch_out_proj != in_channels else nn.Identity()
        ups = []
        ch_in = ch_out_proj
        for i in range(n_up):
            ch_out = max(hidden_channels, ch_in // 2)  # keep width stable or gently reduce
            ups += [
                nn.ConvTranspose1d(ch_in, ch_in, kernel_size=4, stride=2, padding=1),  # length * 2
                nn.GroupNorm(1, ch_in),
                nn.GELU(),
                ConvBlock1D(ch_in, ch_out, k=5, dilation=1),
            ]
            ch_in = ch_out
        self.up = nn.Sequential(*ups) if ups else nn.Identity()
        self.post = ConvBlock1D(hidden_channels, out_channels, k=3, dilation=1)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h):  # (B, C_in, latent_dim)
        y = self.pre(h)                # (B, C_in, latent_dim)
        y = self.se(y)                 # (B, C_in, latent_dim)
        y = self.block(y)              # (B, C_in, latent_dim)
        y = self.proj(y)               # (B, C_h*2^k, latent_dim)
        y = self.up(y)                 # (B, C_h, ~latent_dim * 2^k)
        y = self.post(y)               # (B, C_out, ~latent_dim * 2^k)
        if y.shape[-1] != self.Nx:
            y = F.interpolate(y, size=self.Nx, mode='linear', align_corners=False) # (B, C_out, Nx)
        return y

class PositionWiseLSTM(nn.Module):
    """
    Run the same LSTM across time for each latent position independently.
    Input:  z_t per time step encoded to (B, L, C_lat, D_lat)
    Output: (B, C_lat, D_lat)   # last hidden per position, projected back to C_lat
    """
    def __init__(self, in_channels, hidden_channels=128, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        H = hidden_channels
        self.proj = nn.Linear(H, in_channels)

    def forward(self, z):  # (B, L, C, D)
        B, L, C, D = z.shape
        z_ = z.permute(0, 3, 1, 2).contiguous()          # (B, D, L, C)
        z_ = z_.view(B * D, L, C)                        # (B*D, L, C)
        o, _ = self.lstm(z_)                             # (B*D, L, H)
        h_last = o[:, -1, :]                             # (B*D, H)
        h_last = self.proj(h_last)                       # (B*D, C)
        h = h_last.view(B, D, -1).permute(0, 2, 1)       # (B, C, D)
        return h
        
class CNNLSTM1D(nn.Module):
    """
    Spatial encoder -> position-wise temporal LSTM -> upsampling decoder.
    """
    def __init__(self, in_channels, out_channels, latent_channels, hidden_channels, Nx, latent_dim=16, lstm_hidden_dim=128, lstm_layers=1, dropout=0.1, use_se=True):
        super().__init__()
        self.Nx = Nx
        self.out_channels = out_channels
        self.encoder = Encoder1D(in_channels, latent_channels, hidden_channels, Nx, latent_dim, k=5, use_se=use_se)
        self.temporal = PositionWiseLSTM(latent_channels, lstm_hidden_dim, lstm_layers, dropout)
        self.decoder = Decoder1D(latent_channels, out_channels, hidden_channels, Nx, latent_dim, k=5, use_se=use_se)

    def forward(self, x):  # x: (B, L, C, Nx)
        B, L, C, Nx = x.shape
        x_ = x.reshape(B * L, C, Nx)                      # (B*L, C, D)
        z = self.encoder(x_)                              # (B*L, C_lat, D_lat)
        z = z.view(B, L, z.shape[1], z.shape[2])          # (B, L, C_lat, D_lat)
        h = self.temporal(z)                              # (B, C_lat, D_lat)
        y = self.decoder(h)                               # (B, C_out, Nx)
        return y

class ChannelZScoreScaler:
    """
    Per-channel z-score across time×space.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = None  # (1, C, 1) float32
        self.std  = None  # (1, C, 1) float32

    def fit(self, arr: np.ndarray): # arr shape (T, C, Nx)
        mean = arr.mean(axis=(0, 2), keepdims=True)       # (1, C, 1)
        std  = arr.std(axis=(0, 2), keepdims=True) + self.eps
        self.mean = mean.astype(np.float32)
        self.std  = std.astype(np.float32)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def transform_segments(self, segments):
        out = []
        for seg in segments:
            out.append((seg - self.mean) / self.std)
        return out

    def inverse(self, arr_norm: np.ndarray) -> np.ndarray:
        return arr_norm * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.std  = state["std"]
        self.eps  = state.get("eps", 1e-8)
        
class CNNLSTMTrainer:
    def __init__(self, data=None, piecewise=False, data_segments=None, seed=0, device=None, seq_len=1, pred_len=1, 
                 in_channels=5, out_channels=5, latent_channels=40, hidden_channels=10, latent_dim=20, lstm_hidden_dim=64, lstm_layers=1, 
                 batch_size=20, num_epochs=20, lr=1e-3, weight_decay=1e-4, grad_clip=None, 
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, val_every=1, print_params=True,
                ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.latent_channels, self.latent_dim = latent_channels, latent_dim
        self.lstm_hidden_dim, self.lstm_layers = lstm_hidden_dim, lstm_layers
        self.batch_size, self.num_epochs = batch_size, num_epochs
        self.grad_clip, self.val_every = grad_clip, val_every
        self.scaler = ChannelZScoreScaler()
        self._load_data(data, piecewise, data_segments, train_ratio, val_ratio, test_ratio)
        Nx = data.shape[2] if data is not None else self.train_dataset[0][0].shape[-1]
        self.model = CNNLSTM1D(in_channels=in_channels, out_channels=out_channels, latent_channels=latent_channels, 
                                  hidden_channels=hidden_channels, Nx=Nx, latent_dim=latent_dim,
                                  lstm_hidden_dim=lstm_hidden_dim, lstm_layers=lstm_layers).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs+1)
        self.loss_fn = nn.MSELoss()
        if print_params:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            enc = sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)
            tmp = sum(p.numel() for p in self.model.temporal.parameters() if p.requires_grad)
            dec = sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)
            print(f"Encoder #parameters: {enc:,}")
            print(f"LSTM    #parameters: {tmp:,}")
            print(f"Decoder #parameters: {dec:,}")
            print(f"TOTAL   #parameters: {total:,} (trainable {trainable:,})")

    def _load_data(self, data=None, piecewise=False, data_segments=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        if piecewise:
            total_size = len(data_segments)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size
            print(f'train size: {train_size}, val size: {val_size}, test size: {test_size}')
            train_cat = np.concatenate(data_segments[:train_size], axis=0)  # (T_tr, C, Nx)
            self.scaler.fit(train_cat)
            train_norm = self.scaler.transform_segments(data_segments[:train_size])
            val_norm   = self.scaler.transform_segments(data_segments[train_size:train_size+val_size])
            test_norm  = self.scaler.transform_segments(data_segments[train_size+val_size:])
            self.train_dataset = PiecewiseSeriesDataset(train_norm, self.seq_len, self.pred_len)
            self.val_dataset   = PiecewiseSeriesDataset(val_norm,   self.seq_len, self.pred_len)
            self.test_dataset  = PiecewiseSeriesDataset(test_norm,  self.seq_len, self.pred_len)

        else:
            dataset = SeriesDataset(data, self.seq_len, self.pred_len) # data shape (T, C, Nx)
            total_size = len(dataset)
            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size
            print(f'train size: {train_size}, val size: {val_size}, test size: {test_size}')
            train_raw = data[:train_size]
            self.scaler.fit(train_raw)
            data_norm = self.scaler.transform(data)
            dataset = SeriesDataset(data_norm, self.seq_len, self.pred_len) # data shape (T, C, Nx)
            self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
            self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size+val_size))
            self.test_dataset = torch.utils.data.Subset(dataset, range(train_size+val_size, total_size))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        preds = []
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.model(xb)
            loss = self.loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)
            preds.append(pred)
        avg_loss = total_loss / len(loader.dataset)
        pred = torch.cat(preds, dim=0)
        return avg_loss, pred

    def train(self, plot=True, save_dir_model=None, save_dir_loss=None):
        best_val = float("inf")
        start_time_total = time.time()
        train_losses = []
        val_losses = []
        for epoch in range(0, self.num_epochs + 1):
            start_time_ep = time.time()
            train_loss = self.train_one_epoch(self.train_loader)
            train_losses.append(train_loss)
            self.scheduler.step()
            if epoch % self.val_every == 0:
                val_loss, _ = self.evaluate(self.val_loader)
                val_losses.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'scaler_state': self.scaler.state_dict(),
                    }
                    if save_dir_model is not None:
                        torch.save(checkpoint, f"{save_dir_model}")
                    status = "✅"
                else:
                    status = ""
                time_used_ep = time.time() - start_time_ep
                print(f"Epoch {epoch}: Train {train_loss:.6f} | Val {val_loss: .6f} | Time {time_used_ep / 60:.4f} mins {status}")
            else:
                val_losses.append(np.nan)
                time_used_ep = time.time() - start_time_ep
                print(f"Epoch {epoch}: Train {train_loss:.6f} | Time {time_used_ep/60:.2f} mins")
        time_used_total = time.time() - start_time_total
        print(f"total time used: {time_used_total / 3600: .2f} hrs")
        
        if save_dir_loss is not None:
            np.savez(f"{save_dir_loss}",
            train=np.asarray(train_losses),
            val=np.asarray(val_losses),
            )
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4, 3))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("Training Loss Curve")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def test(self, pretrain=False, pretrain_dir=None, return_physical=False, nondimensionalize=False, scales=None):
        if pretrain:
            checkpoint = torch.load(f"{pretrain_dir}", map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        test_loss, test_pred = self.evaluate(self.test_loader)
        print(f"Test Loss (normalized space) = {test_loss:.6f}")

        if return_physical:
            pred_phy = self.scaler.inverse(test_pred.detach().cpu().numpy())
            if nondimensionalize:
                return pred_phy / scales[None, :] # (T, C)
            else:
                return pred_phy
        else:
            return test_pred.to("cpu")
