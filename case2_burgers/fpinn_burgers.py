"""
Fractional PINNs (fPINNs) for a coupled 2D time-fractional PDE system (u, v)
on Ω=[0,1]×[0,1], t∈[0,1].
"""

import os
from dataclasses import dataclass
from collections import OrderedDict
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from scipy.special import gamma

# -----------------------------------------------------------------------------
# Device & seeds
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TORCH_SEED = 66
NP_SEED = 66
torch.manual_seed(TORCH_SEED)
np.random.seed(NP_SEED)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # Network
    layers: Tuple[int, ...] = (3, 60, 60, 60, 2)   # identical

    # Training grid
    M_train: int = 128                             # spatial resolution
    N_time_train: int = 10
    t_end_train: float = 1.0                       # t ∈ [0,1]

    # Fractional derivative settings
    alpha: float = 0.5
    R: float = 0.01

    # Optimizer & scheduler
    n_iters_adam: int = 2000
    lr_adam: float = 1e-3
    step_size: int = 100
    gamma_lr: float = 0.97

    # Paths
    pre_model_save_path: str = "./model/checkpoint50.pt"
    model_save_path: str = "./model/checkpoint100.pt"
    fig_save_path: str = "./figures/"

    # Inference grid
    M_infer: int = 128
    NN_infer: int = 16
    dt_infer: float = 0.1

    # Exact solution files
    u_exact_path: str = "./model/u_e.npy"


CFG = Config()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def save_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: StepLR,
                    save_path: str) -> None:
    """Save model & optimizer states."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        save_path,
    )


def load_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer = None,
                    scheduler: StepLR = None,
                    save_path: str = ""):
    """Load model states."""
    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print("Pretrained model loaded!")
    return model, optimizer, scheduler


def frobenius_norm(tensor: np.ndarray) -> float:
    """Frobenius norm"""
    return np.sqrt(np.sum(tensor ** 2))


def make_spacetime_grid(M: int, time_nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        x_train, y_train, t_train: each of shape (N*T, 1),
        where spatial index i is the outer loop and time index j is the inner loop.
    """
    x = np.linspace(0.0, 1.0, M)
    y = np.linspace(0.0, 1.0, M)
    X, Y = np.meshgrid(x, y)

    x_space = X.reshape(x.shape[0] ** 2, 1)
    y_space = Y.reshape(y.shape[0] ** 2, 1)

    t_train = np.expand_dims(time_nodes, -1)  # (T,1)

    N = x_space.shape[0]  # M^2
    T = t_train.shape[0]

    XX = np.tile(x_space[:, 0:1], (1, T))      # (N,T)
    YY = np.tile(y_space[:, 0:1], (1, T))      # (N,T)
    TT = np.tile(t_train, (1, N)).T            # (N,T)

    x_out = XX.flatten()[:, None]              # (N*T,1)
    y_out = YY.flatten()[:, None]              # (N*T,1)
    t_out = TT.flatten()[:, None]              # (N*T,1)

    return x_out, y_out, t_out

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
class DNN(nn.Module):
    """Fully-connected network with Tanh activations."""

    def __init__(self, layers: Tuple[int, ...]):
        super().__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh

        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append((f"layer_{i}", nn.Linear(layers[i], layers[i + 1])))
            layer_list.append((f"activation_{i}", self.activation()))

        layer_list.append((f"layer_{self.depth - 1}", nn.Linear(layers[-2], layers[-1])))

        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PINNWrapper(nn.Module):
    """
    Enforces homogeneous initial condition using a trial function:
        u(x,y,t) = B(t)*u_N(x,y,t), v(x,y,t) = B(t)*v_N(x,y,t)
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        t = inputs[:, 2:3]

        N = self.net(torch.cat([x, y, t], dim=1))
        u_N = N[:, 0:1]
        v_N = N[:, 1:2]

        # Trial function (kept identical)
        B = t
        u = B * u_N
        v = B * v_N

        return torch.cat([u, v], dim=1)


# -----------------------------------------------------------------------------
# Fractional time derivative
# -----------------------------------------------------------------------------
class FractionalL1Operator(nn.Module):
    def __init__(self, alpha: float, n: int, device: torch.device):
        super().__init__()
        self.alpha = alpha
        self.n = n
        self.dt = 1.0 / n  # identical: dt = 1/n in training

        w = [1.0]
        for j in range(1, n):
            w.append((j + 1) ** (1 - alpha) - j ** (1 - alpha))

        # Build integration matrix (n+1) x (n+1)
        int_mat = np.zeros((n + 1, n + 1))
        int_mat[0, 0] = 0.0
        for i in range(1, n + 1):
            int_mat[i, 0] = -w[i - 1]
            int_mat[i, i] = w[0]
            for j in range(1, i):
                int_mat[i, j] = w[i - j] - w[i - j - 1]

        self.register_buffer("int_mat", torch.tensor(int_mat, dtype=torch.float32, device=device))
        self.coef = (self.dt ** (-alpha)) / gamma(2 - alpha)  # scalar factor (python float)

    def forward(self, u_flat: torch.Tensor, num_spatial: int) -> torch.Tensor:
        """
        Args:
            u_flat: shape (num_spatial*(n+1),)  (flattened with time as inner dimension)
            num_spatial: M^2

        Returns:
            D_alpha u (flattened), same length as u_flat
        """
        # Keep reshape dimensions identical: (M^2, n+1)
        u_reshaped = u_flat.reshape(num_spatial, self.n + 1)

        splits = torch.split(u_reshaped, 1, dim=0)
        splits = [s.squeeze(0) for s in splits]
        splits_tensor = torch.stack(splits)

        result = torch.matmul(splits_tensor, self.int_mat.T)
        output_tensor = result.flatten()

        return self.coef * output_tensor

# -----------------------------------------------------------------------------
# Physics loss generator (kept identical residuals & boundary conditions)
# -----------------------------------------------------------------------------
class LossGenerator(nn.Module):
    """
    Constructs PDE residuals and boundary terms for fPINNs.
    """

    def __init__(self,
                 model: nn.Module,
                 x: np.ndarray,
                 y: np.ndarray,
                 t: np.ndarray,
                 M: int,
                 alpha: float,
                 n_time: int,
                 R: float):
        super().__init__()

        # Store model reference
        self.model = model

        # Store grid parameters
        self.M = M
        self.alpha = alpha
        self.n_time = n_time
        self.R = R

        # Convert collocation points to tensors
        self.x = torch.tensor(x, requires_grad=True).float().to(DEVICE)
        self.y = torch.tensor(y, requires_grad=True).float().to(DEVICE)
        self.t = torch.tensor(t, requires_grad=True).float().to(DEVICE)

        # Fractional operator
        self.frac_op = FractionalL1Operator(alpha=alpha, n=n_time, device=DEVICE)

    def get_phy_Loss(self):
        """
        Returns:
            f_u, f_v, u boundary residuals (4), v boundary residuals (4)
        """
        re = self.model(torch.cat([self.x, self.y, self.t], dim=1))
        u = re[:, 0]
        v = re[:, 1]

        # First and second derivatives for u
        u_x = torch.autograd.grad(
            u, self.x, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, self.x, grad_outputs=torch.ones_like(u_x),
            retain_graph=True, create_graph=True
        )[0]

        u_y = torch.autograd.grad(
            u, self.y, grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, self.y, grad_outputs=torch.ones_like(u_y),
            retain_graph=True, create_graph=True
        )[0]

        # First and second derivatives for v
        v_x = torch.autograd.grad(
            v, self.x, grad_outputs=torch.ones_like(v),
            retain_graph=True, create_graph=True
        )[0]
        v_xx = torch.autograd.grad(
            v_x, self.x, grad_outputs=torch.ones_like(v_x),
            retain_graph=True, create_graph=True
        )[0]

        v_y = torch.autograd.grad(
            v, self.y, grad_outputs=torch.ones_like(v),
            retain_graph=True, create_graph=True
        )[0]
        v_yy = torch.autograd.grad(
            v_y, self.y, grad_outputs=torch.ones_like(v_y),
            retain_graph=True, create_graph=True
        )[0]

        # Fractional time derivative
        num_spatial = self.M ** 2
        D_alphau = self.frac_op(u, num_spatial=num_spatial)
        D_alphav = self.frac_op(v, num_spatial=num_spatial)

        # Forcing terms
        f1 = gamma(2.5) / gamma(2) * self.t * torch.sin(np.pi * self.x) * torch.sin(np.pi * self.y) + \
             np.pi * self.t ** 3 * torch.sin(np.pi * self.x) * torch.cos(np.pi * self.x) * torch.sin(
            np.pi * self.y) ** 2 + \
             np.pi / 2 * self.t ** 3 * torch.sin(np.pi * self.x) ** 2 * torch.sin(np.pi * self.y) * torch.cos(
            np.pi * self.y) + \
             0.02 * np.pi ** 2 * self.t ** 1.5 * torch.sin(np.pi * self.x) * torch.sin(np.pi * self.y)

        f1 = f1.squeeze()
        f2 = f1 / 2

        # fPDE residuals
        R = self.R
        f_u = D_alphau + u * u_x.squeeze() + v * u_y.squeeze() - R * (u_xx.squeeze() + u_yy.squeeze()) - f1
        f_v = D_alphav + u * v_x.squeeze() + v * v_y.squeeze() - R * (v_xx.squeeze() + v_yy.squeeze()) - f2

        # Boundary values computed through the same model
        # u boundaries
        u_b0 = self.model(torch.cat([self.x, torch.zeros_like(self.y), self.t], dim=1))[:, 0]  # y=0
        u_b1 = self.model(torch.cat([self.x, torch.ones_like(self.y), self.t], dim=1))[:, 0]   # y=1
        u_b2 = self.model(torch.cat([torch.zeros_like(self.x), self.y, self.t], dim=1))[:, 0]  # x=0
        u_b3 = self.model(torch.cat([torch.ones_like(self.x), self.y, self.t], dim=1))[:, 0]   # x=1

        # v boundaries
        v_b0 = self.model(torch.cat([self.x, torch.zeros_like(self.y), self.t], dim=1))[:, 1]  # y=0
        v_b1 = self.model(torch.cat([self.x, torch.ones_like(self.y), self.t], dim=1))[:, 1]   # y=1
        v_b2 = self.model(torch.cat([torch.zeros_like(self.x), self.y, self.t], dim=1))[:, 1]  # x=0
        v_b3 = self.model(torch.cat([torch.ones_like(self.x), self.y, self.t], dim=1))[:, 1]   # x=1

        # Homogeneous Dirichlet boundary conditions
        L_b1, L_b2, L_b3, L_b4 = u_b0, u_b1, u_b2, u_b3
        Lv_b1, Lv_b2, Lv_b3, Lv_b4 = v_b0, v_b1, v_b2, v_b3

        return f_u, f_v, L_b1, L_b2, L_b3, L_b4, Lv_b1, Lv_b2, Lv_b3, Lv_b4


def compute_loss(loss_func: LossGenerator) -> torch.Tensor:
    """Compute total physics-informed loss."""
    mse_loss = nn.MSELoss()

    f_u, f_v, f_b1, f_b2, f_b3, f_b4, f_v1, f_v2, f_v3, f_v4 = loss_func.get_phy_Loss()

    zeros_fu = torch.zeros_like(f_u).to(DEVICE)
    zeros_fv = torch.zeros_like(f_v).to(DEVICE)

    loss_pde = mse_loss(f_u, zeros_fu) + mse_loss(f_v, zeros_fv)

    loss_bc = (
        mse_loss(f_b1, torch.zeros_like(f_b1).to(DEVICE)) +
        mse_loss(f_b2, torch.zeros_like(f_b2).to(DEVICE)) +
        mse_loss(f_b3, torch.zeros_like(f_b3).to(DEVICE)) +
        mse_loss(f_b4, torch.zeros_like(f_b4).to(DEVICE)) +
        mse_loss(f_v1, torch.zeros_like(f_v1).to(DEVICE)) +
        mse_loss(f_v2, torch.zeros_like(f_v2).to(DEVICE)) +
        mse_loss(f_v3, torch.zeros_like(f_v3).to(DEVICE)) +
        mse_loss(f_v4, torch.zeros_like(f_v4).to(DEVICE))
    )

    return loss_pde + 100.0 * loss_bc / 128.0


def train(model: nn.Module,
          x_train: np.ndarray,
          y_train: np.ndarray,
          t_train: np.ndarray,
          n_iters: int,
          learning_rate: float,
          save_path: str,
          pre_model_save_path: str,
          M: int,
          alpha: float,
          n_time: int,
          R: float) -> List[float]:
    """
    Training loop
    """
    train_loss_list: List[float] = []
    best_loss = 1e4

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=CFG.step_size, gamma=CFG.gamma_lr)

    # (Optional) load previous model
    # model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, pre_model_save_path)

    for pg in optimizer.param_groups:
        print(pg["lr"])

    loss_func = LossGenerator(
        model=model,
        x=x_train,
        y=y_train,
        t=t_train,
        M=M,
        alpha=alpha,
        n_time=n_time,
        R=R,
    )

    model.train()
    for epoch in range(n_iters):
        optimizer.zero_grad()

        loss = compute_loss(loss_func)
        loss.backward(retain_graph=True)  # kept identical
        optimizer.step()
        scheduler.step()

        batch_loss = loss.item()
        print(f"[{epoch + 1}/{n_iters} {(epoch + 1) / n_iters * 100:.0f}%] loss: {batch_loss:.10f}")
        train_loss_list.append(batch_loss)

        if batch_loss < best_loss:
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = batch_loss

    return train_loss_list


# -----------------------------------------------------------------------------
# Post-processing & visualization
# -----------------------------------------------------------------------------
def post_process(output: np.ndarray,
                 true: np.ndarray,
                 axis_lim: List[float],
                 uv_lim: List[float],
                 num: int,
                 fig_save_path: str):
    """
    Plot absolute error field
    """
    xmin, xmax, ymin, ymax = axis_lim
    u_min, u_max, v_min, v_max = uv_lim

    x = np.linspace(xmin, xmax, 128)
    x_star, y_star = np.meshgrid(x, x)

    u_star = true[num, 0, :, :]
    v_star = true[num, 1, :, :]

    u_pred = output[num, 0, :, :]
    v_pred = output[num, 1, :, :]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    uuuv = np.abs(v_star - v_pred)
    # print(np.max(uuuv))

    cf = ax.scatter(
        x_star, y_star, c=uuuv, alpha=1, edgecolors="none",
        cmap="RdYlBu", marker="s", s=4
    )

    ax.axis("square")

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(cf, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    plt.savefig(os.path.join(fig_save_path, f"uv_comparison_{str(num).zfill(3)}.png"))
    plt.close("all")

    return u_star, u_pred, v_star, v_pred


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ensure_dir("./model")
    ensure_dir(CFG.fig_save_path)

    # -----------------------------
    # Build training data
    # -----------------------------
    time_nodes_train = np.linspace(0.0, CFG.t_end_train, CFG.N_time_train + 1)  # (N+1,)
    x_train, y_train, t_train = make_spacetime_grid(CFG.M_train, time_nodes_train)

    # -----------------------------
    # Build model
    # -----------------------------
    base_net = DNN(CFG.layers).to(DEVICE)
    model = PINNWrapper(base_net).to(DEVICE)

    # # Parameter statistics
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.numel():,}")
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"总参数量：{total_params}, 可训练参数量：{trainable_params}")

    # -----------------------------
    # Train
    # -----------------------------
    train_loss = train(
        model=model,
        x_train=x_train,
        y_train=y_train,
        t_train=t_train,
        n_iters=CFG.n_iters_adam,
        learning_rate=CFG.lr_adam,
        save_path=CFG.model_save_path,
        pre_model_save_path=CFG.pre_model_save_path,
        M=CFG.M_train,
        alpha=CFG.alpha,
        n_time=CFG.N_time_train,
        R=CFG.R,
    )
    np.savetxt("train_loss.txt", train_loss)

    # -----------------------------
    # Inference
    # -----------------------------
    M = CFG.M_infer
    NN = CFG.NN_infer
    dt = CFG.dt_infer

    time_nodes_infer = np.linspace(0.0, dt * NN, NN + 1)
    x_inf, y_inf, t_inf = make_spacetime_grid(M, time_nodes_infer)

    # Load best checkpoint
    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_path=CFG.model_save_path)

    xp = torch.tensor(x_inf).float().to(DEVICE)
    yp = torch.tensor(y_inf).float().to(DEVICE)
    tp = torch.tensor(t_inf).float().to(DEVICE)

    # Forward pass
    re = model(torch.cat([xp, yp, tp], dim=1))
    u_pred = re[:, :1].detach().cpu().numpy()  # (N*T,1)
    v_pred = re[:, 1:].detach().cpu().numpy()  # (N*T,1)

    N_spatial = M * M
    T_infer = NN + 1

    u_mat = u_pred.reshape(N_spatial, T_infer)           # (M^2, T)
    v_mat = v_pred.reshape(N_spatial, T_infer)           # (M^2, T)

    u_3d = u_mat.reshape(M, M, T_infer)                  # (M, M, T)
    v_3d = v_mat.reshape(M, M, T_infer)                  # (M, M, T)

    u_p = np.transpose(u_3d, (2, 0, 1))                  # (T, M, M)
    u_p = np.expand_dims(u_p, axis=1)                    # (T, 1, M, M)

    v_p = np.transpose(v_3d, (2, 0, 1))                  # (T, M, M)
    v_p = np.expand_dims(v_p, axis=1)                    # (T, 1, M, M)

    # -----------------------------
    # Load reference / exact solution
    # -----------------------------
    u_e = np.load(CFG.u_exact_path)
    v_e = np.load(CFG.u_exact_path) / 2

    truth = np.concatenate((u_e, v_e), axis=1)           # (T, 2, M, M)
    output = np.concatenate((u_p, v_p), axis=1)          # (T, 2, M, M)

    # -----------------------------
    # Post-process: save per-time error fields
    # -----------------------------
    ten_true = []
    ten_pred = []
    ten_true_u = []
    ten_pred_u = []
    ten_true_v = []
    ten_pred_v = []

    for i in range(0, NN + 1):
        u_star, u_hat, v_star, v_hat = post_process(
            output, truth,
            axis_lim=[0, 1, 0, 1],
            uv_lim=[0, 1, 0, 1],
            num=i,
            fig_save_path=CFG.fig_save_path,
        )
        ten_true.append([u_star, v_star])
        ten_pred.append([u_hat, v_hat])

        ten_true_u.append([u_star])
        ten_pred_u.append([u_hat])

        ten_true_v.append([v_star])
        ten_pred_v.append([v_hat])

    # -----------------------------
    # Error metrics
    # -----------------------------
    a_MSE = []
    for i in range(1, NN + 1):
        err = np.array(ten_pred[:i]) - np.array(ten_true[:i])
        Npix = 128 * 128
        acc_rmse = np.sqrt(np.sum(err ** 2) / (Npix * i))
        a_MSE.append(acc_rmse)

    RMSE_u = []
    for i in range(1, NN + 1):
        RMSE_u_value = np.sqrt(np.mean((np.array(ten_pred_u[i]) - np.array(ten_true_u[i])) ** 2))
        RMSE_u.append(RMSE_u_value)

    RMSE_v = []
    for i in range(1, NN + 1):
        RMSE_v_value = np.sqrt(np.mean((np.array(ten_pred_v[i]) - np.array(ten_true_v[i])) ** 2))
        RMSE_v.append(RMSE_v_value)

    print("The predicted error is: ", a_MSE)
    np.savetxt("a_MSE.txt", a_MSE)
    np.savetxt("RMSE_u.txt", RMSE_u)
    np.savetxt("RMSE_v.txt", RMSE_v)

    # # -----------------------------
    # # Time-series plot at (x=0.5,y=0.5)
    # # -----------------------------
    # u_pred_field = output[:, 0, :, :]
    # u_pred_field = np.swapaxes(u_pred_field, 1, 2)   # kept identical
    # u_true_field = truth[:, 0, :, :]
    #
    # t_true = np.linspace(0, dt * NN, NN + 1)
    # t_pred = np.linspace(0, dt * NN, NN + 1)
    #
    # plt.plot(t_pred, u_pred_field[0:NN + 1, 64, 64], label="x=0.5, y=0.5, fPINNs")
    # plt.plot(t_true, u_true_field[0:NN + 1, 64, 64], "--", label="x=0.5, y=0.5, Exact")
    # plt.xlabel("t")
    # plt.ylabel("u")
    # plt.xlim(0, dt * NN)
    # plt.legend()
    # plt.savefig(os.path.join(CFG.fig_save_path, "x=64,y=64.png"))
    # plt.close("all")


if __name__ == "__main__":
    main()
