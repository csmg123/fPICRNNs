"""
FPICRNNs for solving TFRDE / Diffusion-Reaction (non-homogeneous Dirichlet BCs)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from scipy.special import gamma


# -----------------------
# Local env
# -----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 66
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float32)


# -----------------------
# Grid Hyperparams
# -----------------------
DT = 0.1
DX = 1.0 / 128
ALPHA = 0.5
KAPPA = 0.01
BC_WEIGHT = 100.0
BC_POWER = 1.5


# -----------------------
# PDDO kernels
# -----------------------
lapl_ops = [[[[0.345204460443930, 0.309591078922457, 0.345204460443930],
              [0.309591078922457, -2.619182157203629, 0.309591078922457],
              [0.345204460443930, 0.309591078922457, 0.345204460443930]]]]


w = torch.randn(4, 1, requires_grad=True).to(device)
nn.init.xavier_normal_(w)


def initialize_weights(module: nn.Module):
    """
    - Conv2d: uniform_(-c*sqrt(1/(3*3*320)), +c*sqrt(1/(3*3*320)))
    - Linear: bias=0
    """
    if isinstance(module, nn.Conv2d):
        c = 1.0
        module.weight.data.uniform_(
            -c * np.sqrt(1 / (3 * 3 * 320)),
            c * np.sqrt(1 / (3 * 3 * 320))
        )
    elif isinstance(module, nn.Linear):
        if module.bias is not None:
            module.bias.data.zero_()


# -----------------------
# Model blocks
# -----------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super().__init__()

        self.Wxi = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding,
                             bias=True, padding_mode="zeros")
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1,
                             bias=False, padding_mode="zeros")

        self.Wxf = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding,
                             bias=True, padding_mode="zeros")
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1,
                             bias=False, padding_mode="zeros")

        self.Wxc = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding,
                             bias=True, padding_mode="zeros")
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1,
                             bias=False, padding_mode="zeros")

        self.Wxo = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding,
                             bias=True, padding_mode="zeros")
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1,
                             bias=False, padding_mode="zeros")

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)
        return ch, cc

    @staticmethod
    def init_hidden_tensor(prev_state):
        return prev_state[0].to(device), prev_state[1].to(device)


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(
            input_channels, hidden_channels,
            input_kernel_size, input_stride, input_padding,
            bias=True, padding_mode="zeros"
        ))
        nn.init.zeros_(self.conv.bias)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


class FPICRNNs(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 input_kernel_size, input_stride, input_padding,
                 dt, num_layers, upscale_factor, step=1, effective_step=None):
        super().__init__()

        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.dt = dt
        self.step = step
        self.effective_step = effective_step if effective_step is not None else [1]
        self.upscale_factor = upscale_factor

        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]

        # encoder
        for i in range(self.num_encoder):
            setattr(self, f"encoder{i}", EncoderBlock(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i]
            ))

        # convlstm
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            setattr(self, f"convlstm{i}", ConvLSTMCell(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i]
            ))

        # output
        self.output_layer = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2, padding_mode="zeros")
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

        # init weights
        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, initial_state, x):
        internal_state = []
        outputs = []
        second_last_state = []

        for step in range(self.step):
            xt = x

            # encoder
            for i in range(self.num_encoder):
                x = getattr(self, f"encoder{i}")(x)

            # convlstm
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                cell = getattr(self, f"convlstm{i}")

                if step == 0:
                    h, c = cell.init_hidden_tensor(prev_state=initial_state[i - self.num_encoder])
                    internal_state.append((h, c))

                h, c = internal_state[i - self.num_encoder]
                x, new_c = cell(x, h, c)
                internal_state[i - self.num_encoder] = (x, new_c)

            # output head
            x = self.pixelshuffle(x)
            x = self.output_layer(x)
            x = self.output_layer(x)
            x = self.output_layer(x)

            # residual Euler
            x = xt + self.dt * x

            if step == (self.step - 2):
                second_last_state = internal_state.copy()

            if step in self.effective_step:
                outputs.append(x)

        return outputs, second_last_state


# -----------------------
# Fixed conv derivative ops
# -----------------------
class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=""):
        super().__init__()
        self.resol = resol
        self.name = name
        self.filter = nn.Conv2d(1, 1, kernel_size, 1, padding=0, bias=False)
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, x):
        return self.filter(x) / self.resol


# -----------------------
# Physics loss
# -----------------------
class LossGenerator(nn.Module):
    def __init__(self, dt=DT, dx=DX, f1_path="./model/f1.npy"):
        super().__init__()
        self.dt = dt
        self.dx = dx
        self.laplaces = Conv2dDerivative(lapl_ops, resol=(dx ** 2), kernel_size=3, name="laplace").to(device)

        f1 = torch.tensor(np.load(f1_path)).to(device)
        if f1.ndim == 3:
            f1 = f1[:, None, :, :]  # (T,1,H,W)
        self.f1 = f1

    @staticmethod
    def _frac_time_derivative_L1_strict(u: torch.Tensor) -> torch.Tensor:
        dt_fixed = 10.0 / 100.0
        alpha = 0.5
        n = 9

        T, _, H, W = u.shape
        N = H * W

        u_flat = u.reshape(T, 1, N)

        w_list = [1.0]
        for j in range(1, n + 1):
            w_list.append((j + 1) ** (1 - alpha) - j ** (1 - alpha))

        b = torch.zeros(1, 1, N, device=u.device, dtype=u.dtype)
        for i in range(1, n + 2):  # i = 1..10
            c = torch.zeros(1, 1, N, device=u.device, dtype=u.dtype)
            for k in range(1, i):
                c = c + (w_list[i - k - 1] - w_list[i - k]) * u_flat[k, :, :]
            a = w_list[0] * u_flat[i, :, :] - c - w_list[i - 1] * u_flat[0, :, :]
            b = torch.cat([b, a], dim=0)

        D = (dt_fixed ** (-alpha) / gamma(2 - alpha)) * b
        return D.reshape(T, 1, H, W)

    def get_loss(self, output: torch.Tensor, loss_type: str) -> torch.Tensor:
        mse = nn.MSELoss()

        u = output[:, 0:1, :, :]

        if loss_type == "phy":
            D_alphau = self._frac_time_derivative_L1_strict(u)           # (T,1,H,W)
            lap_u = self.laplaces(u)                                     # (T,1,H-2,W-2)

            f1 = self.f1
            fu = D_alphau[:, :, 1:-1, 1:-1] - KAPPA * lap_u + u[:, :, 1:-1, 1:-1] - f1[:, :, 1:-1, 1:-1]
            return mse(fu, torch.zeros_like(fu)) + mse(fu, torch.zeros_like(fu))

        if loss_type == "BC":
            ub = u[1:]  # (10,1,128,128) 训练时通常这样

            fu_x1 = ub[:, :, :, 0]     # left  (10,1,128)
            fu_x2 = ub[:, :, :, -1]    # right (10,1,128)
            fu_x3 = ub[:, :, 0, :]     # bottom(10,1,128)
            fu_x4 = ub[:, :, -1, :]    # top   (10,1,128)

            x = torch.linspace(0, 1, 128, device=device, dtype=ub.dtype)
            t = torch.linspace(0.1, 1.0, 10, device=device, dtype=ub.dtype)
            X, Tm = torch.meshgrid(x, t, indexing="xy")  # X,T shape: (10,128)

            x1_t = (Tm ** BC_POWER) * torch.sin(2 * torch.pi * X)  # (10,128)
            x1_b = x1_t.unsqueeze(1)                               # (10,1,128)
            x2_t = x1_b

            bc = ((fu_x1 - x1_b) ** 2 +
                  (fu_x2 - x1_b) ** 2 +
                  (fu_x3 - x1_b) ** 2 +
                  (fu_x4 - x2_t) ** 2).mean() / fu_x1.shape[-1]

            return bc
        raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_loss(output: torch.Tensor, loss_func: LossGenerator):
    f_phy = loss_func.get_loss(output, "phy")
    f_bc_raw = loss_func.get_loss(output, "BC")
    f_ic = torch.tensor(0.0, device=device)
    f_data = torch.tensor(0.0, device=device)

    f = f_phy + f_bc_raw + f_ic + f_data
    return f, f_phy, BC_WEIGHT * f_bc_raw, f_ic, f_data


# -----------------------
# Checkpoint I/O
# -----------------------
def save_checkpoint(model, optimizer, scheduler, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, save_path)


def load_checkpoint(model, optimizer, scheduler, save_path):
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print("Pretrained model loaded!")
    return model, optimizer, scheduler


# -----------------------
# Training
# -----------------------
def train(model, u0, initial_state, n_iters, time_batch_size, learning_rate,
          save_path, pre_model_save_path=None):
    train_loss_list, phy_loss_list, bc_loss_list, ic_loss_list, data_loss_list = [], [], [], [], []

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)

    if pre_model_save_path is not None and os.path.exists(pre_model_save_path):
        model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, pre_model_save_path)
    elif pre_model_save_path is not None:
        print(f"[WARN] pre_model_save_path not found, train from scratch: {pre_model_save_path}")

    loss_func = LossGenerator(dt=DT, dx=DX, f1_path="./model/f1.npy")

    steps = time_batch_size + 1
    num_time_batch = int((time_batch_size + 1) / time_batch_size)

    best_loss = 1e10

    prev_output = None
    state_detached = None

    for epoch in range(n_iters):
        optimizer.zero_grad()

        batch_loss = 0.0
        phy_loss = 0.0
        bc_loss = 0.0
        ic_loss = 0.0
        data_loss = 0.0

        for time_batch_id in range(num_time_batch):
            if time_batch_id == 0:
                hidden_state = initial_state
                u_in = u0
            else:
                hidden_state = state_detached
                u_in = prev_output[-2:-1].detach()

            outputs, second_last_state = model(hidden_state, u_in)
            out = torch.cat(tuple(outputs), dim=0)
            out = torch.cat((u_in.to(device), out), dim=0)  # prepend u0

            f, f_phy, f_bc_scaled, f_ic, f_data = compute_loss(out, loss_func)

            loss = f_phy + f_bc_scaled
            loss.backward(retain_graph=True)

            batch_loss += loss.item()
            phy_loss += f_phy.item()
            bc_loss += f_bc_scaled.item()
            ic_loss += f_ic.item()
            data_loss += f_data.item()

            prev_output = out

            # detach second_last_state
            state_detached = []
            for (h, c) in second_last_state:
                state_detached.append((h.detach(), c.detach()))

        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{n_iters}] Loss={loss.item():.4e}  phy={f_phy.item():.4e}  bc(100x)={f_bc_scaled.item():.4e}")

        train_loss_list.append(batch_loss)
        phy_loss_list.append(phy_loss)
        bc_loss_list.append(bc_loss)
        ic_loss_list.append(ic_loss)
        data_loss_list.append(data_loss)

        if batch_loss < best_loss:
            best_loss = batch_loss
            save_checkpoint(model, optimizer, scheduler, save_path)

    return train_loss_list, phy_loss_list, bc_loss_list, ic_loss_list, data_loss_list


# -----------------------
# Plotting / Post-process
# -----------------------
def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
    xmin, xmax, ymin, ymax = axis_lim

    x = np.linspace(xmin, xmax, 128)
    x_star, y_star = np.meshgrid(x, x)

    u_star = true[num, 0, :, :]
    u_pred = output[num, 0, :, :]

    # uuuv = np.abs(u_star - u_pred)
    # print(np.max(uuuv))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax.scatter(x_star, y_star, c=u_star, alpha=1, edgecolors="none",
                    cmap="RdYlBu", marker="s", s=4)
    ax.axis("square")

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(cf, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    os.makedirs(fig_save_path, exist_ok=True)
    plt.savefig(os.path.join(fig_save_path, f"uv_comparison_{num:03d}.png"))
    plt.close("all")

    return u_star, u_pred


def plot_train_curves(train_loss, phy_loss, bc_loss, ic_loss, data_loss, fig_save_path):
    os.makedirs(fig_save_path, exist_ok=True)
    plt.figure()
    plt.plot(phy_loss, label="phy_loss")
    plt.plot(train_loss, label="train loss")
    plt.plot(bc_loss, label="bc_loss")
    plt.plot(ic_loss, label="ic loss")
    plt.plot(data_loss, label="data loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(fig_save_path, "train loss.png"), dpi=300)
    plt.close()


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    fig_save_path = "./figures/"
    os.makedirs(fig_save_path, exist_ok=True)

    u0 = torch.zeros(1, 2, 128, 128, device=device)

    # truth
    u_e = np.load("./model/u_e.npy")
    v_e = np.load("./model/u_e.npy")
    truth = np.concatenate((u_e, v_e), axis=1)

    # initial states
    h0 = torch.randn(1, 128, 16, 16)
    c0 = torch.randn(1, 128, 16, 16)
    initial_state = [(h0, c0)]  # num_convlstm=1

    # build model (train)
    time_batch_size = 9
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))

    model = FPICRNNs(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=DT,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps,
        effective_step=effective_step
    ).to(device)

    # save paths
    save_dir = f'./tfrde_seed_{SEED}'
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "fpicrnns_tfrde_nhdbc.pt")

    n_iters_adam = 2000
    lr_adam = 1e-3

    train_loss, phy_loss, bc_loss, ic_loss, data_loss = train(
        model=model,
        u0=u0,
        initial_state=initial_state,
        n_iters=n_iters_adam,
        time_batch_size=time_batch_size,
        learning_rate=lr_adam,
        save_path=model_save_path,
        pre_model_save_path=model_save_path  # 文件存在就续训；不存在就从头训
    )

    np.savetxt("train_loss.txt", np.array(train_loss))
    plot_train_curves(train_loss, phy_loss, bc_loss, ic_loss, data_loss, fig_save_path)

    # ---------------- inference ----------------
    time_batch_size_load = 15
    steps_load = time_batch_size_load + 1
    effective_step_load = list(range(0, steps_load))

    model_test = FPICRNNs(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=DT,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps_load,
        effective_step=effective_step_load
    ).to(device)

    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"checkpoint not found: {model_save_path}")

    model_test, _, _ = load_checkpoint(model_test, optimizer=None, scheduler=None, save_path=model_save_path)

    outputs, _ = model_test(initial_state, u0)
    output = torch.cat(tuple(outputs), dim=0)
    output = torch.cat((u0.to(device), output), dim=0)
    pred = output.detach().cpu().numpy()

    # ---------------- evaluation ----------------
    NN = 17
    ten_true, ten_pred = [], []

    for i in range(0, NN):
        u_star, u_pred = post_process(
            output=pred[:NN, :, :, :],
            true=truth[:NN, :, :, :],
            axis_lim=[0, 1, 0, 1],
            uv_lim=[0, 0.1, 0, 0.1],
            num=i,
            fig_save_path=fig_save_path
        )
        ten_true.append([u_star])
        ten_pred.append([u_pred])

    a_MSE = []
    for i in range(1, NN + 1):
        err = np.array(ten_pred[:i]) - np.array(ten_true[:i])
        N = 128 * 128
        acc_rmse = np.sqrt(np.sum(err ** 2) / (N * i))
        a_MSE.append(acc_rmse)

    RMSE_u = []
    for i in range(1, NN):
        rmse = np.sqrt(np.mean((np.array(ten_pred[i]) - np.array(ten_true[i])) ** 2))
        RMSE_u.append(rmse)

    np.savetxt("a_MSE.txt", np.array(a_MSE))
    np.savetxt("RMSE_u.txt", np.array(RMSE_u))

    print("done")
