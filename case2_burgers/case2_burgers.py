"""
FPICRNNs for solving burgers equations
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import weight_norm
from scipy.special import gamma


# -----------------------
# Local env
# -----------------------
# target_path = r''
# sys.path.append(target_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 66
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float32)


# -----------------------
# Grid Hyperparams
# -----------------------
DT = 0.1
DX = 1.0 / 128.0

ALPHA = 0.5
N_FRAC = 9
DIFF_COEF = 0.01
BC_WEIGHT = 100.0


# -----------------------
# PDDO kernels
# -----------------------
lapl_ops = [[[[0.345204460443930, 0.309591078922457, 0.345204460443930],
              [0.309591078922457, -2.619182157203629, 0.309591078922457],
              [0.345204460443930, 0.309591078922457, 0.345204460443930]]]]

partial_x = [[[[-0.086301115118584, 0, 0.086301115118584],
               [-0.327397769700730, 0, 0.327397769700730],
               [-0.086301115118584, 0, 0.086301115118584]]]]

partial_y = [[[[-0.086301115118584, -0.327397769700730, -0.086301115118584],
               [0, 0, 0],
               [0.086301115118584, 0.327397769700730, 0.086301115118584]]]]


# -----------------------
# Weight init
# -----------------------
def initialize_weights(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        c = 1
        module.weight.data.uniform_(
            -c * np.sqrt(1 / (3 * 3 * 320)),
            c * np.sqrt(1 / (3 * 3 * 320))
        )
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        if module.bias is not None:
            module.bias.data.zero_()

# -----------------------
# Model blocks
# -----------------------
class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell"""
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super().__init__()

        self.Wxi = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding, bias=True, padding_mode='zeros')
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False, padding_mode='zeros')

        self.Wxf = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding, bias=True, padding_mode='zeros')
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False, padding_mode='zeros')

        self.Wxc = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding, bias=True, padding_mode='zeros')
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False, padding_mode='zeros')

        self.Wxo = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding, bias=True, padding_mode='zeros')
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False, padding_mode='zeros')

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
    """CNN encoder block."""
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(
            input_channels, hidden_channels,
            input_kernel_size, input_stride, input_padding,
            bias=True, padding_mode='zeros'
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
            setattr(self, f'encoder{i}', EncoderBlock(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i]
            ))

        # convlstm
        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            setattr(self, f'convlstm{i}', ConvLSTMCell(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i]
            ))

        # output heads
        self.output_layer = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2, padding_mode='zeros')
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

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
                x = getattr(self, f'encoder{i}')(x)

            # convlstm
            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                cell = getattr(self, f'convlstm{i}')
                if step == 0:
                    h, c = cell.init_hidden_tensor(prev_state=initial_state[i - self.num_encoder])
                    internal_state.append((h, c))

                h, c = internal_state[i - self.num_encoder]
                x, new_c = cell(x, h, c)
                internal_state[i - self.num_encoder] = (x, new_c)

            # output
            x = self.pixelshuffle(x)
            x = self.output_layer(x)
            x = self.output_layer(x)
            x = self.output_layer(x)

            # residual connection
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
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
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
class loss_generator(nn.Module):
    def __init__(self, dt=DT, dx=DX, f1_path='./model/f1.npy', f2_path='./model/f2.npy'):
        super().__init__()

        self.laplaces = Conv2dDerivative(lapl_ops, resol=(dx ** 2), kernel_size=3, name='laplace').to(device)

        self.dx = Conv2dDerivative(partial_x, resol=(dx * 1), kernel_size=3, name='dx').to(device)
        self.dy = Conv2dDerivative(partial_y, resol=(dx * 1), kernel_size=3, name='dy').to(device)

        self.f1 = torch.from_numpy(np.load(f1_path)).to(device)
        self.f2 = torch.from_numpy(np.load(f2_path)).to(device)

    def _frac_L1(self, x_flat: torch.Tensor) -> torch.Tensor:
        dt_local = (10.0 / 100)
        alpha = ALPHA
        n = N_FRAC
        w = [1]
        for j in range(1, n + 1):
            w.append((j + 1) ** (1 - alpha) - j ** (1 - alpha))
        b = torch.zeros(1, 1, x_flat.shape[-1]).to(device)
        for i in range(1, n + 2):  # 1..10 (n=9)
            c = torch.zeros(1, 1, x_flat.shape[-1]).to(device)
            for k in range(1, i):
                c = c + (w[i - k - 1] - w[i - k]) * x_flat[k, :, :]
            a = w[0] * x_flat[i, :, :] - c - w[i - 1] * x_flat[0, :, :]
            b = torch.cat([b, a], dim=0)
        D_alpha = dt_local ** (-alpha) / gamma(2 - alpha) * b[:, :, :]
        return D_alpha

    def get_loss(self, output: torch.Tensor, loss_type: str) -> torch.Tensor:
        mse_loss = nn.MSELoss()
        # ----- u 分量 -----
        u = output[:, 0:1, :, :]
        lent, _, leny, lenx = u.shape  # 原脚本用 lenx=u.shape[3], leny=u.shape[2]
        u_flat = u.reshape(lent, 1, lenx * leny)
        D_alphau = self._frac_L1(u_flat).reshape(lent, 1, lenx, leny)  # 训练时 lent=11，能对齐原脚本
        laplace_u = self.laplaces(u)         # (T,1,126,126)
        u_x = self.dx(u[:, 0:1, :, :])       # (T,1,126,126)
        u_y = self.dy(u[:, 0:1, :, :])       # (T,1,126,126)


        # ----- v 分量 -----
        v = output[:, 1:2, :, :]
        lent, _, leny, lenx = v.shape
        v_flat = v.reshape(lent, 1, lenx * leny)
        D_alphav = self._frac_L1(v_flat).reshape(lent, 1, lenx, leny)
        laplace_v = self.laplaces(v)         # (T,1,126,126)
        v_x = self.dx(v[:, 0:1, :, :])       # (T,1,126,126)
        v_y = self.dy(v[:, 0:1, :, :])       # (T,1,126,126)

        # ----------------
        # Physics residual
        # ----------------
        if loss_type == 'phy':
            R = DIFF_COEF
            fu = (
                D_alphau[:, :, 1:-1, 1:-1]
                + u[:, :, 1:-1, 1:-1] * u_x
                + v[:, :, 1:-1, 1:-1] * u_y
                - R * laplace_u
                - self.f1[:, :, 1:-1, 1:-1]
            )
            fv = (
                D_alphav[:, :, 1:-1, 1:-1]
                + u[:, :, 1:-1, 1:-1] * v_x
                + v[:, :, 1:-1, 1:-1] * v_y
                - R * laplace_v
                - self.f2[:, :, 1:-1, 1:-1]
            )
            return mse_loss(fu, torch.zeros_like(fu).to(device)) + mse_loss(fv, torch.zeros_like(fv).to(device))

        # ----------------
        # Boundary loss
        # ----------------
        if loss_type == 'BC':
            x = torch.linspace(0, 1, 128).to(device)
            t = torch.linspace(0.1, 1, 10).to(device)
            _X, _T = torch.meshgrid(x, t, indexing='xy')
            fu_x1 = u[1:, :, :, 0]
            fu_x2 = u[1:, :, :, -1]
            fu_x3 = u[1:, :, 0, :]
            fu_x4 = u[1:, :, -1, :]

            fv_x1 = v[1:, :, :, 0]
            fv_x2 = v[1:, :, :, -1]
            fv_x3 = v[1:, :, 0, :]
            fv_x4 = v[1:, :, -1, :]

            x1 = torch.zeros(128, device=device)
            x1_d = x1.unsqueeze(0).unsqueeze(0).repeat(fu_x1.shape[0], 1, 1)

            f_bc = (
                (fu_x1 - x1_d) ** 2 + (fu_x2 - x1_d) ** 2 + (fu_x3 - x1_d) ** 2 + (fu_x4 - x1_d) ** 2
                + (fv_x1 - x1_d) ** 2 + (fv_x2 - x1_d) ** 2 + (fv_x3 - x1_d) ** 2 + (fv_x4 - x1_d) ** 2
            ).mean() / (fu_x1.shape[-1])

            return f_bc

        raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_loss(output: torch.Tensor, loss_func: loss_generator):
    f_phy = loss_func.get_loss(output, 'phy')
    f_bc = loss_func.get_loss(output, 'BC')
    loss = f_phy + BC_WEIGHT * f_bc
    return loss, f_phy, BC_WEIGHT * f_bc


# -----------------------
# Training
# -----------------------
def save_checkpoint(model, optimizer, scheduler, save_path):
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }, save_path)


def load_checkpoint_optional(model, optimizer, scheduler, save_path):
    if (save_path is None) or (not os.path.isfile(save_path)):
        return model, optimizer, scheduler, False

    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f'Pretrained model loaded: {save_path}')
    return model, optimizer, scheduler, True


def train(model, u0, initial_state, n_iters, learning_rate, save_path, pre_model_save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)

    if pre_model_save_path is not None:
        model, optimizer, scheduler, _ = load_checkpoint_optional(model, optimizer, scheduler, pre_model_save_path)

    loss_func = loss_generator(dt=DT, dx=DX, f1_path='./model/f1.npy', f2_path='./model/f2.npy')

    train_loss_list = []
    phy_loss_list = []
    bc_loss_list = []

    best_loss = 1e4

    for epoch in range(n_iters):
        optimizer.zero_grad()

        outputs, second_last_state = model(initial_state, u0)
        output = torch.cat(tuple(outputs), dim=0)             # (steps,2,128,128)
        output = torch.cat((u0.to(device), output), dim=0)    # (steps+1,2,128,128)

        loss, f_phy, f_bc_scaled = compute_loss(output, loss_func)
        loss.backward(retain_graph=True)

        optimizer.step()
        scheduler.step()

        train_loss_list.append(loss.item())
        phy_loss_list.append(f_phy.item())
        bc_loss_list.append(f_bc_scaled.item())  # 已经乘过 100

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{n_iters}]  Loss={loss.item():.6e}  Phy={f_phy.item():.6e}  BCx100={f_bc_scaled.item():.6e}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            save_checkpoint(model, optimizer, scheduler, save_path)

    return train_loss_list, phy_loss_list, bc_loss_list


# -----------------------
# Post-processing
# -----------------------
def frobenius_norm(tensor: np.ndarray) -> float:
    return float(np.sqrt(np.sum(tensor ** 2)))


def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
    xmin, xmax, ymin, ymax = axis_lim

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

    cf = ax.scatter(x_star, y_star, c=uuuv, alpha=1, edgecolors='none',
                    cmap='RdYlBu', marker='s', s=4)
    ax.axis('square')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(cf, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    os.makedirs(fig_save_path, exist_ok=True)
    plt.savefig(os.path.join(fig_save_path, f'uv_comparison_{str(num).zfill(3)}.png'))
    plt.close('all')

    return u_star, u_pred, v_star, v_pred


# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    time_steps = 9
    dt = DT
    dx = DX

    fig_save_path = './figures/'
    os.makedirs(fig_save_path, exist_ok=True)

    # input
    u0 = torch.zeros(1, 2, 128, 128)
    u0 = torch.tensor(u0, dtype=torch.float32).to(device)

    u_e = np.load('./model/u_e.npy')
    v_e = np.load('./model/u_e.npy')
    truth = np.concatenate((u_e, v_e), axis=1)
    all_imgs = np.array(truth)[np.newaxis, :]

    num_convlstm = 1
    h0 = torch.randn(1, 128, 16, 16)  # CPU
    c0 = torch.randn(1, 128, 16, 16)  # CPU
    initial_state = [(h0, c0) for _ in range(num_convlstm)]

    # model params
    time_batch_size = 9
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    lr_adam = 1e-3

    # paths
    n_iters_adam = 2000
    pre_model_save_path = None
    # save paths
    save_dir = f'./tfrde_seed_{SEED}'
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, 'fpicrnns_burgers.pt')

    model = FPICRNNs(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=dt,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps,
        effective_step=effective_step
    ).to(device)


    train_loss, phy_loss, bc_loss = train(
        model=model,
        u0=u0,
        initial_state=initial_state,
        n_iters=n_iters_adam,
        learning_rate=lr_adam,
        save_path=model_save_path,
        pre_model_save_path=pre_model_save_path
    )
    np.savetxt('train_loss.txt', np.array(train_loss))

    # ---------------- inference ----------------
    time_batch_size_load = 15
    steps_load = time_batch_size_load + 1
    effective_step_load = list(range(0, steps_load))
    NN = 17

    model_test = FPICRNNs(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=dt,
        num_layers=[3, 1],
        upscale_factor=8,
        step=steps_load,
        effective_step=effective_step_load
    ).to(device)

    model_test, _, _, loaded = load_checkpoint_optional(model_test, optimizer=None, scheduler=None, save_path=model_save_path)
    if not loaded:
        print(f"[WARN] checkpoint not found: {model_save_path}")
        print("[WARN] skip inference to avoid random/untrained output. (你可以把 n_iters_adam > 0 让它先训练再推理)")
        sys.exit(0)

    # plot train loss
    plt.figure()
    if len(phy_loss) > 0:
        plt.plot(phy_loss, label='phy_loss')
    if len(train_loss) > 0:
        plt.plot(train_loss, label='train loss')
    if len(bc_loss) > 0:
        plt.plot(bc_loss, label='bc_loss(x100)')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(fig_save_path, 'train loss.png'), dpi=300)
    plt.close()

    outputs, _ = model_test(initial_state, u0)
    output = torch.cat(tuple(outputs), dim=0)
    output = torch.cat((u0.to(device), output), dim=0)
    pred = output.detach().cpu().numpy()

    u_e = np.load('./model/u_e.npy')
    v_e = np.load('./model/u_e.npy') / 2
    truth = np.concatenate((u_e, v_e), axis=1)

    ten_true = []
    ten_pred = []

    ten_true_u, ten_pred_u = [], []
    ten_true_v, ten_pred_v = [], []

    for i in range(0, NN):
        u_star, u_pred, v_star, v_pred = post_process(
            pred[:NN, :, :, :],
            truth[:NN, :, :, :],
            [0, 1, 0, 1],
            [0, 0.1, 0, 0.1],
            num=i,
            fig_save_path=fig_save_path
        )

        ten_true.append([u_star])
        ten_pred.append([u_pred])

        ten_true_u.append([u_star])
        ten_pred_u.append([u_pred])

        ten_true_v.append([v_star])
        ten_pred_v.append([v_pred])

    a_MSE = []
    for i in range(1, NN + 1):
        err = np.array(ten_pred[:i]) - np.array(ten_true[:i])
        N = 128 * 128
        acc_rmse = np.sqrt(np.sum(err ** 2) / (N * i))
        a_MSE.append(acc_rmse)

    RMSE_u = []
    for i in range(1, NN):
        RMSE_u_value = np.sqrt(np.mean((np.array(ten_pred_u[i]) - np.array(ten_true_u[i])) ** 2))
        RMSE_u.append(RMSE_u_value)

    RMSE_v = []
    for i in range(1, NN):
        RMSE_v_value = np.sqrt(np.mean((np.array(ten_pred_v[i]) - np.array(ten_true_v[i])) ** 2))
        RMSE_v.append(RMSE_v_value)

    np.savetxt('a_MSE.txt', np.array(a_MSE))
    np.savetxt('RMSE_u.txt', np.array(RMSE_u))
    np.savetxt('RMSE_v.txt', np.array(RMSE_v))

    print('done')
