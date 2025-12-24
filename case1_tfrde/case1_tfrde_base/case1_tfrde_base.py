"""
FPICRNNs for solving ftrde (base)
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
# target_path = r'E:'
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
DX = 1.0 / 128
ALPHA = 0.5
KAPPA = 0.01
BC_WEIGHT = 100.0


# -----------------------
# PDDO kernels
# -----------------------
lapl_ops = [[[[0.345204460443930, 0.309591078922457, 0.345204460443930],
              [0.309591078922457, -2.619182157203629, 0.309591078922457],
              [0.345204460443930, 0.309591078922457, 0.345204460443930]]]]



def initialize_weights(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        c = 1.0
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
    """Convolutional LSTM Cell."""
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super().__init__()
        self.Wxi = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding, bias=True)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False)

        self.Wxf = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding, bias=True)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False)

        self.Wxc = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding, bias=True)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False)

        self.Wxo = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding, bias=True)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False)

        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        g = torch.tanh(self.Wxc(x) + self.Whc(h))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

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

        # output head
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)
        self.output_layer = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2, padding_mode='zeros')
        nn.init.zeros_(self.output_layer.bias)

        self.apply(initialize_weights)

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

            # residual update
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
    """Loss generator for physics residual + BC loss."""
    def __init__(self, dt=DT, dx=DX, f1_path='./model/f1.npy'):
        super().__init__()
        self.dt = dt
        self.dx = dx

        self.laplaces = Conv2dDerivative(lapl_ops, resol=(dx ** 2), kernel_size=3, name='laplace').to(device)

        # load forcing once
        self.f1 = torch.tensor(np.load(f1_path), dtype=torch.float32, device=device)

        # make f1 shape consistent: (T,1,H,W)
        if self.f1.ndim == 3:
            self.f1 = self.f1[:, None, :, :]

    def frac_time_derivative_L1(self, u: torch.Tensor) -> torch.Tensor:
        """
        L1 approximation of time-fractional derivative.
        u: (T,1,H,W)
        return: (T,1,H,W)
        """
        T, _, H, W = u.shape
        N = H * W
        u_flat = u.reshape(T, 1, N)

        n = T - 2
        w = [1.0]
        for j in range(1, n + 1):
            w.append((j + 1) ** (1 - ALPHA) - j ** (1 - ALPHA))

        b = torch.zeros(1, 1, N, device=device)
        for i in range(1, n + 2):
            c = torch.zeros(1, 1, N, device=device)
            for k in range(1, i):
                c = c + (w[i - k - 1] - w[i - k]) * u_flat[k, :, :]
            a = w[0] * u_flat[i, :, :] - c - w[i - 1] * u_flat[0, :, :]
            b = torch.cat([b, a], dim=0)

        D = (self.dt ** (-ALPHA) / gamma(2 - ALPHA)) * b
        return D.reshape(T, 1, H, W)

    def get_loss(self, output, loss_type: str):
        """
        output: (T, C, H, W)
        """
        mse = nn.MSELoss()
        u = output[:, 0:1, :, :]  # only u channel

        # physics residual: D_t^alpha u - kappa*Lap(u) + u - f = 0
        if loss_type == 'phy':
            D_alpha_u = self.frac_time_derivative_L1(u)               # (T,1,H,W)
            lap_u = self.laplaces(u)                                  # (T,1,H-2,W-2)

            # align shapes
            f1 = self.f1
            if f1.shape[0] != u.shape[0]:
                pass

            res = D_alpha_u[:, :, 1:-1, 1:-1] - KAPPA * lap_u + u[:, :, 1:-1, 1:-1] - f1[:, :, 1:-1, 1:-1]
            return mse(res, torch.zeros_like(res))

        # boundary condition loss: all boundaries = 0
        if loss_type == 'BC':
            ub = u[1:]  # (T-1,1,H,W)
            left = ub[:, :, :, 0]
            right = ub[:, :, :, -1]
            bottom = ub[:, :, 0, :]
            top = ub[:, :, -1, :]

            zeros_l = torch.zeros_like(left)
            zeros_r = torch.zeros_like(right)
            zeros_b = torch.zeros_like(bottom)
            zeros_t = torch.zeros_like(top)

            return ((left - zeros_l) ** 2 + (right - zeros_r) ** 2 + (bottom - zeros_b) ** 2 + (top - zeros_t) ** 2).mean()

        raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_loss(output, loss_func: loss_generator):
    f_phy = loss_func.get_loss(output, 'phy')
    f_bc = loss_func.get_loss(output, 'BC')
    loss = f_phy + BC_WEIGHT * f_bc
    return loss, f_phy, f_bc

# -----------------------
# Training
# -----------------------
def save_checkpoint(model, optimizer, scheduler, save_path):
    print(f"Saving model to: {save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_path)


def load_checkpoint(model, optimizer, scheduler, save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print('Pretrained model loaded!')
    return model, optimizer, scheduler


def train(model, u0, initial_state, n_iters, learning_rate, save_path, pre_model_save_path=None):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)

    if pre_model_save_path is not None:
        model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler, pre_model_save_path)

    loss_func = loss_generator(dt=DT, dx=DX, f1_path='./model/f1.npy')

    best = float('inf')
    logs = {'loss': [], 'phy': [], 'bc': []}

    for epoch in range(n_iters):
        optimizer.zero_grad()

        # forward
        outputs, second_last_state = model(initial_state, u0)
        output = torch.cat(tuple(outputs), dim=0)
        output = torch.cat((u0.to(device), output), dim=0)  # prepend u0

        loss, f_phy, f_bc = compute_loss(output, loss_func)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        logs['loss'].append(loss.item())
        logs['phy'].append(f_phy.item())
        logs['bc'].append(f_bc.item())

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{n_iters}]  loss={loss.item():.4e}  phy={f_phy.item():.4e}  bc={f_bc.item():.4e}")

            # save best
            if loss.item() < best:
                best = loss.item()
                save_checkpoint(model, optimizer, scheduler, save_path)
                print(f"Model improved and saved at epoch {epoch}, best={best:.6e}")

    return logs


# -----------------------
# Plotting / Metrics
# -----------------------
def frobenius_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x ** 2)))

def post_process_errmap(pred, true, num, fig_save_path, axis_lim=(0, 1, 0, 1)):
    os.makedirs(fig_save_path, exist_ok=True)

    xmin, xmax, ymin, ymax = axis_lim

    u_star = true[num, 0, :, :]
    u_pred = pred[num, 0, :, :]
    err = np.abs(u_star - u_pred)

    plt.figure(figsize=(4.6, 4.2))
    im = plt.imshow(
        err,
        origin="lower",
        aspect="equal",
        cmap="RdYlBu",
        extent=[xmin, xmax, ymin, ymax],
    )
    plt.colorbar(im)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, f"err_u_{num:03d}.png"), dpi=250)
    plt.close()

    return u_star, u_pred



def plot_train_logs(logs, fig_save_path):
    os.makedirs(fig_save_path, exist_ok=True)
    plt.figure()
    plt.plot(logs['phy'], label='phy_loss')
    plt.plot(logs['loss'], label='total_loss')
    plt.plot(logs['bc'], label='bc_loss')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_path, 'train_loss.png'), dpi=300)
    plt.close()

# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    # output dirs
    fig_save_path = './figures/'
    os.makedirs(fig_save_path, exist_ok=True)

    # input initial frame
    u0 = torch.zeros(1, 2, 128, 128, device=device)

    # truth data
    u_e = np.load('./model/u_e.npy')
    v_e = np.load('./model/u_e.npy')
    truth = np.concatenate((u_e, v_e), axis=1)  # expected shape like (T,2,128,128)

    # initial states
    num_convlstm = 1
    h0 = torch.randn(1, 128, 16, 16, device=device)
    c0 = torch.randn(1, 128, 16, 16, device=device)
    initial_state = [(h0, c0) for _ in range(num_convlstm)]

    # model params
    time_batch_size = 9
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    lr_adam = 1e-3
    n_iters_adam = 2000

    # save paths
    save_dir = f'./tfrde_seed_{SEED}'
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, 'fpicrnns_tfrde.pt')

    # build model
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

    # train
    logs = train(model, u0, initial_state, n_iters_adam, lr_adam, model_save_path, pre_model_save_path=None)

    np.savetxt('train_loss.txt', np.array(logs['loss']))
    plot_train_logs(logs, fig_save_path)

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

    model_test, _, _ = load_checkpoint(model_test, optimizer=None, scheduler=None, save_path=model_save_path)

    outputs, _ = model_test(initial_state, u0)
    output = torch.cat(tuple(outputs), dim=0)
    output = torch.cat((u0.to(device), output), dim=0)
    pred = output.detach().cpu().numpy()

    # ---------------- evaluation & plots ----------------
    NN = 17
    ten_true, ten_pred = [], []

    for i in range(min(NN, pred.shape[0])):
        u_star, u_pred = post_process_errmap(pred, truth, i, fig_save_path)
        ten_true.append(u_star)
        ten_pred.append(u_pred)

    ten_true = np.array(ten_true)  # (T,H,W)
    ten_pred = np.array(ten_pred)

    a_MSE = []
    for i in range(1, ten_pred.shape[0]):
        val = frobenius_norm(ten_pred[:i] - ten_true[:i]) / np.sqrt(128 * 128 * i)
        a_MSE.append(val)

    RMSE_u = []
    for i in range(1, ten_pred.shape[0] + 1):
        val = np.sqrt(np.mean((ten_pred[:i] - ten_true[:i]) ** 2))
        RMSE_u.append(val)

    np.savetxt('a_MSE.txt', np.array(a_MSE))
    np.savetxt('RMSE_u.txt', np.array(RMSE_u))

    print('done')
