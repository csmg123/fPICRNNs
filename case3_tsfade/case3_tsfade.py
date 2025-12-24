import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import weight_norm

# ============================================================
# 1. Config
# ============================================================
DEVICE_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(DEVICE_ID)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED = 66
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float32)

# cudnn 加速
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 是否训练
DO_TRAIN = True

# 网格/方程参数
GRID = 128
DT = 0.1
DX = 1.0 / 128
ALPHA = 0.5
BETA = 1.5
T_POW = 1.5

# 训练：T_total_train = 11（t=0..1.0）
TIME_BATCH_SIZE_TRAIN = 9
STEPS_TRAIN = TIME_BATCH_SIZE_TRAIN + 1      # 10
T_TOTAL_TRAIN = STEPS_TRAIN + 1              # 11

# 推理：输出 17 帧（t=0..1.6）
TIME_BATCH_SIZE_TEST = 15
STEPS_TEST = TIME_BATCH_SIZE_TEST + 1        # 16
T_TOTAL_TEST = STEPS_TEST + 1                # 17

# 训练超参
N_ITERS = 2000
LR = 1e-3
USE_AMP = True

# 是否强制生成匹配的 f.npy
FORCE_GENERATE_F1 = True
F_PATH = './model/f.npy'

ENFORCE_V_ZERO = True

# 保存路径
MODEL_SAVE_PATH = './case3_tsfade.pt'
FIG_SAVE_PATH = './figures/'
os.makedirs(FIG_SAVE_PATH, exist_ok=True)
os.makedirs("./model", exist_ok=True)

# 推理后处理帧数
NN = 17


# ============================================================
# 2. PDDO Kernels
# ============================================================
partial_x = [[[[-0.086301115118584, 0, 0.086301115118584],
               [-0.327397769700730, 0, 0.327397769700730],
               [-0.086301115118584, 0, 0.086301115118584]]]]

partial_y = [[[[-0.086301115118584, -0.327397769700730, -0.086301115118584],
               [0, 0, 0],
               [0.086301115118584, 0.327397769700730, 0.086301115118584]]]]


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        c = 1
        module.weight.data.uniform_(
            -c * np.sqrt(1 / (3 * 3 * 320)),
            c * np.sqrt(1 / (3 * 3 * 320))
        )
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


# ============================================================
# 3. FPICRNNs Model Blocks
# ============================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super().__init__()
        self.Wxi = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding,
                             bias=True, padding_mode='zeros')
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels,
                             3, 1, padding=1, bias=False, padding_mode='zeros')

        self.Wxf = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding,
                             bias=True, padding_mode='zeros')
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels,
                             3, 1, padding=1, bias=False, padding_mode='zeros')

        self.Wxc = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding,
                             bias=True, padding_mode='zeros')
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels,
                             3, 1, padding=1, bias=False, padding_mode='zeros')

        self.Wxo = nn.Conv2d(input_channels, hidden_channels,
                             input_kernel_size, input_stride, input_padding,
                             bias=True, padding_mode='zeros')
        self.Who = nn.Conv2d(hidden_channels, hidden_channels,
                             3, 1, padding=1, bias=False, padding_mode='zeros')

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

    def init_hidden_tensor(self, prev_state):
        return (Variable(prev_state[0]).to(device), Variable(prev_state[1]).to(device))


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super().__init__()
        self.conv = weight_norm(nn.Conv2d(
            input_channels, hidden_channels,
            input_kernel_size, input_stride, input_padding,
            bias=True, padding_mode='zeros'
        ))
        self.act = nn.ReLU()
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))


class FPICRNNs(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 input_kernel_size, input_stride, input_padding, dt,
                 num_layers, upscale_factor, step=1, effective_step=(1,)):
        super().__init__()
        self.dt = float(dt)
        self.step = int(step)
        self.effective_step = set(effective_step)
        self.upscale_factor = int(upscale_factor)

        self.num_encoder = int(num_layers[0])
        self.num_convlstm = int(num_layers[1])

        in_channels = [input_channels] + hidden_channels

        self.encoders = nn.ModuleList([
            EncoderBlock(in_channels[i], hidden_channels[i],
                         input_kernel_size[i], input_stride[i], input_padding[i])
            for i in range(self.num_encoder)
        ])

        self.convlstms = nn.ModuleList([
            ConvLSTMCell(in_channels[self.num_encoder + i],
                         hidden_channels[self.num_encoder + i],
                         input_kernel_size[self.num_encoder + i],
                         input_stride[self.num_encoder + i],
                         input_padding[self.num_encoder + i])
            for i in range(self.num_convlstm)
        ])

        self.output_layer = nn.Conv2d(2, 2, kernel_size=5, stride=1,
                                      padding=2, padding_mode='zeros')
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, initial_state, x):
        internal_state = [None] * len(self.convlstms)
        outputs = []
        second_last_state = None

        for t in range(self.step):
            xt = x

            for enc in self.encoders:
                x = enc(x)

            for li, cell in enumerate(self.convlstms):
                if t == 0:
                    internal_state[li] = cell.init_hidden_tensor(initial_state[li])
                h, c = internal_state[li]
                x, new_c = cell(x, h, c)
                internal_state[li] = (x, new_c)

            x = self.pixelshuffle(x)
            x = self.output_layer(x)
            x = self.output_layer(x)
            x = self.output_layer(x)

            x = xt + self.dt * x

            if ENFORCE_V_ZERO:
                x = torch.cat([x[:, 0:1], torch.zeros_like(x[:, 1:2])], dim=1)

            if t == (self.step - 2):
                second_last_state = [(h, c) for (h, c) in internal_state]

            if t in self.effective_step:
                outputs.append(x)

        if second_last_state is None:
            second_last_state = [(h, c) for (h, c) in internal_state]

        return outputs, second_last_state


# ============================================================
# 4. Derivative Modules
# ============================================================
class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super().__init__()
        self.resol = float(resol)
        self.name = name
        self.filter = nn.Conv2d(1, 1, kernel_size, 1, padding=0, bias=False)
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        return self.filter(x) / self.resol


# ============================================================
# 5. Fractional Derivatives
# ============================================================
class FractionalGL2D(nn.Module):
    def __init__(self, alpha=0.5, beta=1.5, dt=0.1, nx=128, ny=128):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dt = float(dt)
        self.nx = int(nx)
        self.ny = int(ny)

        hx = 1.0 / (self.nx - 1)
        hy = 1.0 / (self.ny - 1)

        gx = torch.zeros(self.nx, dtype=torch.float32)
        gx[0] = 1.0
        for k in range(1, self.nx):
            gx[k] = (1.0 - (self.beta + 1.0) / k) * gx[k - 1]

        gy = torch.zeros(self.ny, dtype=torch.float32)
        gy[0] = 1.0
        for k in range(1, self.ny):
            gy[k] = (1.0 - (self.beta + 1.0) / k) * gy[k - 1]

        self.register_buffer("kernel_x", (hx ** (-self.beta)) * gx.view(1, 1, self.nx))
        self.register_buffer("kernel_y", (hy ** (-self.beta)) * gy.view(1, 1, self.ny))
        self.pad_x = self.nx - 1
        self.pad_y = self.ny - 1

        self.coef_time = (self.dt ** (-self.alpha)) / math.gamma(2 - self.alpha)
        self._A = None
        self._A_T = None
        self._A_device = None
        self._A_dtype = None

    def _build_time_matrix(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n = T - 2
        w = torch.zeros(n + 1, device=device, dtype=dtype)
        w[0] = 1.0
        if n >= 1:
            j = torch.arange(1, n + 1, device=device, dtype=dtype)
            w[1:] = (j + 1).pow(1 - self.alpha) - j.pow(1 - self.alpha)

        A = torch.zeros((T, T), device=device, dtype=dtype)
        for i in range(1, T):
            A[i, i] = w[0]
            A[i, 0] = A[i, 0] - w[i - 1]
            for k in range(1, i):
                A[i, k] = A[i, k] - (w[i - k - 1] - w[i - k])
        return A

    def time(self, u: torch.Tensor) -> torch.Tensor:
        u = u.float()
        T, _, H, W = u.shape
        if (self._A is None or self._A_T != T or self._A_device != u.device or self._A_dtype != u.dtype):
            self._A = self._build_time_matrix(T, u.device, u.dtype)
            self._A_T = T
            self._A_device = u.device
            self._A_dtype = u.dtype

        u_flat = u[:, 0].reshape(T, -1)   # [T,N]
        b = self._A @ u_flat
        D = self.coef_time * b
        return D.reshape(T, 1, H, W)

    def space_x(self, u: torch.Tensor) -> torch.Tensor:
        u = u.float()
        T, _, H, W = u.shape
        assert W == self.nx

        u_seq = u.permute(0, 2, 1, 3).contiguous().view(T * H, 1, W)

        full_right = F.conv1d(u_seq, self.kernel_x, padding=self.pad_x)
        right = full_right[:, :, self.pad_x:self.pad_x + W]

        u_rev = torch.flip(u_seq, dims=[2])
        full_right_rev = F.conv1d(u_rev, self.kernel_x, padding=self.pad_x)
        left = torch.flip(full_right_rev[:, :, self.pad_x:self.pad_x + W], dims=[2])

        ux = left + right  # 与你的 a+b 一致（m=0 双计）
        return ux.view(T, H, 1, W).permute(0, 2, 1, 3).contiguous()

    def space_y(self, u: torch.Tensor) -> torch.Tensor:
        u = u.float()
        T, _, H, W = u.shape
        assert H == self.ny

        u_seq = u.permute(0, 3, 1, 2).contiguous().view(T * W, 1, H)

        full_right = F.conv1d(u_seq, self.kernel_y, padding=self.pad_y)
        right = full_right[:, :, self.pad_y:self.pad_y + H]

        u_rev = torch.flip(u_seq, dims=[2])
        full_right_rev = F.conv1d(u_rev, self.kernel_y, padding=self.pad_y)
        left = torch.flip(full_right_rev[:, :, self.pad_y:self.pad_y + H], dims=[2])

        uy = left + right
        return uy.view(T, W, 1, H).permute(0, 2, 3, 1).contiguous()


# ============================================================
# 6. Loss
# ============================================================
class LossGeneratorFast(nn.Module):
    def __init__(self, dt, dx, alpha=0.5, beta=1.5, grid_size=128, f_path='./model/f.npy'):
        super().__init__()
        self.dx_op = Conv2dDerivative(partial_x, resol=dx, kernel_size=3, name='dx').to(device)
        self.dy_op = Conv2dDerivative(partial_y, resol=dx, kernel_size=3, name='dy').to(device)
        self.frac = FractionalGL2D(alpha=alpha, beta=beta, dt=dt, nx=grid_size, ny=grid_size).to(device)

        if f_path is not None and os.path.exists(f_path):
            f_np = np.load(f_path)
            f_t = torch.tensor(f_np, dtype=torch.float32)
            if f_t.ndim == 3:
                f_t = f_t.unsqueeze(1)
            self.register_buffer('f1', f_t)
        else:
            self.f1 = None
            print(f"[Warning] f_path not found: {f_path}. Will use zeros as f.")

    def physics_loss(self, output):
        u = output[:, 0:1, :, :].float()

        D_alphau = self.frac.time(u)      # [T,1,H,W]
        ux = self.frac.space_x(u)         # [T,1,H,W]
        uy = self.frac.space_y(u)         # [T,1,H,W]
        u_x = self.dx_op(u)               # [T,1,H-2,W-2]
        u_y = self.dy_op(u)               # [T,1,H-2,W-2]

        if self.f1 is None:
            f1 = torch.zeros_like(u)
        else:
            f1 = self.f1
            if f1.shape[0] != output.shape[0]:
                f1 = f1[:output.shape[0]]
            f1 = f1.to(output.device).float()

        fu = (
            D_alphau[:, :, 1:-1, 1:-1]
            - 0.1 * (u_x + u_y)
            - 0.01 * (ux[:, :, 1:-1, 1:-1] + uy[:, :, 1:-1, 1:-1])
            - f1[:, :, 1:-1, 1:-1]
        )
        return (fu * fu).mean()

    def bc_loss(self, output):
        # u=0 边界（保持原样）
        u = output[:, 0:1, :, :]
        u1 = u[1:]
        fu_x1 = u1[:, :, :, 0]
        fu_x2 = u1[:, :, :, -1]
        fu_x3 = u1[:, :, 0, :]
        fu_x4 = u1[:, :, -1, :]
        f = (fu_x1.pow(2) + fu_x2.pow(2) + fu_x3.pow(2) + fu_x4.pow(2)).mean() / fu_x1.shape[-1]
        return f


def compute_loss(output, loss_func: LossGeneratorFast):
    f_phy = loss_func.physics_loss(output)
    f_bc = loss_func.bc_loss(output)
    f_bc_scaled = 100.0 * f_bc
    total = f_phy + f_bc_scaled
    return total, f_phy, f_bc_scaled


# ============================================================
# 7. Exact Solution & f.npy Generation
# ============================================================
def exact_u(T, dt, M, device):
    x = torch.linspace(0, 1, M, device=device)
    y = torch.linspace(0, 1, M, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    t = (torch.arange(T, device=device, dtype=torch.float32) * dt).view(T, 1, 1, 1)
    u = (t ** T_POW) * (torch.sin(math.pi * X) ** 2).view(1, 1, M, M) * (torch.sin(math.pi * Y) ** 2).view(1, 1, M, M)
    return u


@torch.no_grad()
def generate_f1_file_train():
    tmp_loss = LossGeneratorFast(dt=DT, dx=DX, alpha=ALPHA, beta=BETA, grid_size=GRID, f_path=None).to(device)
    u = exact_u(T_TOTAL_TRAIN, DT, GRID, device)  # [11,1,128,128]

    D = tmp_loss.frac.time(u)
    ux = tmp_loss.frac.space_x(u)
    uy = tmp_loss.frac.space_y(u)
    u_x = tmp_loss.dx_op(u)
    u_y = tmp_loss.dy_op(u)

    f1 = torch.zeros_like(u)
    f1[:, :, 1:-1, 1:-1] = (
        D[:, :, 1:-1, 1:-1]
        - 0.1 * (u_x + u_y)
        - 0.01 * (ux[:, :, 1:-1, 1:-1] + uy[:, :, 1:-1, 1:-1])
    )

    np.save(F_PATH, f1.cpu().numpy().astype(np.float32))
    fu = (
        D[:, :, 1:-1, 1:-1]
        - 0.1 * (u_x + u_y)
        - 0.01 * (ux[:, :, 1:-1, 1:-1] + uy[:, :, 1:-1, 1:-1])
        - f1[:, :, 1:-1, 1:-1]
    )


# ============================================================
# 8. Checkpoint IO
# ============================================================
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
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f"Pretrained model loaded! ({save_path})")
    return model, optimizer, scheduler, True


# ============================================================
# 9. Train / Eval Pipeline
# ============================================================
def train_one_run(model, input0, initial_state, n_iters, lr, save_path, use_amp=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)

    loss_func = LossGeneratorFast(dt=DT, dx=DX, alpha=ALPHA, beta=BETA, grid_size=GRID, f_path=F_PATH).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))

    best_loss = 1e30
    train_loss_list, phy_loss_list, bc_loss_list = [], [], []

    for epoch in range(n_iters):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available())):
            outputs, _ = model(initial_state, input0)      # list length=STEPS_TRAIN
            out = torch.cat(outputs, dim=0)                # [10,2,128,128]
            out = torch.cat((input0, out), dim=0)          # [11,2,128,128]

        total_loss, f_phy, f_bc = compute_loss(out.float(), loss_func)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss_list.append(total_loss.item())
        phy_loss_list.append(f_phy.item())
        bc_loss_list.append(f_bc.item())

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{n_iters}]  Loss: {total_loss.item():.6e}  phy: {f_phy.item():.6e}  bc*100: {f_bc.item():.6e}")

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            save_checkpoint(model, optimizer, scheduler, save_path)
            if epoch % 100 == 0:
                print(f" Model improved and saved at epoch {epoch}, loss = {best_loss:.6e}")

    np.savetxt('train_loss.txt', np.array(train_loss_list))
    np.savetxt('phy_loss.txt', np.array(phy_loss_list))
    np.savetxt('bc_loss.txt', np.array(bc_loss_list))

    return train_loss_list, phy_loss_list, bc_loss_list


def run_inference(model_test, input0, initial_state, ckpt_path):
    model_test, _, _, loaded = load_checkpoint_optional(model_test, optimizer=None, scheduler=None, save_path=ckpt_path)
    if not loaded:
        print(f"[WARN] checkpoint not found: {ckpt_path}")
        print("[WARN] inference skipped (no trained weights). Set DO_TRAIN=True or provide checkpoint.")
        return None

    model_test.eval()
    with torch.no_grad():
        out_list, _ = model_test(initial_state, input0)
        out = torch.cat(out_list, dim=0)           # [16,2,128,128]
        output = torch.cat((input0, out), dim=0)   # [17,2,128,128]
    return output


# ============================================================
# 10. Metrics / Post-process
# ============================================================
def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))


def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
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

    cf = ax.scatter(x_star, y_star, c=u_pred, alpha=1, edgecolors='none',
                    cmap='RdYlBu', marker='s', s=4)
    ax.axis('square')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(cf, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    os.makedirs(fig_save_path, exist_ok=True)
    plt.savefig(os.path.join(fig_save_path, 'uv_comparison_' + str(num).zfill(3) + '.png'), dpi=200)
    plt.close('all')

    return u_star, u_pred, v_star, v_pred


def evaluate_and_save(output_torch, nn_frames=NN):
    u_true = exact_u(output_torch.shape[0], DT, GRID, device).squeeze(1).detach().cpu().numpy()  # [T,128,128]
    truth = np.zeros((output_torch.shape[0], 2, GRID, GRID), dtype=np.float32)
    truth[:, 0] = u_true
    truth[:, 1] = 0.0

    ten_true = []
    ten_pred = []

    output = output_torch[:, :, :, :].detach().cpu().numpy()
    NN_use = min(nn_frames, output.shape[0])

    for i in range(0, NN_use):
        u_star, u_pred, v_star, v_pred = post_process(
            output[:NN_use, :, :, :],
            truth[:NN_use, :, :, :],
            [0, 1, 0, 1],
            [0, 0.1, 0, 0.1],
            num=i,
            fig_save_path=FIG_SAVE_PATH
        )
        ten_true.append([u_star])
        ten_pred.append([u_pred])

    a_MSE = []
    for i in range(1, NN_use):
        a_MSE_value = frobenius_norm(np.array(ten_pred[:i]) - np.array(ten_true[:i])) / np.sqrt((128 * 128 * (i + 1)))
        a_MSE = np.append(a_MSE, a_MSE_value)

    RMSE_u = []
    for i in range(1, NN_use + 1):
        RMSE_u_value = np.sqrt(np.mean((np.array(ten_pred[:i]) - np.array(ten_true[:i])) ** 2))
        RMSE_u = np.append(RMSE_u, RMSE_u_value)

    np.savetxt('a_MSE.txt', a_MSE)
    np.savetxt('RMSE_u.txt', RMSE_u)
    print("done")


# ============================================================
# 11. Main
# ============================================================
if __name__ == '__main__':

    if FORCE_GENERATE_F1:
        generate_f1_file_train()
    input0 = torch.zeros(1, 2, GRID, GRID, dtype=torch.float32, device=device)
    num_convlstm = 1
    h0 = torch.randn(1, 128, 16, 16, device=device)
    c0 = torch.randn(1, 128, 16, 16, device=device)
    initial_state = [(h0, c0) for _ in range(num_convlstm)]
    model = FPICRNNs(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=DT,
        num_layers=[3, 1],
        upscale_factor=8,
        step=STEPS_TRAIN,
        effective_step=list(range(0, STEPS_TRAIN))
    ).to(device)
    if DO_TRAIN:
        _optimizer = optim.Adam(model.parameters(), lr=LR)
        _scheduler = StepLR(_optimizer, step_size=100, gamma=0.97)
        model, _optimizer, _scheduler, _loaded = load_checkpoint_optional(model, _optimizer, _scheduler, MODEL_SAVE_PATH)
        del _optimizer, _scheduler, _loaded

        train_one_run(model, input0, initial_state, N_ITERS, LR, MODEL_SAVE_PATH, use_amp=USE_AMP)
    model_test = FPICRNNs(
        input_channels=2,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        dt=DT,
        num_layers=[3, 1],
        upscale_factor=8,
        step=STEPS_TEST,
        effective_step=list(range(0, STEPS_TEST))
    ).to(device)

    output = run_inference(model_test, input0, initial_state, MODEL_SAVE_PATH)
    if output is None:
        sys.exit(0)

    evaluate_and_save(output, nn_frames=NN)
