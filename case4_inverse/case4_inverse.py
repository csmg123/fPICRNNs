"""
FPICRNNs_inverse
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt

# ============================================================
# 1) Config
# ============================================================
CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(66)
np.random.seed(66)

# data / grid
GRID_H = 128
GRID_W = 128

# train/infer time
time_steps = 10
dt = 0.1
dx = 1.0 / 128

# training hyper-params
time_batch_size = 9
steps = time_batch_size + 1
effective_step = list(range(0, steps))
num_time_batch = int(time_steps / time_batch_size)
n_iters_adam = 100000
lr_adam = 1e-3

pre_model_save_path = None
# save_dir = f'./tfrde_'
# os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join( 'fpicrnns_burgers.pt')
fig_save_path = "./figures/"

# ensure folders
os.makedirs("./model", exist_ok=True)
os.makedirs(fig_save_path, exist_ok=True)

# ============================================================
# 2) Kernels (PDDO)
# ============================================================
partial_y = [[[
    [-0.000468920326102395, -0.00900859236568725, -0.0241273465151258, -0.00900859236568725, -0.000468920326102395],
    [-0.00450429618284362, -0.0865336091161437, -0.231759445604615, -0.0865336091161437, -0.00450429618284362],
    [0, 0, 0, 0, 0 ],
    [0.00450429618284362, 0.0865336091161437, 0.231759445604615, 0.0865336091161437, 0.00450429618284362],
    [0.000468920326102395, 0.00900859236568725, 0.0241273465151258, 0.00900859236568725, 0.000468920326102395]
    ]]]

partial_x = [[[
    [-0.000468920326102395, -0.00450429618284362, 0, 0.00450429618284362, 0.000468920326102395],
    [-0.00900859236568725, -0.0865336091161437, 0, 0.0865336091161437, 0.00900859236568725],
    [-0.0241273465151258, -0.231759445604615, 0, 0.231759445604615, 0.0241273465151258],
    [-0.00900859236568725, -0.0865336091161437, 0, 0.0865336091161437, 0.00900859236568725],
    [-0.000468920326102395, -0.00450429618284362, 0, 0.00450429618284362, 0.000468920326102395],
]]]

lapl_op = [[[
    [0.00323955421847909, 0.0355232697130541, 0.0712924688376999, 0.0355232697130541, 0.00323955421847909],
    [0.0355232697130541, 0.169258339550749, -0.00483568533067183, 0.169258339550749, 0.0355232697130541],
    [0.0712924688376999, -0.00483568533067183, -1.24000486680946, -0.00483568533067183, 0.0712924688376999],
    [0.0355232697130541, 0.169258339550749, -0.00483568533067183, 0.169258339550749, 0.0355232697130541],
    [0.00323955421847909, 0.0355232697130541, 0.0712924688376999, 0.0355232697130541, 0.00323955421847909],
]]]

# ============================================================
# 3) Utils
# ============================================================
def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        c = 1
        module.weight.data.uniform_(
            -c * np.sqrt(1 / (3 * 3 * 320)),
            c * np.sqrt(1 / (3 * 3 * 320)),
        )
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()

def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))

# ============================================================
# 4) Model
# ============================================================
class EncoderBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size, input_stride, input_padding):
        super().__init__()
        self.conv = weight_norm(
            nn.Conv2d(
                input_channels,
                hidden_channels,
                input_kernel_size,
                input_stride,
                input_padding,
                bias=True,
                padding_mode="circular",
            )
        )
        self.act = nn.ReLU()
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.conv(x))

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size, input_stride, input_padding):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_kernel_size = 3
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.input_kernel_size, self.input_stride, self.input_padding,
                             bias=True, padding_mode='circular')
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.hidden_kernel_size, 1, padding=1, bias=False,
                             padding_mode='circular')

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

class FPICRNNs(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 input_kernel_size, input_stride, input_padding, dt,
                 num_layers, upscale_factor, step=1, effective_step=[1]):
        super().__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.dt = dt
        self.upscale_factor = upscale_factor
        self.num_encoder = num_layers[0]
        self.num_convlstm = num_layers[1]

        for i in range(self.num_encoder):
            name = 'encoder{}'.format(i)
            cell = EncoderBlock(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i]
            )
            setattr(self, name, cell)
            self._all_layers.append(cell)

        for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
            name = 'convlstm{}'.format(i)
            cell = ConvLSTMCell(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i]
            )
            setattr(self, name, cell)
            self._all_layers.append(cell)

        self.output_layer = nn.Conv2d(2, 2, kernel_size=5, stride=1,
                                      padding=2, padding_mode='circular')
        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)

        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, initial_state, x):
        self.initial_state = initial_state
        internal_state = []
        outputs = []
        second_last_state = []

        for step in range(self.step):
            xt = x

            for i in range(self.num_encoder):
                name = 'encoder{}'.format(i)
                x = getattr(self, name)(x)

            for i in range(self.num_encoder, self.num_encoder + self.num_convlstm):
                name = 'convlstm{}'.format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state=self.initial_state[i - self.num_encoder]
                    )
                    internal_state.append((h, c))

                (h, c) = internal_state[i - self.num_encoder]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i - self.num_encoder] = (x, new_c)

            x = self.pixelshuffle(x)
            x = self.output_layer(x)
            x = self.output_layer(x)
            x = self.output_layer(x)

            x = xt + self.dt * x

            if step == (self.step - 2):
                second_last_state = internal_state.copy()

            if step in self.effective_step:
                outputs.append(x)

        return outputs, second_last_state

# ============================================================
# 5) Operators & Loss
# ============================================================
class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super().__init__()
        self.resol = resol
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size,
                                1, padding=self.padding, bias=False)

        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol

class LossGenerator(nn.Module):
    def __init__(self, u_train, idx, dx, alpha, D, v):
        super().__init__()

        self.laplace = Conv2dDerivative(
            DerFilter=lapl_op,
            resol=(dx ** 2),
            kernel_size=5,
            name='laplace_operator'
        ).to(device)

        self.dx = Conv2dDerivative(
            DerFilter=partial_x,
            resol=(dx * 1),
            kernel_size=5,
            name='dx_operator'
        ).to(device)

        self.dy = Conv2dDerivative(
            DerFilter=partial_y,
            resol=(dx * 1),
            kernel_size=5,
            name='dy_operator'
        ).to(device)

        self.alpha = alpha
        self.D = D
        self.v = v
        self.idx = idx
        self.u = torch.tensor(u_train).float().to(device)

    def get_phy_Loss(self, output):
        u_x = self.dx(output[:, 0:1, :, :])
        u_y = self.dy(output[:, 0:1, :, :])

        u = output[:, 0:1, :, :]
        lent = u.shape[0]
        lenx = u.shape[2]
        leny = u.shape[3]

        u = u.reshape(lent, 1, lenx * leny)

        dt = (10.0 / 100)
        n = 9
        w = [1]
        for j in range(1, n + 1):
            w.append((j + 1) ** (1 - self.alpha) - j ** (1 - self.alpha))

        b = torch.zeros(1, 1, lenx * leny).to(device)
        for i in range(1, n + 2):
            c = torch.zeros(1, 1, lenx * leny).to(device)
            for k in range(1, i):
                c = c + (w[i - k - 1] - w[i - k]) * u[k, :, :]

            a = w[0] * u[i, :, :] - c - w[i - 1] * u[0, :, :]
            b = torch.cat([b, a], dim=0)

        D_alphau = (dt ** (-self.alpha) / torch.lgamma(2 - self.alpha).exp()) * b[:, :, :]
        D_alphau = D_alphau.reshape(lent, 1, lenx, leny)

        u = output[:, 0:1, :, :]

        f1 = np.load('./model/f1.npy')
        f1 = torch.tensor(f1).to(device)
        f_u = D_alphau + self.v * (u_x + u_y) + self.D * u - f1

        output_transposed = output.permute(1, 2, 3, 0)
        flattened = output_transposed[0, :, :, :].reshape(-1)
        u_pred = flattened[idx]
        f_data = self.u - u_pred

        fu_x1 = u[1:, :, :, 0]
        fu_x2 = u[1:, :, :, -1]
        fu_x3 = u[1:, :, 0, :]
        fu_x4 = u[1:, :, -1, :]
        x1 = torch.zeros(128, device='cuda')
        x2 = torch.zeros(128, device='cuda')
        x1_d = x1.unsqueeze(0).unsqueeze(0).repeat(fu_x1.shape[0], 1, 1)
        x2_d = x2.unsqueeze(0).unsqueeze(0).repeat(fu_x2.shape[0], 1, 1)
        f1_bc = ((fu_x1 - x1_d) ** 2 + (fu_x2 - x2_d) ** 2 + (fu_x3 - x1_d) ** 2 + (fu_x4 - x2_d) ** 2).mean() / (
            fu_x1.shape[-1])

        return f_u, f_data, f1_bc

def compute_loss(output, loss_func):
    mse_loss = nn.MSELoss()
    f_u, f_v, f1_bc = loss_func.get_phy_Loss(output)
    l1 = mse_loss(f_u, torch.zeros_like(f_u).to(device))
    l2 = mse_loss(f_v, torch.zeros_like(f_v).to(device))
    l3 = mse_loss(f1_bc, torch.zeros_like(f_v).to(device))
    loss = l1 + 10 * l2 + l3
    return loss

# ============================================================
# 6) Checkpoint IO
# ============================================================
def ckpt_save(model, optimizer, scheduler, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }, save_path)

def ckpt_load_optional(model, optimizer, scheduler, load_path):
    if load_path is None or (not os.path.isfile(load_path)):
        return model, optimizer, scheduler, False

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')
    return model, optimizer, scheduler, True

# ============================================================
# 7) Train / Infer
# ============================================================
def run_train(u_train, idx, model, input0, initial_state,
              n_iters, time_batch_size, learning_rate, dt, dx,
              save_path, pre_ckpt_path, num_time_batch):
    train_loss_list = []
    prev_output = []
    fanyan = []
    best_loss = 1e4

    alpha = torch.tensor(0.7, requires_grad=True).to(device)
    alpha = torch.nn.Parameter(alpha)
    D = torch.tensor(1.3, requires_grad=True).to(device)
    D = torch.nn.Parameter(D)
    v = torch.tensor(1.5, requires_grad=True).to(device)
    v = torch.nn.Parameter(v)

    optimizer = optim.Adam(list(model.parameters()) + [alpha, D, v], lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.97)

    model, optimizer, scheduler, loaded = ckpt_load_optional(model, optimizer, scheduler, pre_ckpt_path)
    if not loaded:
        print(f"[INFO] pre checkpoint not found: {pre_ckpt_path} -> train from scratch")

    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    loss_func = LossGenerator(u_train, idx, dx, alpha, D, v)

    for epoch in range(n_iters):
        optimizer.zero_grad()
        batch_loss = 0.0

        for time_batch_id in range(num_time_batch):
            if time_batch_id == 0:
                hidden_state = initial_state
                u0 = input0
            else:
                hidden_state = state_detached
                u0 = prev_output[-2:-1].detach()

            output, second_last_state = model(hidden_state, u0)
            output = torch.cat(tuple(output), dim=0)
            output = torch.cat((u0.to(device), output), dim=0)

            loss = compute_loss(output, loss_func)
            loss.backward(retain_graph=True)
            batch_loss += loss.item()

            prev_output = output
            state_detached = []
            for i in range(len(second_last_state)):
                (h, c) = second_last_state[i]
                state_detached.append((h.detach(), c.detach()))

        optimizer.step()
        scheduler.step()

        print('[%d/%d %d%%] loss: %.10f' % ((epoch + 1), n_iters, ((epoch + 1) / n_iters * 100.0), batch_loss))
        train_loss_list.append(batch_loss)

        fanyan.append({'alpha': alpha.item(), 'D': D.item(), 'v': v.item()})

        if batch_loss < best_loss:
            ckpt_save(model, optimizer, scheduler, save_path)
            best_loss = batch_loss

    return train_loss_list, fanyan

def run_infer(model, initial_state, input0, ckpt_path):
    model, _, _, loaded = ckpt_load_optional(model, optimizer=None, scheduler=None, load_path=ckpt_path)
    if not loaded:
        print(f"[INFO] infer checkpoint not found: {ckpt_path} -> use current/random weights")

    output, _ = model(initial_state, input0)
    output = torch.cat(tuple(output), dim=0)
    output = torch.cat((input0.to(device), output), dim=0)
    return output

# ============================================================
# 8) Post-process & Metrics
# ============================================================
def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
    xmin, xmax, ymin, ymax = axis_lim

    x = np.linspace(xmin, xmax, 128)
    x_star, y_star = np.meshgrid(x, x)

    u_star = true[num, 0, :, :]
    v_star = true[num, 1, :, :]

    u_pred = output[num, 0, :, :]
    v_pred = output[num, 1, :, :]

    if torch.is_tensor(u_pred):
        u_pred = u_pred.detach().cpu().numpy()
    if torch.is_tensor(v_pred):
        v_pred = v_pred.detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    uuuv = np.abs(v_star - v_pred)

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

# ============================================================
# 9) Main
# ============================================================
if __name__ == "__main__":
    u0 = torch.zeros(1, 2, 128, 128)
    input0 = torch.tensor(u0, dtype=torch.float32).to(device)

    num_convlstm = 1
    (h0, c0) = (torch.randn(1, 128, 16, 16), torch.randn(1, 128, 16, 16))
    initial_state = []
    for _ in range(num_convlstm):
        initial_state.append((h0, c0))

    M, N = 128, 128
    u_e = np.zeros([128, 128, 1])
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, N)
    x_star, y_star = np.meshgrid(x, y)
    for i in range(1, 11):
        Z = (i * dt) ** 4 * np.sin(4 * np.pi * x_star) * np.sin(4 * np.pi * y_star)
        Z = Z[:, :, np.newaxis]
        u_e = np.concatenate([u_e, Z], -1)
    u_e = u_e.reshape(-1)

    N_u = 20000
    idx = np.random.choice(u_e.shape[0], N_u, replace=False)

    noise_level = 0
    u_train = u_e[idx]
    signal_std = np.std(u_train)
    noise_std = noise_level * signal_std
    noise = np.random.normal(0, noise_std, u_train.shape)
    u_train = u_train + noise

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

    train_loss, fanyan = run_train(
        u_train, idx, model, input0, initial_state,
        n_iters_adam, time_batch_size, lr_adam, dt, dx,
        model_save_path, pre_model_save_path, num_time_batch
    )
    np.savetxt("train_loss.txt", train_loss)

    fanyan_array = np.array(fanyan)
    alpha_values = [entry["alpha"] for entry in fanyan_array]
    D_values = [entry["D"] for entry in fanyan_array]
    v_values = [entry["v"] for entry in fanyan_array]

    np.savetxt("alpha_values.txt", alpha_values)
    np.savetxt("D_values.txt", D_values)
    np.savetxt("v_values.txt", v_values)

    plt.figure(figsize=(8, 4))
    plt.plot(alpha_values, marker='o', linestyle='-', color='b')
    plt.title('alpha')
    plt.xlabel('iter')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(D_values, marker='o', linestyle='-', color='b')
    plt.title('D')
    plt.xlabel('iter')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(v_values, marker='o', linestyle='-', color='b')
    plt.title('v')
    plt.xlabel('iter')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    time_batch_size_load = 9
    steps_load = time_batch_size_load + 1
    effective_step_load = list(range(0, steps_load))

    model_inf = FPICRNNs(
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

    output = run_infer(model_inf, initial_state, input0, model_save_path)

    # load truth & post-process
    u_e = np.load('./model/u_e.npy')
    v_e = np.load('./model/u_e.npy')
    truth = np.concatenate((u_e, v_e), axis=1)

    ten_true = []
    ten_pred = []
    output_t = output[:, :, :, :]

    for i in range(0, 16):
        u_star, u_pred, v_star, v_pred = post_process(
            output_t, truth, [0, 1, 0, 1], [0, 1, 0, 1], num=i, fig_save_path=fig_save_path
        )
        ten_true.append([u_star])
        ten_pred.append([u_pred])

    a_MSE = []
    for i in range(1, 17):
        a_MSE_value = frobenius_norm(np.array(ten_pred[:i]) - np.array(ten_true[:i])) / np.sqrt((128 * 128 * (i + 1)))
        a_MSE = np.append(a_MSE, a_MSE_value)

    RMSE_u = []
    for i in range(1, 17):
        RMSE_u_value = np.sqrt(np.mean((np.array(ten_pred[:i]) - np.array(ten_true[:i])) ** 2))
        RMSE_u = np.append(RMSE_u, RMSE_u_value)

    np.savetxt('a_MSE.txt', a_MSE)
    np.savetxt('RMSE_u.txt', RMSE_u)

