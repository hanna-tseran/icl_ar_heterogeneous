from datetime import datetime
import multiprocessing as mp
import os
import random
import sys

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn

np.set_printoptions(threshold=sys.maxsize)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# Experiment parameters
ACTIVATION_FUNCTION = 'gelu' #'gelu' # 'relu' #'tanh'
EPOCHS_NUM = 1000
M = 10000#0
N = 100
REPORT_EPOCH = 100
RUNS_NUM = 1
TASKS_NUM = 100000

TRAIN = .8
VAL = .1
# TEST = .1

learning_rate = 1/(1.2 * M)


# Data generation
class ARProcess():
    def __init__(self):
        self.alpha = np.random.uniform(-1 + .1, 1 - .1)

    def get_x_1(self):
        return np.random.normal(scale=np.sqrt(1 - self.alpha**2))

    def sample(self, x_i):
        return self.alpha * x_i + np.random.normal()


# Data preprocessing
def embed(task):
    return np.asarray([
        task[:-1] + [1],
        task[1:] + [0],
        np.zeros(N - 1).tolist() + [task[-1]],
        np.zeros(N - 1).tolist() + [1]
    ])


# Loss
def mse_loss(output, target):
    return torch.mean((output - target)**2)/2


# Model
class AttentionHead(nn.Module):
    def __init__(self, W_K, W_Q, W_V, W_O):
        super().__init__()
        
        self.W_K = torch.tensor(W_K, dtype=torch.double)
        self.W_Q = torch.tensor(W_Q, dtype=torch.double)
        self.W_V = torch.tensor(W_V, dtype=torch.double)
        self.W_O = torch.tensor(W_O, dtype=torch.double)

    def forward(self, E):
        V = torch.einsum('ij,bjk->bik', self.W_V, E)
        K = torch.einsum('ij,bjk->bik', self.W_K, E)
        Q = torch.einsum('ij,bj->bi', self.W_Q, E[:,:, -1])
        KQ = torch.einsum('bij,bi->bj', K, Q)
        VKQ = torch.einsum('bij,bj->bi', V, KQ)
        OVKQ = torch.einsum('ij,bj->bi', self.W_O, VKQ)
        return OVKQ


class Transformer(nn.Module):
    def __init__(self, attention_head, M):
        super().__init__()
        
        self.attention_head = attention_head
        self.n = 4
        self.M = M
        self.beta = torch.tensor(np.random.normal(size=(self.M, self.n)), dtype=torch.double)
        self.gamma = torch.nn.Parameter(torch.zeros(M, dtype=torch.double))

        if ACTIVATION_FUNCTION == 'tanh':
            self.activation_f = torch.tanh
        elif ACTIVATION_FUNCTION == 'relu':
            self.activation_f = torch.relu
        elif ACTIVATION_FUNCTION == 'gelu':
            self.activation_f = nn.functional.gelu
        else:
            print('ERROR: Non existent activation function')

    def forward(self, X):        
        f_att = self.attention_head(X)
        inner = self.activation_f(torch.einsum('ij,bj->bi', self.beta, f_att))
        out = (torch.einsum('j,bj->b', self.gamma, inner))
        return out


def plot_losses(train_losses, val_losses, epoch_arr, xlabel='Epoch', ylabel='Loss', filename='test', job_id=0,
        log=False):

    irr_error = .5 * np.ones(len(epoch_arr))

    if log:
        train_losses = np.log(train_losses)
        val_losses = np.log(val_losses)
        ylabel = 'Log Loss'
        irr_error = np.log(irr_error)

    train_loss_mean = np.mean(train_losses, axis=0)
    train_loss_std = np.std(train_losses, axis=0)
    val_loss_mean = np.mean(val_losses, axis=0)
    val_loss_std = np.std(val_losses, axis=0)
    print(f'train_loss_mean log={log}:\n{train_loss_mean}')
    print(f'train_loss_std log={log}:\n{train_loss_std}')
    print(f'val_loss_mean log={log}:\n{val_loss_mean}')
    print(f'val_loss_std log={log}:\n{val_loss_std}')
    print(f'final train loss mean log={log}: {train_loss_mean[-1]}')
    print(f'final val loss mean log={log}: {val_loss_mean[-1]}')

    fig = plt.figure(figsize=(20, 9), dpi=100)
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=44)
    ax.tick_params(axis='both', which='minor', labelsize=44)

    plt.plot(epoch_arr, train_loss_mean, label="Train loss", color='#0072B2', linewidth=5)
    plt.fill_between(epoch_arr, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, color='#0072B2',
        alpha=0.075)

    plt.plot(epoch_arr, val_loss_mean, label="Validation loss", color='#D55E00', linewidth=5)
    plt.fill_between(epoch_arr, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, color='#D55E00',
        alpha=0.075)

    plt.plot(epoch_arr, irr_error, label="Irreducible error", color='k', linewidth=4, linestyle='dashed')

    plt.ylabel(ylabel, size=50)
    if not log:
        plt.ylim(bottom=0.4)

    plt.xlabel("Epochs", size=50)
    if EPOCHS_NUM == 20000:
        plt.xticks([0, 5000, 10000, 15000, 20000])
    plt.tight_layout()
    plt.legend(loc='upper right', fontsize=40, title_fontsize=50)
    plt.savefig(
        f'figures/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{filename}_{job_id}_EPOCHS_NUM={EPOCHS_NUM}_M={M}_N={N}'
        + f'_TASKS_NUM={TASKS_NUM}_RUNS_NUM={RUNS_NUM}_ACT_F={ACTIVATION_FUNCTION}_learning_rate={learning_rate}.png',
        bbox_inches='tight')

def plot_attention(transformer, E, job_id=0):
    K = np.einsum('ij,jk->ik', transformer.attention_head.W_K.detach().numpy(), E.detach().numpy())
    Q = np.einsum('ij,jk->ik', transformer.attention_head.W_Q.detach().numpy(), E.detach().numpy())
    print(f'K shape:{K.shape}')
    print(f'Q shape:{Q.shape}')

    scores = np.einsum('ij,ik->jk', K, Q)
    print(f'scores shape:{scores.shape}')

    plt.figure(figsize=(5, 4))
    plt.imshow(scores.T, aspect='auto', cmap='inferno')
    plt.colorbar()
    plt.title("Self-attention (N×N)")
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.tight_layout()

    plt.savefig(
        f'figures/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{job_id}_attention_EPOCHS_NUM={EPOCHS_NUM}_M={M}_N={N}'
        + f'_TASKS_NUM={TASKS_NUM}_RUNS_NUM={RUNS_NUM}_ACT_F={ACTIVATION_FUNCTION}_learning_rate={learning_rate}.png',
        bbox_inches='tight')


def plot_function(transformer, add_name='', job_id=0):
    def phi(v, c, x, dummy):
        return (c / v) * x

    def net(X):
        inner = transformer.activation_f(
            torch.from_numpy(np.einsum('ij,jb->ib', transformer.beta.detach().numpy(), X))).detach().numpy()
        out = (np.einsum('j,jb->b', transformer.gamma.detach().numpy(), inner))
        return out

    def net2d(X):
        inner = transformer.activation_f(
            torch.from_numpy(np.einsum('ij,jbc->ibc', transformer.beta.detach().numpy(), X))).detach().numpy()
        out = (np.einsum('j,jbc->bc', transformer.gamma.detach().numpy(), inner))
        return out

    x = np.linspace(-1, 1, 100)
    v = 1.
    c = .1
    phi_out = phi(np.full_like(x, v), np.full_like(x, c), x, np.full_like(x, 0.))
    mlp_out = net(np.stack([np.full_like(x, v), np.full_like(x, c), x, np.full_like(x, 0.)], axis=0))
    print(phi_out)
    print(mlp_out)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=44)
    ax.tick_params(axis='both', which='minor', labelsize=44)

    plt.plot(x, phi_out, label="$\phi$", color='k', linewidth=2, linestyle='dashed')
    plt.plot(x, mlp_out, label="$f_{MLP}$", color='#0072B2', linewidth=3)

    plt.xlabel("$x_N$", size=50)
    plt.ylabel("Function output", size=50)
    plt.legend(loc='upper left', fontsize=40, title_fontsize=50)
    plt.tight_layout()
    plt.savefig(
        f'figures/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{job_id}_function_{add_name}_EPOCHS_NUM={EPOCHS_NUM}_M={M}_N={N}'
        + f'_TASKS_NUM={TASKS_NUM}_RUNS_NUM={RUNS_NUM}_ACT_F={ACTIVATION_FUNCTION}_learning_rate={learning_rate}.png',
        bbox_inches='tight')

    #######
    fig = plt.figure(figsize=(20, 18))
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(axis='x', which='major', labelsize=44, pad=20)
    ax.tick_params(axis='x', which='minor', labelsize=44, pad=20)

    ax.tick_params(axis='y', which='major', labelsize=44, pad=20)
    ax.tick_params(axis='y', which='minor', labelsize=44, pad=20)

    ax.tick_params(axis='z', which='major', labelsize=44, pad=55)
    ax.tick_params(axis='z', which='minor', labelsize=44, pad=55)

    v = np.linspace(1., 2., 100)
    c = np.linspace(-1., 1., 100)
    x = .1
    v, c = np.meshgrid(v, c)

    net_out = net2d(np.stack([v, c, np.full_like(v, x), np.full_like(v, 0.)], axis=0))
    surf = ax.plot_surface(v, c, net_out, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0, antialiased=False, alpha=.6)

    phi_out = phi(v, c, np.full_like(v, x), np.full_like(v, 0.))
    surf = ax.plot_wireframe(v, c, phi_out, color="k", linewidth=2)

    ax.set_xlabel('$v_N$', size=50, labelpad=50)
    ax.set_ylabel('$c_N$', size=50, labelpad=60)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Function output', rotation=90, size=50, labelpad=110)


    plt.yticks([-1., 0., 1.])

    plt.tight_layout()
    ax.view_init(elev=20, azim=-105)

    plt.savefig(
        f'figures/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{job_id}_function2d_{add_name}_EPOCHS_NUM={EPOCHS_NUM}_M={M}_N={N}'
        + f'_TASKS_NUM={TASKS_NUM}_RUNS_NUM={RUNS_NUM}_ACT_F={ACTIVATION_FUNCTION}_learning_rate={learning_rate}.png')

    ####### x
    fig = plt.figure(figsize=(20, 18))
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(axis='x', which='major', labelsize=44, pad=20)
    ax.tick_params(axis='x', which='minor', labelsize=44, pad=20)

    ax.tick_params(axis='y', which='major', labelsize=44, pad=20)
    ax.tick_params(axis='y', which='minor', labelsize=44, pad=20)

    ax.tick_params(axis='z', which='major', labelsize=44, pad=40)
    ax.tick_params(axis='z', which='minor', labelsize=44, pad=40)

    v = 1.5
    c = np.linspace(-1., 1., 100)
    x = np.linspace(-1., 1., 100)
    c, x = np.meshgrid(c, x)

    net_out = net2d(np.stack([np.full_like(c, v), c, x, np.full_like(c, 0.)], axis=0))
    surf = ax.plot_surface(c, x, net_out, rstride=1, cstride=1, cmap='viridis',
                           linewidth=0, antialiased=False, alpha=.6)

    phi_out = phi(np.full_like(c, v), c, x, np.full_like(c, 0.))
    surf = ax.plot_wireframe(c, x, phi_out, color="k", linewidth=2)

    ax.set_xlabel('$c_N$', size=50, labelpad=60)
    ax.set_ylabel('$x_N$', size=50, labelpad=60)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Function output', rotation=90, size=50, labelpad=100)


    plt.xticks([-1., -.5, 0., .5, 1.])
    plt.yticks([-1., 0., 1.])

    plt.tight_layout()
    ax.view_init(elev=10, azim=110)

    plt.savefig(
        f'figures/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{job_id}_function2dx_{add_name}_EPOCHS_NUM={EPOCHS_NUM}_M={M}_N={N}'
        + f'_TASKS_NUM={TASKS_NUM}_RUNS_NUM={RUNS_NUM}_ACT_F={ACTIVATION_FUNCTION}_learning_rate={learning_rate}.png')


# Main training loop
def train(gpu_id=0):
    # Data generation
    X = []
    Y = []
    for _ in range(TASKS_NUM):
        ar_process = ARProcess()
        task = [ar_process.get_x_1()]
        for i in range(1, N):
            task.append(ar_process.sample(task[i-1]))
        X.append(task)
        Y.append(ar_process.sample(task[-1]))


    # Data preprocessing
    embeddings = np.asarray([embed(task) for task in X])

    train_x = torch.tensor(embeddings[:int(TRAIN * TASKS_NUM)], dtype=torch.double)#.permute(1, 2, 0)
    val_x = torch.tensor(embeddings[int(TRAIN * TASKS_NUM):int(TRAIN * TASKS_NUM) + int(VAL * TASKS_NUM)],
        dtype=torch.double)
    test_x = torch.tensor(embeddings[int(TRAIN * TASKS_NUM) + int(VAL * TASKS_NUM):],
        dtype=torch.double)

    train_y = torch.tensor(Y[:int(TRAIN * TASKS_NUM)], dtype=torch.double)
    val_y = torch.tensor(Y[int(TRAIN * TASKS_NUM):int(TRAIN * TASKS_NUM) + int(VAL * TASKS_NUM)], dtype=torch.double)
    test_y = torch.tensor(Y[int(TRAIN * TASKS_NUM) + int(VAL * TASKS_NUM):], dtype=torch.double)

    # Model
    W_K = np.identity(4)
    W_Q = np.zeros((4,4))
    W_Q[0, 0] = 1

    W_O = np.zeros((4,4))
    W_O[0, 0] = 1 / (N - 1)
    W_O[1, 1] = 1 / (N - 1)
    W_O[2, 2] = 1

    W_V = np.zeros((4,4))
    W_V[0, 0] = 1
    W_V[1, 1] = 1
    W_V[2, 2] = 1
    W_V[0, 3] = -1

    attention_head = AttentionHead(
        W_K=W_K,
        W_Q=W_Q,
        W_V=W_V,
        W_O=W_O
    )
    transformer = Transformer(
        attention_head=attention_head,
        M=M
    )

    # Training
    model = transformer
    loss_fn = mse_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if gpu_id == 0:
        plot_function(transformer=model, add_name='before', job_id=os.environ.get("JOB_ID"))

    train_loss_arr = []
    val_loss_arr = []
    for epoch_id in range(EPOCHS_NUM):
        model.train()
        optimizer.zero_grad()
        output = model(train_x)
        loss = loss_fn(output, train_y)
        loss.backward()
        optimizer.step()
        
        train_loss_arr.append(loss.detach().item())
        if not (epoch_id + 1) % REPORT_EPOCH:
            print(f'Epoch: {epoch_id + 1}', flush=True)
            print(f'Loss: {train_loss_arr[-1]}', flush=True)

        # Record validation loss
        model.eval()
        with torch.no_grad():
            val_loss_arr.append(loss_fn(model(val_x), val_y).item())
            
    train_loss_arr.append(loss_fn(model(train_x), train_y).item())
    val_loss_arr.append(loss_fn(model(val_x), val_y).item())
    test_loss = loss_fn(model(test_x), test_y).item()
    print(f'train_loss_arr:\n{train_loss_arr}')
    print(f'val_loss_arr:\n{val_loss_arr}')
    print(f'test_loss:\n{test_loss}')

    if gpu_id == 0:
        plot_function(transformer=model, add_name='after', job_id=os.environ.get("JOB_ID"))

    return train_loss_arr, val_loss_arr, test_loss


def run_experiment(seed, gpu_id, ngpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.init()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print('Starting experiments', flush=True)
    print('Experiment setup:\n'
        + f'EPOCHS_NUM: {EPOCHS_NUM}'
        + f' REPORT_EPOCH: {REPORT_EPOCH}'
        + f' M: {M}'
        + f' N: {N}'
        + f' ngpus: {ngpus}'
        + f' TASKS_NUM: {TASKS_NUM}'
        + f' learning_rate: {learning_rate}'
        + f' ACTIVATION_FUNCTION: {ACTIVATION_FUNCTION}'
        )

    print(f'Running the experiment number {gpu_id + 1}.', flush=True)
    train_loss_arr, val_loss_arr, test_loss = train(gpu_id)
    return {"train_loss_arr": train_loss_arr, "val_loss_arr": val_loss_arr, "test_loss": test_loss}


if __name__ == "__main__":
    # train()
    experiment_start_time = datetime.now()
    mp.set_start_method("spawn", force=True)
    ngpus = torch.cuda.device_count()
    ngpus_arr = [torch.cuda.device_count() for _ in range(ngpus)]
    gpu_ids = list(range(ngpus))
    seeds = list(range(ngpus))

    job_id = os.environ.get("JOB_ID")
    print("This job's ID is:", job_id)

    with mp.Pool(processes=ngpus) as pool:
        results = pool.starmap(run_experiment, zip(seeds, gpu_ids, ngpus_arr))
    print(results)

    train_losses = [r['train_loss_arr'] for r in results]
    val_losses = [r['val_loss_arr'] for r in results]
    test_losses = [r['test_loss'] for r in results]
    plot_losses(train_losses, val_losses, list(range(EPOCHS_NUM + 1)), job_id=job_id, log=False, filename='standard')
    plot_losses(train_losses, val_losses, list(range(EPOCHS_NUM + 1)), job_id=job_id, log=True, filename='log')
    print(f'Test loss mean: {np.mean(test_losses)} and std: {np.std(test_losses)}')

    total_time = datetime.now() - experiment_start_time
    hours = int(total_time.seconds / 3600)
    minutes = int(total_time.seconds / 60 - hours * 60)
    seconds = int(total_time.seconds - hours * 3600 - minutes * 60)
    print(f'Full experiment took: {hours}h {minutes}min {int(seconds)}s')
