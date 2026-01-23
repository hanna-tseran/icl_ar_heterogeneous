from datetime import datetime
import multiprocessing as mp
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import torch
import torch.nn as nn

np.set_printoptions(threshold=sys.maxsize)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# Experiment parameters
ACTIVATION_FUNCTION = 'gelu' #'gelu' # 'relu' #'tanh'
D = 10
EPOCHS_NUM = 8000
M = 10000
n = D * 4
N = 100
REPORT_EPOCH = 100
RUNS_NUM = 1
TASKS_NUM = 30000

TRAIN = .8
VAL = .1
# TEST = .1

learning_rate = 1/(1.1 * M)


# Data generation
class ARProcess():
    def __init__(self):
        self.A = np.random.normal(size=(D, D))
        self.A *= .1 / np.max(np.abs(np.linalg.eigvals(self.A)))
        eigvals = np.linalg.eigvals(self.A)

    def get_x_1(self):
        sigma = linalg.solve_discrete_lyapunov(self.A, np.eye(D))
        x_1 = np.random.multivariate_normal(mean=np.zeros(D), cov=sigma)
        return x_1

    def sample(self, x_i):
        return self.A @ x_i + np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D))

# Data preprocessing
def embed(task):
    return np.block([
        [np.asarray(task[:-1]).T, np.eye(D)],
        [np.asarray(task[1:]).T, np.zeros(shape=(D, D))],
        [np.zeros(shape=(D, N-1)), np.diag(task[-1])],
        [np.zeros(shape=(D, N-1)), np.eye(D)]
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
        V = torch.einsum('ij,jkb->ikb', self.W_V, E)
        K = torch.einsum('ij,jkb->ikb', self.W_K, E)
        # print(f' E[:, -D:] shape:{E[:, -D:].shape}')
        Q = torch.einsum('ij,jkb->ikb', self.W_Q, E[:, -D:])
        KQ = torch.einsum('ijb,ikb->jkb', K, Q)
        VKQ = torch.einsum('ijb,jkb->ikb', V, KQ)
        OVKQ = torch.einsum('ij,jkb->ikb', self.W_O, VKQ)
        OVKQ = OVKQ.reshape(shape=(OVKQ.shape[0] * OVKQ.shape[1], OVKQ.shape[2]))
        return OVKQ


class Transformer(nn.Module):
    def __init__(self, attention_head, M):
        super().__init__()
        
        self.attention_head = attention_head
        self.n = n
        self.M = M
        self.beta = torch.tensor(np.random.normal(size=(self.M, self.n * D)), dtype=torch.double)

        self.gamma = torch.nn.Parameter(torch.tensor(np.random.normal(size=(D, self.M), scale=np.sqrt(2/self.M)),
            dtype=torch.double))
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
        inner = self.activation_f(torch.einsum('ij,jb->ib', self.beta, f_att))
        out = (torch.einsum('ij,jb->ib', self.gamma, inner))
        return out


# Plotting
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
        f'figures/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{filename}'
        + f'_{job_id}_D={D}_EPOCHS_NUM={EPOCHS_NUM}_M={M}_N={N}'
        + f'_TASKS_NUM={TASKS_NUM}_RUNS_NUM={RUNS_NUM}_ACT_F={ACTIVATION_FUNCTION}_learning_rate={learning_rate}.png',
        bbox_inches='tight')


# Main training loop
def train():
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

    train_x = torch.tensor(embeddings[:int(TRAIN * TASKS_NUM)], dtype=torch.double).permute(1, 2, 0)
    val_x = torch.tensor(embeddings[int(TRAIN * TASKS_NUM):int(TRAIN * TASKS_NUM) + int(VAL * TASKS_NUM)],
        dtype=torch.double).permute(1, 2, 0)
    test_x = torch.tensor(embeddings[int(TRAIN * TASKS_NUM) + int(VAL * TASKS_NUM):],
        dtype=torch.double).permute(1, 2, 0)

    train_y = torch.tensor(np.asarray(Y[:int(TRAIN * TASKS_NUM)]).T, dtype=torch.double)
    val_y = torch.tensor(np.asarray(Y[int(TRAIN * TASKS_NUM):int(TRAIN * TASKS_NUM) + int(VAL * TASKS_NUM)]).T,
        dtype=torch.double)
    test_y = torch.tensor(np.asarray(Y[int(TRAIN * TASKS_NUM) + int(VAL * TASKS_NUM):]).T, dtype=torch.double)

    # Model
    W_K = np.eye(n)
    W_Q1d = np.zeros((4,4))
    W_Q1d[0, 0] = 1
    W_Q = np.kron(W_Q1d, np.eye(D))

    W_O1d = np.zeros((4,4))
    W_O1d[0, 0] = 1 / (N - 1)
    W_O1d[1, 1] = 1 / (N - 1)
    W_O1d[2, 2] = 1
    W_O = np.kron(W_O1d, np.eye(D))

    W_V1d = np.zeros((4,4))
    W_V1d[0, 0] = 1
    W_V1d[1, 1] = 1
    W_V1d[2, 2] = 1
    W_V1d[0, 3] = -1
    W_V = np.kron(W_V1d, np.eye(D))

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
        + f' D: {D}'
        + f' M: {M}'
        + f' N: {N}'
        + f' ngpus: {ngpus}'
        + f' TASKS_NUM: {TASKS_NUM}'
        + f' learning_rate: {learning_rate}'
        + f' ACTIVATION_FUNCTION: {ACTIVATION_FUNCTION}'
        )

    print(f'Running the experiment number {gpu_id + 1}.', flush=True)
    train_loss_arr, val_loss_arr, test_loss = train()
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
