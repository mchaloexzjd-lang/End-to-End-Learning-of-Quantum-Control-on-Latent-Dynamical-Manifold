import os
import h5py
import numpy as np
import torch

from RK4_elvo import RK4_bloch
from auxiliary_calc_two_level import bessel_pulse_single, bloch_from_rho_

SEED = 42
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_sample = 12800
batch_size = N_sample

T = 5  #
steps = 200
tn = T * steps
dtype = torch.float32
save_path = f"data/tow_level_Train.h5"

def lhs_sampling(param_ranges, N):
    D = len(param_ranges)
    result = np.zeros((N, D))
    for i, (low, high) in enumerate(param_ranges):
        cut = np.linspace(0, 1, N + 1)
        u = np.random.rand(N) * (cut[1] - cut[0])
        samples = cut[:N] + u
        np.random.shuffle(samples)
        result[:, i] = low + samples * (high - low)
    return result


def create_data():
    os.makedirs(os.path.normpath(save_path).split(os.sep)[0], exist_ok=True)

    cycle_list = np.random.uniform(5, 5, size=N_sample)
    Ct_list = np.array([bessel_pulse_single(T, steps=steps, cycle=cycle) for cycle in cycle_list])

    param_ranges = [(0.005, 0.055), (1.5, 55), (4.5, 15.5)]
    samples = lhs_sampling(param_ranges, N=N_sample)
    Gm = samples[:, 0]
    gm = samples[:, 1]
    ther = samples[:, 2]

    St_list = np.linspace(0, 1, tn + 1)[None, :]
    St_list = np.broadcast_to(St_list, (N_sample, St_list.shape[1]))
    Gm_np = np.full((N_sample, tn + 1, 1), Gm[:, None, None])
    gm_np = np.full((N_sample, tn + 1, 1), gm[:, None, None])
    ther_np = np.full((N_sample, tn + 1, 1), ther[:, None, None])

    param = np.concatenate([Gm_np, gm_np, ther_np], axis=-1)
    all_bloch_list = []
    all_hs_list = []

    for i in range(0, N_sample, batch_size):
        St_batch = torch.tensor(St_list[i:i + batch_size], dtype=dtype, device=device).unsqueeze(-1)
        Ct_batch = torch.tensor(Ct_list[i:i + batch_size], dtype=dtype, device=device).unsqueeze(-1)
        param_batch = torch.tensor(param[i:i + batch_size], dtype=dtype, device=device)

        rho_seq_batch, hs_list = RK4_bloch(St_batch, Ct_batch, param_batch, T, tn, 1)
        all_bloch_list.append(bloch_from_rho_(rho_seq_batch).cpu().numpy())
        all_hs_list.append(hs_list.cpu().numpy())
    all_bloch_list = np.concatenate(all_bloch_list, axis=0)
    all_hs_list = np.concatenate(all_hs_list, axis=0)

    with h5py.File(save_path, "w") as f:
        f.create_dataset("bloch_seqs", data=all_bloch_list, compression='lzf')
        f.create_dataset("St_seqs", data=St_list[..., None], compression='lzf')
        f.create_dataset("Ct_seqs", data=Ct_list[..., None], compression='lzf')
        f.create_dataset("H_seqs", data=all_hs_list, compression='lzf')
        f.create_dataset("params", data=param, compression='lzf')
    print('data save to：', save_path)

if __name__ == '__main__':
    create_data()
