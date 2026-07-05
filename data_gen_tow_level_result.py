import os
import h5py
import numpy as np
import torch

from RK4_elvo import RK4_bloch
from visualization import visualize_bloch_prediction
from data_io_processing_two_level import collect_sequences
from auxiliary_calc_two_level import bessel_pulse_single, bloch_from_rho_, rho_from_bloch_, get_fidelity_list

SEED = 42
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
draw_flag = True

N_each = 40
N_sample = N_each * 3
batch_size = N_sample
T = 5  #
steps = 200
tn = T * steps
dtype = torch.float32
save_path = f"data/two_level_result.h5"

def create_data():
    os.makedirs(os.path.normpath(save_path).split(os.sep)[0], exist_ok=True)

    cycle_list = np.random.uniform(5, 5, size=N_each*3)
    Ct_list = np.array([bessel_pulse_single(T, steps=steps, cycle=cycle) for cycle in cycle_list])

    Gm_1 = np.linspace(0.01, 0.05, N_each)
    gm_1 = np.full(N_each, 4.0)
    ther_1 = np.full(N_each, 10)

    Gm_2 = np.full(N_each, 0.03)
    gm_2 = np.linspace(1.5, 25, N_each)
    ther_2 = np.full(N_each, 10)

    Gm_3 = np.full(N_each, 0.03)
    gm_3 = np.full(N_each, 4.0)
    ther_3 = np.linspace(5, 15, N_each)

    Gm = np.concatenate([Gm_1, Gm_2, Gm_3])
    gm = np.concatenate([gm_1, gm_2, gm_3])
    ther = np.concatenate([ther_1, ther_2, ther_3])

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

def plot_example(idx=0, data_path=save_path):
    bloch_seqs, St_seqs, Ct_seqs, params_batch, H_seqs = collect_sequences(data_path)
    tn_plus1 = St_seqs.shape[1]
    times = np.linspace(0, T, tn_plus1)
    St = St_seqs[idx, :, 0]  # (tn+1,)
    Ct = Ct_seqs[idx, :, 0]  # (tn+1,)
    bloch_seq = bloch_seqs[idx, :, :]
    params = params_batch[idx]
    rhos_seq = rho_from_bloch_(torch.tensor(bloch_seq, device=device)).cpu().numpy()
    fidelity = get_fidelity_list(H_seqs[idx], rhos_seq)
    visualize_bloch_prediction(times, St, Ct=Ct, true_bloch=bloch_seq, param=params[0], idx=idx, f_true=fidelity)


if __name__ == '__main__':
    create_data()
    # for i in range(10):
    #     plot_example(idx=i)
