import numpy as np
import torch
from scipy.special import jn_zeros

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
I2_ = torch.eye(2, dtype=torch.complex64, device=device)
sigma_x_ = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
sigma_y_ = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
sigma_z_ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
basis_rho_ = torch.stack([sigma_x_, sigma_y_, sigma_z_])  # (3,2,2)


def dagger_(A):
    return A.transpose(-2, -1).conj()


def com(A, B):
    return A @ B - B @ A


def bloch_from_rho_(rho_seq):
    return torch.einsum('btij,kij->btk', rho_seq, basis_rho_).real  # (B, tn, 3)


def rho_from_bloch_(r):
    rho = 0.5 * (I2_ + torch.einsum('...k,kij->...ij', r.to(torch.complex64), basis_rho_))
    return rho


def get_fidelity_list(Hs, rhos):
    f = []
    for i in range(len(Hs)):
        eigenvalue, eigenvector = np.linalg.eigh(Hs[i])
        pi = eigenvector[:, 0]
        f.append(np.sqrt(np.real(pi.conj().T @ rhos[i] @ pi)))
    return f


def get_fidelity_tensor(Hs, rhos):
    eigvals, eigvecs = torch.linalg.eigh(Hs)
    pi = eigvecs[..., 0].unsqueeze(-1)
    fidelity = torch.matmul(pi.conj().transpose(-1, -2), torch.matmul(rhos, pi)).real
    fidelity = torch.clamp(fidelity, 0.0, 1.0)
    return fidelity.squeeze(-1).squeeze(-1)


def bessel_pulse_single(T=10, steps=200, cycle=4, n=2):
    x = jn_zeros(0, n + 1)[n]
    tn = T * steps
    t = np.linspace(0, T, tn + 1)
    period = T / cycle
    tau = period / 2
    omega = 2 * np.pi / period
    I = np.pi * x / tau
    pulse = I * np.sin(omega * t)
    return pulse


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    B, tn = 1024, 1000
    #
    # rho_seq = torch.rand(B, tn, 2, 2, dtype=torch.complex64, device=device)
    # start = time.time()
    # for i in range(2000):
    #     out = bloch_from_rho_(rho_seq)
    # print(time.time() - start)
    #
    # r_seq = torch.rand(B, tn, 3, dtype=torch.float32, device=device) * 2 - 1  # [-1,1] Bloch向量
    #
    # start = time.time()
    # for i in range(1000):
    #     out = rho_from_bloch_(r_seq)
    # end = time.time()
    # print(end - start)
