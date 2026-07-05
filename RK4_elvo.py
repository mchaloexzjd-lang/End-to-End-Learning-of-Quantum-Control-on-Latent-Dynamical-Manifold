import torch
from auxiliary_calc_two_level import com, dagger_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
jx = sx / 2
jz = sz / 2
jz_ = jz.view(1, 1, 2, 2)
jx_ = jx.view(1, 1, 2, 2)
jminus = torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64, device=device)
L = jminus.view(1, 2, 2)
L_dagger = dagger_(L)


def o1t(x, y, H, a1, gm):
    return a1 * L - gm * x + com(-1j * H - (L_dagger @ x + L @ y), x)


def o2t(x, y, H, a2, gm):
    return a2 * L_dagger - gm * y + com(-1j * H - (L_dagger @ x + L @ y), y)


def rhost(x, y, z, H):
    return (-1j * com(H, z)
            + com(L, torch.matmul(z, dagger_(x)))
            - com(L_dagger, torch.matmul(x, z))
            + com(L_dagger, torch.matmul(z, dagger_(y)))
            - com(L, torch.matmul(y, z)))


def RK4_bloch(St, Ct, param, T, tn, mode=0):
    Batch_size = Ct.shape[0]
    h = T / tn
    o1 = torch.zeros(Batch_size, 2, 2, dtype=torch.complex64, device=device)
    o2 = torch.zeros(Batch_size, 2, 2, dtype=torch.complex64, device=device)

    St_expand = torch.empty(Batch_size, 2 * tn + 1, 1, device=St.device, dtype=St.dtype)
    St_expand[:, 0::2] = St
    St_expand[:, 1::2] = (St[:, :-1] + St[:, 1:]) / 2
    hs_expand = (1.0 - St_expand.unsqueeze(-1)) * jz_ + St_expand.unsqueeze(-1) * jx_

    Ct_expand = torch.empty(Batch_size, 2 * tn + 1, 1, device=Ct.device, dtype=Ct.dtype)
    Ct_expand[:, 0::2] = Ct
    Ct_expand[:, 1::2] = (Ct[:, :-1] + Ct[:, 1:]) / 2

    Gm = param[..., 0]
    gm = param[..., 1]
    ther = param[..., 2]
    a1 = (Gm * ther * gm / 2 - 1j * Gm * (gm ** 2) / 2).to(torch.complex64)
    a2 = Gm * ther * gm / 2
    Hc = (1 + Ct_expand.unsqueeze(-1)) * hs_expand

    eigvals, eigvecs = torch.linalg.eigh(hs_expand[:, 0])
    rho = eigvecs[:, :, :1] @ eigvecs[:, :, :1].conj().transpose(-2, -1)
    rho_seq = torch.zeros(Batch_size, tn + 1, 2, 2, dtype=torch.complex64, device=device)
    rho_seq[:, 0] = rho

    for i in range(tn):
        a1_batch = a1[:, i, None, None]
        a2_batch = a2[:, i, None, None]
        gm_batch = gm[:, i, None, None]

        H = Hc[:, 2 * i]
        k1 = o1t(o1, o2, H, a1_batch, gm_batch)
        kk1 = o2t(o1, o2, H, a2_batch, gm_batch)
        kkk1 = rhost(o1, o2, rho, H)

        H = Hc[:, 2 * i + 1]
        k2 = o1t(o1 + h / 2 * k1, o2 + h / 2 * kk1, H, a1_batch, gm_batch)
        kk2 = o2t(o1 + h / 2 * k1, o2 + h / 2 * kk1, H, a2_batch, gm_batch)
        kkk2 = rhost(o1 + h / 2 * k1, o2 + h / 2 * kk1, rho + h / 2 * kkk1, H)

        k3 = o1t(o1 + h / 2 * k2, o2 + h / 2 * kk2, H, a1_batch, gm_batch)
        kk3 = o2t(o1 + h / 2 * k2, o2 + h / 2 * kk2, H, a2_batch, gm_batch)
        kkk3 = rhost(o1 + h / 2 * k2, o2 + h / 2 * kk2, rho + h / 2 * kkk2, H)

        H = Hc[:, 2 * i + 2]
        k4 = o1t(o1 + h * k3, o2 + h * kk3, H, a1_batch, gm_batch)
        kk4 = o2t(o1 + h * k3, o2 + h * kk3, H, a2_batch, gm_batch)
        kkk4 = rhost(o1 + h * k3, o2 + h * kk3, rho + h * kkk3, H)

        o1 = o1 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        o2 = o2 + h / 6 * (kk1 + 2 * kk2 + 2 * kk3 + kk4)
        rho = rho + h / 6 * (kkk1 + 2 * kkk2 + 2 * kkk3 + kkk4)

        rho_seq[:, i + 1] = rho

    if mode == 1:
        hs = (1.0 - St.unsqueeze(-1)) * jz_ + St.unsqueeze(-1) * jx_
        return rho_seq, hs.detach()
    else:
        return rho_seq
