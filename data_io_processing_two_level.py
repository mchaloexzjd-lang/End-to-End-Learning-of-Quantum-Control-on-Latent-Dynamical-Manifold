import h5py
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def Standard(seqs):
    scaler = StandardScaler()
    shape = seqs.shape
    seqs = seqs.reshape(-1, shape[-1])
    scaler.fit(seqs)
    seqs = scaler.transform(seqs).reshape(shape)
    return seqs, scaler


def collect_sequences(data_path):
    with h5py.File(data_path, 'r') as f:
        bloch_seqs = f["bloch_seqs"][:]
        St_seqs = f["St_seqs"][:]
        Ct_seqs = f["Ct_seqs"][:]
        H_seqs = f["H_seqs"][:]
        params = f["params"][:]
    return bloch_seqs, St_seqs, Ct_seqs, params, H_seqs

class SequenceDataset(Dataset):
    def __init__(self, bloch_seq, St_seq, Ct_seq, params):
        self.bloch_seq = torch.tensor(bloch_seq, dtype=torch.float32)
        self.St_seq = torch.tensor(St_seq, dtype=torch.float32)
        self.Ct_seq = torch.tensor(Ct_seq, dtype=torch.float32)
        self.params = torch.tensor(params, dtype=torch.float32)

    def __len__(self):
        return len(self.bloch_seq)

    def __getitem__(self, idx):
        return (self.bloch_seq[idx],
                self.St_seq[idx],
                self.Ct_seq[idx],
                self.params[idx]
                )

class SequenceDataset0(Dataset):
    def __init__(self, bloch_seq, Ct_seq, params):
        self.bloch_seq = torch.tensor(bloch_seq, dtype=torch.float32)
        self.Ct_seq = torch.tensor(Ct_seq, dtype=torch.float32)
        self.params = torch.tensor(params, dtype=torch.float32)

    def __len__(self):
        return len(self.bloch_seq)

    def __getitem__(self, idx):
        return (self.bloch_seq[idx],
                self.Ct_seq[idx],
                self.params[idx]
                )
