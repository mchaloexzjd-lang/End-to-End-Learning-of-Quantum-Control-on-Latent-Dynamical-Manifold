import os
import csv
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from RK4_elvo import RK4_bloch
from data_io_processing_two_level import collect_sequences, SequenceDataset, Standard
from auxiliary_calc_two_level import rho_from_bloch_, bloch_from_rho_


scaler_torch = torch.cuda.amp.GradScaler()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BLOCH_SIZE = 3
St_SIZE = 2
Ct_SIZE = 2
PARAM_DIM = 3

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
T = 5
tn = T * 200

i_min = 5
M = 30
delta_Ct_dim = M - i_min + 1
omega_control = torch.pi / 5
num_coeff = M - i_min + 1

t_control = torch.linspace(0, T, tn + 1, device=device).view(1, tn + 1, 1)
i_vals = torch.arange(i_min, i_min + num_coeff, device=device).view(1, 1, num_coeff)
SIN_MATRIX = torch.sin((i_vals + 1) * omega_control * t_control)

target = torch.tensor([[-1.0], [1.0]], dtype=torch.complex64, device=device) / torch.sqrt(torch.tensor(2))
target_dagger = target.conj().T  # shape (1,2)


def control_improve(delta_I_list, ct_ideal):
    control = torch.sum(delta_I_list[:, None] * SIN_MATRIX, dim=-1, keepdim=True)
    control = control + ct_ideal
    return control


def get_tf_ratio(epoch):
    tf_max = 1.0
    tf_min = 0.2
    mid_epoch = 60
    k = 20
    return tf_min + (tf_max - tf_min) / (1 + np.exp((epoch - mid_epoch) / k))


class LSTM_Bloch_Predictor(nn.Module):
    def __init__(self, nn_param):
        hidden_dim, num_layers, dropout = nn_param[0], nn_param[1], nn_param[2]
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = BLOCH_SIZE + St_SIZE + PARAM_DIM
        self.input_dim += Ct_SIZE
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, BLOCH_SIZE)
        self.fc_control = nn.Linear(hidden_dim, delta_Ct_dim)

    # train_forward
    def forward_chunk_ss(self, bloch_seq, St_seq, Ct_seq, param, tf_ratio=1.0, chunk_size=20, Ct_imp=False):
        batch_size = bloch_seq.shape[0]
        St_seq = torch.cat((St_seq[:, :-1], St_seq[:, 1:]), dim=-1)
        Ct_seq = torch.cat((Ct_seq[:, :-1], Ct_seq[:, 1:]), dim=-1)

        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros_like(h)
        outputs = []

        y_prev = bloch_seq[:, 0:1, :]

        for start in range(0, tn, chunk_size):
            end = min(start + chunk_size, tn)

            use_tf = (torch.rand(1).item() < tf_ratio)
            if use_tf:
                input_chunk = torch.cat([
                    bloch_seq[:, start:end, :],
                    St_seq[:, start:end, :],
                    Ct_seq[:, start:end, :],
                    param[:, start:end, :],
                ], dim=-1)
                out_chunk, (h, c) = self.lstm(input_chunk, (h, c))
                pred_chunk = self.fc(out_chunk)
            else:
                preds = []
                for t in range(start, end):
                    input_t = torch.cat([y_prev, St_seq[:, t:t + 1], Ct_seq[:, t:t + 1], param[:, t:t + 1]], dim=-1)
                    out, (h, c) = self.lstm(input_t, (h, c))
                    pred = self.fc(out)
                    preds.append(pred)
                    y_prev = pred.detach()
                pred_chunk = torch.cat(preds, dim=1)

            outputs.append(pred_chunk)
            y_prev = pred_chunk[:, -1:, :]
        if Ct_imp:
            delta_I_list = self.fc_control(h[-1].detach())
            return torch.cat(outputs, dim=1), delta_I_list
        else:
            return torch.cat(outputs, dim=1)

    # -------- use_forward --------
    def forward_ar(self, bloch_seq0, St_seq, Ct_seq, param, Ct_imp=False):
        batch_size = Ct_seq.shape[0]
        y_prev = bloch_seq0
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros_like(h)
        output_seq = torch.zeros(batch_size, tn, BLOCH_SIZE, device=device)
        St_seq = torch.cat((St_seq[:, :-1], St_seq[:, 1:]), dim=-1)
        Ct_seq = torch.cat((Ct_seq[:, :-1], Ct_seq[:, 1:]), dim=-1)
        for t in range(tn):
            input_t = torch.cat([y_prev, St_seq[:, t:t + 1], Ct_seq[:, t:t + 1], param[:, t:t + 1]],  dim=-1)
            out, (h, c) = self.lstm(input_t, (h, c))
            pred = self.fc(out[:, -1:, :])
            output_seq[:, t, :] = pred[:, 0]
            y_prev = pred
        if Ct_imp:
            delta_I_list = self.fc_control(h[-1])
            return output_seq, delta_I_list
        else:
            return output_seq

def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    loop = tqdm(range(1, EPOCHS + 1), desc="training")

    scaler = joblib.load(PARAMS_SCALE_PATH)
    mean = torch.tensor(scaler.mean_, device=device, dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, device=device, dtype=torch.float32)
    train_loader_len = len(train_loader.dataset)

    for ep in loop:
        model.train()
        tf_ratio = get_tf_ratio(ep)
        running_loss0 = running_loss1 = running_loss2 = 0
        if ep < 30:
            chunk_size = 100
        elif ep < 50:
            chunk_size = 50
        elif ep < 80:
            chunk_size = 20
        else:
            chunk_size = 10
        for bloch_seq, St_seq, Ct_seq, param in train_loader:
            optimizer.zero_grad()
            bloch_seq = bloch_seq.to(device)
            St_seq = St_seq.to(device)
            Ct_seq = Ct_seq.to(device)
            param = param.to(device)
            if ep > 70:
                with torch.cuda.amp.autocast():
                    pred_seq, delta_I_list = model.forward_chunk_ss(bloch_seq, St_seq, Ct_seq, param,
                                                                    tf_ratio, chunk_size, Ct_imp=True)
                    loss0 = criterion(pred_seq, bloch_seq[:, 1:])
            else:
                with torch.cuda.amp.autocast():
                    pred_seq = model.forward_chunk_ss(bloch_seq, St_seq, Ct_seq, param, tf_ratio, chunk_size)
                    loss0 = criterion(pred_seq, bloch_seq[:, 1:])
            loss1 = torch.tensor(0)
            loss2 = torch.tensor(0)
            bound_penalty = torch.tensor(0)
            param_orig = param * scale + mean

            if ep > 40:
                if ep <= 70:
                    min_val = -0.15
                    max_val = 0.15
                    delta_I_list = (max_val - min_val) * torch.rand(Ct_seq.shape[0], delta_Ct_dim,
                                                                    device=device) + min_val
                    Ct_seq1 = control_improve(delta_I_list, Ct_seq)
                    Ct_seq1_orig = Ct_seq1 * 60
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=False):
                            bloch_seq1 = RK4_bloch(St_seq, Ct_seq1_orig, param_orig, T=T, tn=tn)
                else:
                    Ct_seq1 = control_improve(delta_I_list, Ct_seq)
                    Ct_seq1_orig = Ct_seq1 * 60
                    with torch.cuda.amp.autocast(enabled=False):
                        bloch_seq1, hs = RK4_bloch(St_seq, Ct_seq1_orig, param_orig, T=T, tn=tn, mode=1)

                    F = (target_dagger @ bloch_seq1[:, -1] @ target).real
                    loss1 = 1 - F.mean()
                    bound_penalty = torch.mean(
                        torch.relu(Ct_seq1_orig - 70) ** 2 +
                        torch.relu(-70 - Ct_seq1_orig) ** 2
                    )
                with torch.cuda.amp.autocast():
                    pred1 = model.forward_chunk_ss(bloch_seq, St_seq, Ct_seq1, param, tf_ratio, chunk_size)
                    loss2 = criterion(pred1, bloch_from_rho_(bloch_seq1.detach()[:, 1:]))
            loss = loss0 + loss1*1e-2 + loss2 + bound_penalty

            scaler_torch.scale(loss).backward()
            scaler_torch.step(optimizer)
            scaler_torch.update()

            bs = bloch_seq.shape[0]
            running_loss0 += loss0.item() * bs
            running_loss1 += loss1.item() * bs
            running_loss2 += loss2.item() * bs

        train_loss0 = running_loss0 / train_loader_len
        train_loss1 = running_loss1 / train_loader_len
        train_loss2 = running_loss2 / train_loader_len
        train_loss_list = [train_loss0, train_loss1, train_loss2]

        val_loss, val_loss_list = evaluate_model_loader(model, val_loader, PARAMS_SCALE_PATH)

        train_loss_str = [f"{v:.3e}" for v in train_loss_list]
        val_loss_str = [f"{v:.3e}" for v in val_loss_list]
        print(f"Epoch {ep}/{EPOCHS}  train_loss={train_loss_str}  val_loss={val_loss_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler_torch.state_dict(),
                'loss': loss,
            }, MODEL_SAVE_PATH + f'{ep}.pth')
            print(f"⭐ Epoch {ep}: best save, best_loss={best_val_loss:.6f}")
        else:
            if ep % 2 == 0:
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler_torch.state_dict(),
                    'loss': loss,
                }, MODEL_SAVE_PATH + f'{ep}.pth')

        with open(LOSS_SAVE_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ep] + train_loss_str + val_loss_str)


def evaluate_model_loader(model, data_loader, path, mode=0):
    model.eval()

    scaler = joblib.load(path)
    mean = torch.tensor(scaler.mean_, device=device, dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, device=device, dtype=torch.float32)

    total_loss = 0
    total_size = 0
    mse_all_sum = 0
    mse_all1_sum = 0
    F_preds_sum = 0
    F_trues_sum = 0
    F_preds1_sum = 0
    F_trues1_sum = 0

    with torch.no_grad():

        for bloch_seq, St_seq, Ct_seq, param in data_loader:
            bloch_seq = bloch_seq.to(device)
            St_seq = St_seq.to(device)
            Ct_seq = Ct_seq.to(device)
            param = param.to(device)

            bs = bloch_seq.shape[0]
            total_size += bs

            bloch_seq0 = bloch_seq[:, :1]

            pred_seqs, delta_I_list = model.forward_ar(bloch_seq0, St_seq, Ct_seq, param, True)
            pred_seqs = torch.cat([bloch_seq0, pred_seqs], dim=1)

            param_orig = param * scale + mean
            Ct_seq1 = control_improve(delta_I_list, Ct_seq)
            Ct_seq1_orig = Ct_seq1 * 60

            bloch_seq1 = RK4_bloch(St_seq, Ct_seq1_orig, param_orig, T=T, tn=tn)
            pred_seqs1 = model.forward_ar(bloch_seq0, St_seq, Ct_seq1, param)
            pred_seqs1 = torch.cat([bloch_seq0, pred_seqs1], dim=1)
            # ---------- MSE ----------
            mse_all = torch.mean((pred_seqs - bloch_seq) ** 2)
            mse_all1 = torch.mean((pred_seqs1 - bloch_from_rho_(bloch_seq1)) ** 2)
            # ---------- Fidelity ----------
            F_preds = (target_dagger @ rho_from_bloch_(pred_seqs[:, -1]) @ target).real.mean()
            F_trues = (target_dagger @ rho_from_bloch_(bloch_seq[:, -1]) @ target).real.mean()
            F_preds1 = (target_dagger @ rho_from_bloch_(pred_seqs1[:, -1]) @ target).real.mean()
            F_trues1 = (target_dagger @ bloch_seq1[:, -1] @ target).real.mean()
            loss = mse_all + mse_all1
            loss += - (F_trues1 - F_trues) * 1e-2
            total_loss += loss.item() * bs
            mse_all_sum += mse_all.item() * bs
            mse_all1_sum += mse_all1.item() * bs
            F_preds_sum += F_preds.item() * bs
            F_trues_sum += F_trues.item() * bs
            F_preds1_sum += F_preds1.item() * bs
            F_trues1_sum += F_trues1.item() * bs
    loss = total_loss / total_size
    mse_all = mse_all_sum / total_size
    mse_all1 = mse_all1_sum / total_size
    F_preds = F_preds_sum / total_size
    F_trues = F_trues_sum / total_size
    F_preds1 = F_preds1_sum / total_size
    F_trues1 = F_trues1_sum / total_size
    if mode == 0:
        return loss, [mse_all, mse_all1, F_preds, F_trues, F_preds1, F_trues1]
    elif mode == 1:
        return ([Ct_seq * 60, Ct_seq1_orig],
                [pred_seqs, bloch_seq, mse_all, F_preds, F_trues],
                [pred_seqs1, bloch_seq1, mse_all1, F_preds1, F_trues1],
                )

def main(nn_param):
    bloch_seqs, St_seqs, Ct_seqs, params, _ = collect_sequences(DATA_PATH)
    scale_dict = {}
    params, scale_dict['params'] = Standard(params)
    os.makedirs(os.path.normpath(MODEL_SAVE_PATH).split(os.sep)[0], exist_ok=True)
    joblib.dump(scale_dict['params'], PARAMS_SCALE_PATH)
    Ct_seqs = Ct_seqs / 60

    n_train = int(len(bloch_seqs) * 0.9)

    train_dataset = SequenceDataset(bloch_seqs[:n_train], St_seqs[:n_train], Ct_seqs[:n_train], params[:n_train])

    val_dataset = SequenceDataset(bloch_seqs[n_train:], St_seqs[n_train:], Ct_seqs[n_train:], params[n_train:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = LSTM_Bloch_Predictor(nn_param).to(device)

    train_model(model, train_loader, val_loader)


if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    for HIDDEN_SIZE in [256]:
        for NUM_LAYERS in [3]:
            for DROPOUT in [0.1]:
                NN_param = [HIDDEN_SIZE, NUM_LAYERS, DROPOUT]
                PARAMS_SCALE_PATH = f"network_save/params_scale_two_level.pkl"
                DATA_PATH = f"data/tow_level_Train.h5"
                LOSS_SAVE_PATH = f"network_save/{NN_param}_loss{SEED}.csv"
                MODEL_SAVE_PATH = f"network_save/{NN_param}{SEED}"
                main(NN_param)
