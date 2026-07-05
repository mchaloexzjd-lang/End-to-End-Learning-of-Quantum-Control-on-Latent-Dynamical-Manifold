import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np


def visualize_bloch_prediction(times, St=None, Ct=None, Ct1=None,
                               true_bloch=None, pred_bloch=None,
                               true_bloch1=None, pred_bloch1=None,
                               f_true=None, f_pred=None,
                               f_true1=None, f_pred1=None, param_noise=None,
                               param=None, mse_t=None, mse_mean=None, idx=None, save_path=None):
    labels = ["σx", "σy", "σz"]
    n_plots = 1 + 3
    if param_noise is not None:
        n_plots += 1
    if mse_t is not None:
        n_plots += 1
    if f_true is not None:
        n_plots += 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
    axes = np.atleast_1d(axes)

    p = 0
    ax = axes[p]
    ax.plot(times, St, 'b-', lw=1.8, label="St")
    ax.set_ylabel("St", color='b')

    if Ct is not None:
        ax2 = ax.twinx()
        ax2.plot(times, Ct, 'orange', lw=1.8, label="Ct")
        if Ct1 is not None:
            ax2.plot(times, Ct1, '--', color='orange', lw=1.8, label="Ct1")
        ax2.set_ylabel("Ct", color='orange')

        lines = ax.get_lines() + ax2.get_lines()
        labels_l = [l.get_label() for l in lines]
        ax.legend(lines, labels_l, loc="best")

    ax.grid(alpha=0.3)
    p += 1

    if param_noise is not None:
        ax = axes[p]
        Gm_noise_rel = (param_noise[:, 0] - param[0]) / param[0]
        gm_noise_rel = (param_noise[:, 1] - param[1]) / param[1]
        ther_noise_rel = (param_noise[:, 2] - param[2]) / param[2]
        ax.plot(times, Gm_noise_rel, lw=1.8, label=f"Gm_noise_rel")
        ax.plot(times, gm_noise_rel, lw=1.8, label=f"gm_noise_rel")
        ax.plot(times, ther_noise_rel, lw=1.8, label=f"ther_noise_rel")
        ax.grid()
        ax.legend()
        p += 1

    for i in range(3):
        ax = axes[p]

        ax.plot(times, true_bloch[:, i], lw=1.8, label=f"True {labels[i]}")

        if pred_bloch is not None:
            ax.plot(times, pred_bloch[:, i], '--', lw=1.8, label=f"Pred {labels[i]}")

        if true_bloch1 is not None:
            ax.plot(times, true_bloch1[:, i], lw=1.8, label=f"True1 {labels[i]}")

        if pred_bloch1 is not None:
            ax.plot(times, pred_bloch1[:, i], '--', lw=1.8, label=f"Pred1 {labels[i]}")

        ax.set_ylim(-1, 1)
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)
        ax.legend()

        p += 1

    # ===== MSE =====
    if mse_t is not None:
        ax = axes[p]
        ax.plot(times, mse_t, color="red", lw=1.8, label="MSE")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(alpha=0.3)
        p += 1

    # ===== Fidelity =====
    if f_true is not None:
        ax = axes[p]

        ax.plot(times, f_true, lw=1.8, label="True")

        if f_pred is not None:
            ax.plot(times, f_pred, '--', lw=1.8, label="Pred")

        if f_true1 is not None:
            ax.plot(times, f_true1, lw=1.8, label="True1")

        if f_pred1 is not None:
            ax.plot(times, f_pred1, '--', lw=1.8, label="Pred1")

        ax.set_ylabel("Fidelity")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time step")

    # ===== title =====
    if param is not None:
        param_str = ", ".join(f"{p:.3f}" for p in param)
    else:
        param_str = ""

    if mse_mean is not None:
        title = f"{idx} Bloch Evolution | param: {param_str} | mse: {mse_mean:.5f}"
    else:
        title = f"{idx} Bloch Evolution | param: {param_str}"

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
