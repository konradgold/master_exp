from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np
from ignite.metrics import Metric
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import os

class BalancedAccuracy(Metric):
    def __init__(self, n_classes, output_transform=lambda x: x, device="cpu"):
        self._preds = []
        self._targets = []
        self.n_classes = n_classes
        super().__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._preds = []
        self._targets = []
        super().reset()

    def update(self, output):
        y_pred, y = output

        self._preds.extend(y_pred.detach().cpu())
        self._targets.extend(y.detach().cpu())

    def compute(self):
        if len(self._preds) == 0:
            return 0.0
    
        y_pred = torch.stack(self._preds).argmax(dim=1).numpy()  # [N, C]

        y_true = torch.stack(self._targets)  # [N, C]

        return balanced_accuracy_score(y_true, y_pred)


def plot_lr_schedule(tcfg, steps_per_epoch, save_dir):
    """
    Simulate the full LR schedule and save a plot to *save_dir*/lr_schedule.png.
    Uses a lightweight dummy optimizer so no GPU memory is needed.
    """
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    dummy_optim = torch.optim.AdamW([dummy_param], lr=tcfg.lr)

    total_steps = tcfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        dummy_optim, T_max=total_steps, eta_min=tcfg.get("min_lr", 0.0)
    )

    warmup_epochs = tcfg.get("warmup_epochs", 0)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    warmup_start_lr = tcfg.get("warmup_start_lr", 0.0)

    lrs = []
    for step in range(total_steps):
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup
            alpha = step / warmup_steps
            lr = warmup_start_lr + alpha * (tcfg.lr - warmup_start_lr)
        else:
            lr = sched.get_last_lr()[0]
        lrs.append(lr)
        dummy_optim.step()
        sched.step()

    epochs = [s / steps_per_epoch for s in range(total_steps)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lrs, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.set_title("Planned LR Schedule")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "lr_schedule.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"LR schedule plot saved to {path}")


def plot_lr_schedule_with_phases(tcfg, steps_per_epoch, save_dir):
    """
    Simulate the full LR schedule using the same piecewise logic as the unified
    manage_lr handler in train_mvit.py:

      1. Linear warmup (warmup_start_lr -> peak_lr)
      2. Cosine decay within the current segment
      3. On a phase transition: cosine-interpolate to the new LR over
         transition_epochs, then restart cosine from that LR for the
         remaining steps.

    Saves *save_dir*/lr_schedule.png with a log y-axis.
    """
    import math

    total_steps = tcfg.epochs * steps_per_epoch
    min_lr = tcfg.get("min_lr", 0.0)
    peak_lr = float(tcfg.lr)
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    warmup_start_lr = tcfg.get("warmup_start_lr", 0.0)

    # Build phase-transition lookup: epoch_number (1-indexed) -> target_lr
    phase_schedule = tcfg.get("phase_schedule", None)
    transition_steps = 0
    phase_map = {}  # step_index -> target_lr  (fires at start of that epoch)
    if phase_schedule is not None and phase_schedule.get("enable", False):
        trans_epochs = phase_schedule.get("transition_epochs", 1)
        transition_steps = max(1, int(trans_epochs * steps_per_epoch))
        for entry in phase_schedule.schedule:
            t_epoch, _, target_lr = int(entry[0]), str(entry[1]), float(entry[2])
            # ignite epoch N fires EPOCH_STARTED before step (N-1)*spe
            phase_map[(t_epoch - 1) * steps_per_epoch] = float(target_lr)

    def cosine_lr_fn(anchor, elapsed, t_max):
        return min_lr + 0.5 * (anchor - min_lr) * (
            1.0 + math.cos(math.pi * min(elapsed, t_max) / max(t_max, 1))
        )

    # Simulation state (mirrors manage_lr + check_phase_switch)
    cos_anchor = peak_lr
    cos_t_max = max(1, total_steps - warmup_steps)
    cos_elapsed = 0

    trans_active = False
    trans_start_lr = 0.0
    trans_end_lr = 0.0
    trans_elapsed = 0

    lrs = []
    for step in range(total_steps):
        # EPOCH_STARTED fires before the first iteration of each epoch
        if step % steps_per_epoch == 0 and step in phase_map:
            current_lr = lrs[-1] if lrs else peak_lr
            trans_active = True
            trans_start_lr = current_lr
            trans_end_lr = phase_map[step]
            trans_elapsed = 0

        # ITERATION_STARTED: determine LR for this step
        if step < warmup_steps:
            alpha = step / max(warmup_steps, 1)
            lr = warmup_start_lr + alpha * (peak_lr - warmup_start_lr)
        elif trans_active:
            if trans_elapsed >= transition_steps:
                lr = trans_end_lr
                trans_active = False
                remaining = max(1, total_steps - step)
                cos_anchor = lr
                cos_t_max = remaining
                cos_elapsed = 0
            else:
                cos_alpha = 0.5 * (1.0 - math.cos(math.pi * trans_elapsed / transition_steps))
                lr = trans_start_lr + cos_alpha * (trans_end_lr - trans_start_lr)
                trans_elapsed += 1
        else:
            lr = cosine_lr_fn(cos_anchor, cos_elapsed, cos_t_max)
            cos_elapsed += 1

        lrs.append(lr)

    epochs = [s / steps_per_epoch for s in range(total_steps)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lrs, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.set_title("Planned LR Schedule")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "lr_schedule.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"LR schedule plot saved to {path}")
