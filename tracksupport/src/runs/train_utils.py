import hydra
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np
from ignite.metrics import Loss, Metric, Accuracy, ConfusionMatrix
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
    manage_lr handler in train.py, supporting dynamic batch sizes:

      1. Linear warmup (warmup_start_lr -> peak_lr)
         Note: Warmup duration is fixed based on initial steps_per_epoch.
      2. Cosine decay within the current segment
      3. On a phase transition:
         - Optionally update batch size (changing steps_per_epoch).
         - Cosine-interpolate to the new LR over transition_epochs.
         - Restart cosine from that LR for the remaining steps.

    Saves *save_dir*/lr_schedule.png with a log y-axis.
    """
    import math

    # Initial configuration
    # Note: total_steps is dynamic now, so we don't calculate a fixed total.
    min_lr = tcfg.get("min_lr", 0.0)
    peak_lr = float(tcfg.lr)
    
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    # warmup_steps is calculated once based on initial SPE and is fixed
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    warmup_start_lr = tcfg.get("warmup_start_lr", 0.0)

    # Build phase-transition lookup: epoch_number -> (phase_name, target_lr, target_bs)
    phase_schedule = tcfg.get("phase_schedule", None)
    phase_transitions = {}
    
    if phase_schedule is not None and phase_schedule.get("enable", False):
        for entry in phase_schedule.schedule:
            if len(entry) >= 4:
                epoch, phase, lr, batch_size = int(entry[0]), str(entry[1]), float(entry[2]), int(entry[3])
            else:
                epoch, phase, lr = int(entry[0]), str(entry[1]), float(entry[2])
                batch_size = None
            phase_transitions[epoch] = (phase, lr, batch_size)

    def cosine_lr_fn(anchor, elapsed, t_max):
        return min_lr + 0.5 * (anchor - min_lr) * (
            1.0 + math.cos(math.pi * min(elapsed, t_max) / max(t_max, 1))
        )

    # Simulation state
    current_spe = steps_per_epoch
    current_bs = tcfg.get("train_batch_size", 1) # used for ratio calculation
    if current_bs is None: current_bs = 1 # fallback

    cos_anchor = peak_lr
    # Initial t_max estimation (assuming no BS changes yet)
    # Note: This t_max is only relevant if we are NOT in warmup and NOT in transition.
    # We will recalculate it dynamically if needed.
    cos_t_max = max(1, tcfg.epochs * steps_per_epoch) 
    cos_elapsed = 0

    trans_active = False
    trans_start_lr = 0.0
    trans_end_lr = 0.0
    trans_elapsed = 0
    trans_duration_steps = 0

    lrs = []
    epochs_axis = []
    
    # We track global steps for cosmetic reasons, but logic relies on epochs
    global_step = 0
    curr_epoch = 0

    while curr_epoch < tcfg.epochs:
        # --- EPOCH_STARTED logic ---
        # Check for phase transition at the start of the epoch
        if curr_epoch in phase_transitions:
            _, new_lr, new_bs = phase_transitions[curr_epoch]
            
            # 1. Update Batch Size (and SPE)
            if new_bs is not None and new_bs > 0:
                # SPE scales inversely with batch size
                # New SPE = Old SPE * (Old BS / New BS)
                ratio = current_bs / new_bs
                current_spe = max(1, int(current_spe * ratio))
                current_bs = new_bs
            
            # 2. Setup Transition
            # Transition happens even if update_bs is None, if phase is in map
            tr_epochs = phase_schedule.get('transition_epochs', 1)
            trans_duration_steps = max(1, int(tr_epochs * current_spe))
            
            trans_active = True
            trans_start_lr = lrs[-1] if lrs else peak_lr
            trans_end_lr = new_lr
            trans_elapsed = 0
            
        # --- ITERATION execution for this epoch ---
        for step_in_epoch in range(current_spe):
            
            # 1. Warmup (Epoch-based)
            if curr_epoch < warmup_epochs:
                # Progress fraction 0.0 -> 1.0 covers the entire warmup duration
                progress_in_epoch = step_in_epoch / current_spe
                total_warmup_progress = (curr_epoch + progress_in_epoch) / warmup_epochs
                lr = warmup_start_lr + total_warmup_progress * (peak_lr - warmup_start_lr)
                
            # 2. Active phase transition
            elif trans_active:
                if trans_elapsed >= trans_duration_steps:
                    lr = trans_end_lr
                    trans_active = False
                    
                    # Recalculate t_max for next cosine phase
                    rem_epochs = tcfg.epochs - curr_epoch
                    rem_in_epoch = max(0, current_spe - step_in_epoch - 1)
                    future_steps = rem_epochs * current_spe + rem_in_epoch
                    
                    cos_anchor = lr
                    cos_t_max = max(1, future_steps)
                    cos_elapsed = 0
                else:
                    cos_alpha = 0.5 * (1.0 - math.cos(math.pi * trans_elapsed / trans_duration_steps))
                    lr = trans_start_lr + cos_alpha * (trans_end_lr - trans_start_lr)
                    trans_elapsed += 1
            
            # 3. Normal cosine decay
            else:
                lr = cosine_lr_fn(cos_anchor, cos_elapsed, cos_t_max)
                cos_elapsed += 1

            lrs.append(lr)
            # Use floating point epoch for x-axis
            epochs_axis.append(curr_epoch + (step_in_epoch / current_spe))
            global_step += 1
            
        curr_epoch += 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs_axis, lrs, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.set_title("Planned LR Schedule (Simulated)")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "lr_schedule.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"LR schedule plot saved to {path}")


def _output_transform_act(output):
    logits = output["y_pred_action"]       # [B, 8]
    y_true_oh = output["y_action"]         # [B, 8] one-hot

    assert logits.shape[0] == y_true_oh.shape[0], "Batch size mismatch"
    assert logits.shape[1] == 8 and y_true_oh.shape[1] == 8

    preds_oh = torch.zeros_like(logits)
    preds_oh.scatter_(1, logits.argmax(dim=1, keepdim=True), 1.0)
    return preds_oh, y_true_oh.argmax(dim=1)

def _output_transform_off(output):
    logits = output["y_pred_offence"]      # [B, 4]
    y_true_oh = output["y_offence"]        # [B, 4] one-hot

    assert logits.shape[0] == y_true_oh.shape[0], "Batch size mismatch"
    assert logits.shape[1] == 4 and y_true_oh.shape[1] == 4

    preds_oh = torch.zeros_like(logits)
    preds_oh.scatter_(1, logits.argmax(dim=1, keepdim=True), 1.0)
    return preds_oh, y_true_oh.argmax(dim=1)


def _build_metrics():
    """Create one fresh metric dict (attach to an evaluator engine)."""
    return {
        "loss": Loss(lambda x, y: x["loss"], output_transform=lambda x: (x, x)),
        "acc_action": Accuracy(output_transform=_output_transform_act),
        "acc_offence": Accuracy(output_transform=_output_transform_off),
        "bal_acc_action": BalancedAccuracy(n_classes=8, output_transform=_output_transform_act),
        "bal_acc_offence": BalancedAccuracy(n_classes=4, output_transform=_output_transform_off),
        "cm_action": ConfusionMatrix(num_classes=8, output_transform=_output_transform_act),
        "cm_offence": ConfusionMatrix(num_classes=4, output_transform=_output_transform_off),
    }

def _get_class_weights(train_dataset, device, logger, debug_mode):
    """Extract and sqrt-dampen class weights from the dataset."""
    if debug_mode:
        logger.info("Debug mode: using unweighted loss for overfitting test")
        return None, None

    w_action = w_offence = None
    try:
        base = (train_dataset.dataset
                if isinstance(train_dataset, torch.utils.data.Subset)
                else train_dataset)
        if hasattr(base, "weights_action"):
            w_action = torch.sqrt(base.weights_action).to(device).float()
            w_action = torch.clamp(w_action, max=50.0)
        if hasattr(base, "weights_offence_severity"):
            w_offence = torch.sqrt(base.weights_offence_severity).to(device).float()
            w_offence = torch.clamp(w_offence, max=50.0)
        if w_action is not None:
            logger.info(f"Class weights action  (sqrt-dampened): {w_action}")
            logger.info(f"Class weights offence (sqrt-dampened): {w_offence}")
    except Exception as e:
        logger.warning(f"Failed to load class weights: {e}")
    return w_action, w_offence


def _maybe_fuse_checkpoint(model, checkpoint_path, local_rank):
    """Fuse the model's BN layers if the checkpoint was saved in fused form."""
    resume_abs_path = hydra.utils.to_absolute_path(checkpoint_path)
    if not os.path.exists(resume_abs_path):
        return
    try:
        chkpt = torch.load(resume_abs_path, map_location="cpu")
        sd = chkpt.get("model", chkpt)
        fused_key = "tracker.tracker.model.model.0.stem1.conv.bias"
        unfused_key = "tracker.tracker.model.model.0.stem1.bn.weight"
        if (fused_key in sd) and (unfused_key not in sd):
            print(f"[Rank {local_rank}] Detected FUSED checkpoint. "
                  "Fusing model to match...")
            tracker = getattr(model, "tracker", None)
            if (tracker
                    and hasattr(tracker, "tracker")
                    and hasattr(tracker.tracker, "fuse")):
                model.tracker.tracker.fuse()
                print(f"[Rank {local_rank}] Model fused successfully.")
    except Exception as e:
        print(f"[Rank {local_rank}] Warning: pre-check of checkpoint failed: {e}")


def calculate_weights(dataset, mode: str = "action"):
    bucket_action = {}
    bucket_offence = {}
    list_action = []
    list_offence = []
    if isinstance(dataset, torch.utils.data.Subset):
          # unwrap Subset to access original dataset
        ds = dataset.dataset
        labels_offence_severity = [dataset.dataset.labels_offence_severity[i] for i in dataset.indices]
        labels_action = [dataset.dataset.labels_action[i] for i in dataset.indices]
    else:
        ds = dataset
        labels_offence_severity = dataset.labels_offence_severity
        labels_action = dataset.labels_action
    for off, act,  in zip(labels_offence_severity, labels_action):
        off_cls = off.argmax().item()
        act_cls = act.argmax().item()

        list_offence.append(off_cls)
        list_action.append(act_cls)

        bucket_action[act_cls] = bucket_action.get(act_cls, 0) + 1
        bucket_offence[off_cls] = bucket_offence.get(off_cls, 0) + 1

    total_action = len(ds)
    total_offence = total_action

    weights_action = {cls: torch.sqrt(torch.Tensor([total_action / count])) for cls, count in bucket_action.items()}
    weights_offence = {cls: torch.sqrt(torch.Tensor([total_offence / count])) for cls, count in bucket_offence.items()}
    weight_offence_tensor = torch.zeros(len(list_offence))
    weight_action_tensor = torch.zeros(len(list_action))

    for idx, (act, off) in enumerate(zip(list_action, list_offence)):
        weight_action_tensor[idx] = weights_action[act]
        weight_offence_tensor[idx] = weights_offence[off]
    
    if mode == "action":
        return weight_action_tensor
    elif mode == "offence":
        return weight_offence_tensor
    else:      
        raise ValueError("Mode must be either 'action' or 'offence'")

