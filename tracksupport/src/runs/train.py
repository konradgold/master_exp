"""
Unified training script for all model types.

Uses a registry-based ``build_model`` approach driven by ``cfg.model_type``
and a phase-aware cosine LR schedule with warmup that works for every model.

Usage:
    torchrun --nproc_per_node=N src/runs/train.py model=vjepa
    torchrun --nproc_per_node=N src/runs/train.py model=mvit
"""

import logging
import math
import os

import ignite

# --- SLURM environment variable stripping ---
for key in list(os.environ.keys()):
    if key.startswith("SLURM_"):
        del os.environ[key]
os.environ["NCCL_P2P_DISABLE"] = "1"

import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
import ignite.distributed as idist
from omegaconf import DictConfig, OmegaConf

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import Checkpoint, FastaiLRFinder, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar, WandBLogger
from ignite.utils import setup_logger

import matplotlib
matplotlib.use("Agg")

from runs.train_utils import _maybe_fuse_checkpoint, calculate_weights, plot_lr_schedule_with_phases, _build_metrics, _get_class_weights
from models import build_model
from datasets.mvfoul import MultiViewDataset

import debugpy

if os.getenv("LOCAL_RANK", "0") == "0" and False:  # only rank 0
    port = 5670
    debugpy.listen(("0.0.0.0", port))
    print(f"Waiting for debugger attach on port {port}...")
    debugpy.wait_for_client()
    debugpy.breakpoint()
# ------------------------------------------------------------------ #
# Engine step functions                                                #
# ------------------------------------------------------------------ #

def _train_step_fn(model, optimizer, criterion_action, criterion_offence,
                   device, grad_accum_steps):
    """Return the training step closure used by the ``Engine``."""

    def _step(engine, batch):
        model.train()
        labels_offence, labels_action, videos, _ = batch

        videos = videos.to(device, non_blocking=True)
        labels_action = labels_action.to(device, non_blocking=True).squeeze(1)
        labels_offence = labels_offence.to(device, non_blocking=True).squeeze(1)

        pred_action, pred_offence, aux = model(videos)

        loss_action = criterion_action(pred_action, labels_action)
        loss_offence = criterion_offence(pred_offence, labels_offence)
        loss = loss_action + loss_offence

        (loss / grad_accum_steps).backward()

        if engine.state.iteration % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        output = {
            "loss": loss,
            "loss_action": loss_action,
            "loss_offence": loss_offence,
            "y_pred_action": pred_action,
            "y_action": labels_action,
            "y_pred_offence": pred_offence,
            "y_offence": labels_offence,
        }

        if aux is not None and isinstance(aux, torch.Tensor) and aux.ndim == 3:
             output["interp_scores_mean"] = aux.mean().item()
             output["interp_scores_std"] = aux.std().item()
             output["interp_scores_min"] = aux.min().item()
             output["interp_scores_max"] = aux.max().item()
        
        return output

    return _step


def _val_step_fn(model, criterion_action, criterion_offence, device):
    """Return the validation step closure used by the ``Engine``."""

    def _step(engine, batch):
        model.eval()
        with torch.no_grad():
            labels_offence, labels_action, videos, _ = batch

            videos = videos.to(device, non_blocking=True)
            labels_action = labels_action.to(device, non_blocking=True).squeeze(1)
            labels_offence = labels_offence.to(device, non_blocking=True).squeeze(1)

            pred_action, pred_offence, _ = model(videos)

            loss_action = criterion_action(pred_action, labels_action)
            loss_offence = criterion_offence(pred_offence, labels_offence)
            loss = loss_action + loss_offence

            return {
                "loss": loss,
                "loss_action": loss_action,
                "loss_offence": loss_offence,
                "y_pred_action": pred_action,
                "y_action": labels_action,
                "y_pred_offence": pred_offence,
                "y_offence": labels_offence,
            }

    return _step

# ------------------------------------------------------------------ #
# Unified LR manager                                                   #
# ------------------------------------------------------------------ #

def _setup_lr_manager(trainer, optimizer, tcfg, steps_per_epoch, total_steps,
                      model, logger, train_loader):
    """
    Attach handlers for warmup + piecewise cosine + phase transitions.

    Phase transitions optionally switch ``model.phase`` (if the model
    supports the attribute) and cosine-interpolate the LR to a new
    target value.
    Also supports changing batch size at phase transitions.
    """
    min_lr = tcfg.get("min_lr", 0.0)
    warmup_epochs = tcfg.get("warmup_epochs", 0)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    warmup_start_lr = tcfg.get("warmup_start_lr", 0.0)

    # Phase-transition lookup
    phase_schedule = tcfg.get("phase_schedule", None)
    phase_transitions = {}               # epoch -> (phase_name, target_lr, target_bs)
    initial_transition_steps = 0
    if phase_schedule is not None and phase_schedule.enable:
        transition_epochs = phase_schedule.get("transition_epochs", 1)
        initial_transition_steps = max(1, int(transition_epochs * steps_per_epoch))
        for entry in phase_schedule.schedule:
            if len(entry) >= 4:
                epoch, phase, lr, batch_size = int(entry[0]), str(entry[1]), float(entry[2]), int(entry[3])
            else:
                epoch, phase, lr = int(entry[0]), str(entry[1]), float(entry[2])
                batch_size = None
            phase_transitions[epoch] = (phase, lr, batch_size)

    # Mutable state for the cosine segment & phase transitions
    _cos = {
        "anchor": float(tcfg.lr),
        "t_max": max(1, total_steps - warmup_steps),
        "elapsed": 0,
    }
    _trans = {
        "active": False, 
        "start_lr": 0.0, 
        "end_lr": 0.0, 
        "elapsed": 0,
        "steps": initial_transition_steps
    }
    _epoch_state = {"step": 0}

    def _cosine_lr(anchor, elapsed, t_max):
        return min_lr + 0.5 * (anchor - min_lr) * (
            1.0 + math.cos(math.pi * min(elapsed, t_max) / t_max)
        )

    @trainer.on(Events.EPOCH_STARTED)
    def check_phase_switch(engine):
        # Always reset step counter at start of epoch
        _epoch_state["step"] = 0
        
        epoch = engine.state.epoch
        if epoch not in phase_transitions:
            return
        new_phase, new_lr, new_bs = phase_transitions[epoch]
        raw_model = model.module if hasattr(model, "module") else model

        # Switch model phase if supported
        old_phase = None
        if hasattr(raw_model, "phase"):
            old_phase = raw_model.phase
            raw_model.phase = new_phase
        
        # Switch Batch Size if requested
        bs_msg = ""
        if new_bs is not None:
            world_size = idist.get_world_size() 
            target_bs = new_bs * world_size
            
            if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "batch_size"):
                old_bs = train_loader.batch_sampler.batch_size
                train_loader.batch_sampler.batch_size = target_bs
                bs_msg = f" | BS {old_bs}->{target_bs} (Base: {new_bs})"
            else:
                bs_msg = f" | WARNING: Could not set batch size to {target_bs}, loader missing batch_sampler.batch_size"

        current_lr = optimizer.param_groups[0]["lr"]
        
        # Recalculate transition steps based on current steps per epoch
        tr_epochs = phase_schedule.get('transition_epochs', 1)
        curr_steps = len(train_loader)
        
        # Update epoch_length to reflect the new batch size.
        # Also rebase state.iteration so that Ignite's epoch computation
        # (epoch = iteration // epoch_length) stays consistent – without
        # this, changing epoch_length mid-run shifts the perceived epoch
        # count and causes premature termination.
        engine.state.epoch_length = curr_steps
        engine.state.iteration = (engine.state.epoch - 1) * curr_steps

        _trans.update(active=True, start_lr=current_lr,
                      end_lr=new_lr, elapsed=0,
                      steps=max(1, int(tr_epochs * curr_steps)))

        trainable = sum(p.numel() for p in raw_model.parameters()
                        if p.requires_grad)
        phase_msg = (f"Phase '{old_phase}' -> '{new_phase}'"
                     if old_phase is not None else "")
        logger.info(
            f"Epoch {epoch}: {phase_msg}"
            f" | LR {current_lr:.2e} -> {new_lr:.2e}"
            f"{bs_msg} | steps_per_epoch={curr_steps} | "
            f"Trainable params: {trainable:,}"
        )

    @trainer.on(Events.ITERATION_STARTED)
    def manage_lr(engine):
        step = engine.state.iteration - 1
        
        # Logic update: Use epoch-based progress for warmup to handle dynamic SPE
        curr_epoch_idx = engine.state.epoch - 1
        
        # 1. Warmup
        if curr_epoch_idx < warmup_epochs:
            current_spe = len(train_loader)
            progress_in_epoch = _epoch_state["step"] / max(1, current_spe)
            # fraction of total warmup time
            total_warmup_progress = (curr_epoch_idx + progress_in_epoch) / max(1, warmup_epochs)
            lr = warmup_start_lr + total_warmup_progress * (float(tcfg.lr) - warmup_start_lr)

        # 2. Active phase transition
        elif _trans["active"]:
            e = _trans["elapsed"]
            t_steps = _trans["steps"]
            if e >= t_steps:
                lr = _trans["end_lr"]
                _trans["active"] = False
                
                # Recalculate t_max for next cosine phase
                curr_steps = len(train_loader)
                rem_epochs = tcfg.epochs - engine.state.epoch
                cur_in_epoch = (engine.state.iteration - 1) % curr_steps
                rem_in_epoch = max(0, curr_steps - cur_in_epoch - 1)
                future_steps = rem_epochs * curr_steps + rem_in_epoch
                
                _cos["anchor"] = lr
                _cos["t_max"] = max(1, future_steps)
                _cos["elapsed"] = 0
            else:
                cos_a = 0.5 * (1.0 - math.cos(math.pi * e / t_steps))
                lr = _trans["start_lr"] + cos_a * (_trans["end_lr"] - _trans["start_lr"])
                _trans["elapsed"] += 1
        # 3. Normal cosine decay
        else:
            lr = _cosine_lr(_cos["anchor"], _cos["elapsed"], _cos["t_max"])
            _cos["elapsed"] += 1

        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        _epoch_state["step"] += 1




# ------------------------------------------------------------------ #
# WandB setup                                                          #
# ------------------------------------------------------------------ #

def _setup_wandb(tcfg, cfg, trainer, evaluator, train_evaluator, optimizer):
    """Configure WandB logging if ``tcfg.wandb.enable`` is True."""
    if not tcfg.wandb.enable:
        return

    local_rank = idist.get_local_rank()
    print(f"[Rank {local_rank}] Configuring WandB...")
    if idist.get_rank() == 0:
        print(f"[Rank {local_rank}] Initializing WandB Logger...")
    else:
        return  # only rank 0 should initialize the logger to avoid duplicates

    wandb_logger = WandBLogger(
        project=tcfg.wandb.project,
        entity=tcfg.wandb.entity,
        name=tcfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    if idist.get_rank() == 0:
        print(f"[Rank {local_rank}] WandB Logger initialized.")

    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=tcfg.wandb.log_frequency),
        tag="train",
        output_transform=lambda x: {
            k: v for k, v in x.items() if k in {
                "loss", "loss_action", "loss_offence",
                "interp_scores_mean", "interp_scores_std",
                "interp_scores_min", "interp_scores_max"
            }
        },
    )
    wandb_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="train_eval",
        metric_names=["loss", "acc_action", "acc_offence",
                       "bal_acc_action", "bal_acc_offence"],
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    wandb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="val",
        metric_names=["loss", "acc_action", "acc_offence",
                       "bal_acc_action", "bal_acc_offence"],
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    wandb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED(every=tcfg.wandb.log_frequency * 10),
        optimizer=optimizer,
    )

    def _close(engine):
        wandb_logger.close()
    trainer.add_event_handler(Events.COMPLETED, _close)


# ------------------------------------------------------------------ #
# Main training loop                                                   #
# ------------------------------------------------------------------ #

def run_training(local_rank: int, cfg: DictConfig):
    print(f"[Rank {local_rank}] Starting run_training...")
    logger = setup_logger(
        name="Trainer",
        level=logging.INFO if idist.get_rank() == 0 else logging.WARNING,
    )
    device = idist.device()
    print(f"[Rank {local_rank}] Device: {device}")

    tcfg = cfg.training

    # ---- Data --------------------------------------------------------
    print(f"[Rank {local_rank}] Creating datasets...")
    ds_kwargs = dict(
        path=tcfg.data_path,
        start=tcfg.start_frame,
        end=tcfg.end_frame,
        fps=tcfg.fps,
        num_views=tcfg.num_views,
        num_frames=cfg.num_frames,
    )
    train_dataset = MultiViewDataset(**ds_kwargs, split="train")
    val_dataset = MultiViewDataset(**ds_kwargs, split="val")

    if tcfg.debug.enable:
        print(f"[Rank {local_rank}] Debug mode enabled: limiting datasets...")
        subset_size = tcfg.debug.size
        if len(train_dataset) > subset_size:
            train_dataset = torch.utils.data.Subset(
                train_dataset, torch.randperm(len(train_dataset))[:subset_size])
        if len(val_dataset) > subset_size:
            val_dataset = torch.utils.data.Subset(
                val_dataset, torch.randperm(len(val_dataset))[:subset_size])
        print(f"[Rank {local_rank}] Debug mode: "
              f"Train={len(train_dataset)}, Val={len(val_dataset)}")
        
    if tcfg.use_weighted_sampler:
        sampler = ignite.distributed.auto.DistributedProxySampler(
            torch.utils.data.WeightedRandomSampler(weights=calculate_weights(train_dataset), num_samples=len(train_dataset), replacement=True),
            rank=idist.get_rank(), num_replicas=idist.get_world_size())
    else:
        sampler = None
    train_loader = idist.auto_dataloader(
        train_dataset, batch_size=tcfg.train_batch_size*idist.get_world_size(), shuffle=(sampler is None),
        sampler=sampler,
        num_workers=tcfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = idist.auto_dataloader(
        val_dataset, batch_size=tcfg.val_batch_size, shuffle=False,
        num_workers=tcfg.num_workers, pin_memory=True,
    )

    # ---- Model -------------------------------------------------------
    print(f"[Rank {local_rank}] Initializing model (type={cfg.model_type})...")
    model = build_model(cfg)
    model.to(device)
    print(f"[Rank {local_rank}] Model initialized and moved to device.")

    # Checkpoint fusion check
    if tcfg.checkpoint.resume and tcfg.checkpoint.resume_from:
        _maybe_fuse_checkpoint(model, tcfg.checkpoint.resume_from, local_rank)

    # DDP wrapping
    print(f"[Rank {local_rank}] Wrapping model with idist.auto_model...")
    if idist.get_world_size() > 1:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        model = idist.auto_model(model, find_unused_parameters=True)
        print(f"[Rank {local_rank}] Trainable parameters: "
              f"{sum(p.numel() for p in trainable_params)}")

    # ---- Optimizer & Loss --------------------------------------------
    # Include ALL params so optimizer state survives phase transitions.
    print(f"[Rank {local_rank}] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay,
    )
    optimizer = idist.auto_optim(optimizer)
    print(f"[Rank {local_rank}] Optimizer wrapped.")

    steps_per_epoch = len(train_loader)
    total_steps = tcfg.epochs * steps_per_epoch

    w_action, w_offence = _get_class_weights(
        train_dataset, device, logger, tcfg.debug.enable)
    criterion_action = nn.CrossEntropyLoss(
        weight=w_action if not tcfg.use_weighted_sampler else None, label_smoothing=tcfg.label_smoothing)
    criterion_offence = nn.CrossEntropyLoss(
        weight=w_offence if not tcfg.use_weighted_sampler else None, label_smoothing=tcfg.label_smoothing)

    # ---- Engines -----------------------------------------------------
    grad_accum_steps = tcfg.get("grad_accum_steps", 1)
    print(f"[Rank {local_rank}] Creating engines...")

    trainer = Engine(_train_step_fn(
        model, optimizer, criterion_action, criterion_offence,
        device, grad_accum_steps))
    evaluator = Engine(_val_step_fn(
        model, criterion_action, criterion_offence, device))
    train_evaluator = Engine(_val_step_fn(
        model, criterion_action, criterion_offence, device))
    print(f"[Rank {local_rank}] Engines created.")

    # ---- Metrics -----------------------------------------------------
    for engine in (evaluator, train_evaluator):
        for name, metric in _build_metrics().items():
            metric.attach(engine, name)

    RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x["loss_action"]).attach(
        trainer, "loss_action")
    RunningAverage(output_transform=lambda x: x["loss_offence"]).attach(
        trainer, "loss_offence")

    # ---- LR schedule -------------------------------------------------
    if not tcfg.fastai_lr.enable:
        _setup_lr_manager(
            trainer, optimizer, tcfg, steps_per_epoch, total_steps, model, logger, train_loader)

    # ---- Progress bars -----------------------------------------------
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["loss"])
    pbar.attach(evaluator)
    pbar.attach(train_evaluator)

    # ---- Evaluation & logging ----------------------------------------
    eval_every = tcfg.get("eval_every_n_epochs", 1)

    @trainer.on(Events.EPOCH_COMPLETED(every=eval_every))
    def run_evaluation(engine):
        train_evaluator.run(train_loader)
        evaluator.run(val_loader)

    @train_evaluator.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        m = engine.state.metrics
        logger.info(
            f"Epoch {trainer.state.epoch} - "
            f"Train Loss: {m['loss']:.4f} | "
            f"Acc Act: {m['acc_action']:.4f} | "
            f"Acc Off: {m['acc_offence']:.4f} | "
            f"Bal Act: {m['bal_acc_action']:.4f} | "
            f"Bal Off: {m['bal_acc_offence']:.4f}"
        )
        if idist.get_rank() == 0:
            print(f"Confusion Matrix Action:\n{m['cm_action']}")
            print(f"Confusion Matrix Offence:\n{m['cm_offence']}")

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        m = engine.state.metrics
        logger.info(
            f"Epoch {trainer.state.epoch} - "
            f"Val Loss: {m['loss']:.4f} | "
            f"Acc Act: {m['acc_action']:.4f} | "
            f"Acc Off: {m['acc_offence']:.4f} | "
            f"Bal Act: {m['bal_acc_action']:.4f} | "
            f"Bal Off: {m['bal_acc_offence']:.4f}"
        )
        if idist.get_rank() == 0:
            print(f"Confusion Matrix Action:\n{m['cm_action']}")
            print(f"Confusion Matrix Offence:\n{m['cm_offence']}")

    # ---- Checkpointing -----------------------------------------------
    
    save_dir = os.path.join(os.getcwd(), "checkpoints")
    print(f"[Rank {local_rank}] Saving checkpoints to {save_dir}...")

    # Unwrap the model for checkpointing to avoid DDP "module." prefix issues
    # both in Checkpoint/Resume and in FastaiLRFinder's state restoration.
    # Unwrap the model for checkpointing to avoid DDP "module." prefix issues.
    raw_model = model.module if hasattr(model, "module") else model

    # Wrapper to handle PEFT partial state dicts (frozen weights missing)
    class PeftCheckpointModel:
        def __init__(self, model):
            self.model = model
        def state_dict(self, *args, **kwargs):
            return self.model.state_dict(*args, **kwargs)
        def load_state_dict(self, state_dict, strict=True):
            return self.model.load_state_dict(state_dict, strict=False)

    to_save = {"model": PeftCheckpointModel(raw_model), "optimizer": optimizer, "trainer": trainer}

    best_handler = ModelCheckpoint(
        dirname=save_dir,
        filename_prefix="best",
        score_name="Loss",
        score_function=lambda engine: -engine.state.metrics["loss"],
        n_saved=1,
        create_dir=True,
        require_empty=False,
        save_on_rank=0,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    evaluator.add_event_handler(Events.COMPLETED, best_handler, to_save)
    if tcfg.checkpoint.store:
        last_handler = ModelCheckpoint(
            dirname=save_dir,
            filename_prefix="last",
            require_empty=False,
            n_saved=1,
            create_dir=True,
            save_on_rank=0,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=eval_every), last_handler, to_save)

    # ---- WandB -------------------------------------------------------
    _setup_wandb(tcfg, cfg, trainer, evaluator, train_evaluator, optimizer)

    # ---- Resume from checkpoint --------------------------------------
    if tcfg.checkpoint.resume and tcfg.checkpoint.resume_from:
        resume_path = hydra.utils.to_absolute_path(tcfg.checkpoint.resume_from)
        if os.path.exists(resume_path):
            print(f"[Rank {local_rank}] Loading checkpoint from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device)

            if tcfg.checkpoint.only_weights:
                # Load only model weights
                print(f"[Rank {local_rank}] Loading ONLY model weights (ignoring optimizer/trainer state)...")
                Checkpoint.load_objects(to_load={"model": to_save["model"]}, checkpoint=checkpoint)
                print(f"[Rank {local_rank}] Weights loaded. Starting fresh training loop.")
            else:
                # Load everything (resume fully)
                Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)
                print(f"[Rank {local_rank}] Checkpoint loaded. "
                      f"Resuming from epoch {trainer.state.epoch}.")
        else:
            print(f"[Rank {local_rank}] Checkpoint path {resume_path} "
                  "not found! Starting from scratch.")

    # ---- Start training ----------------------------------------------
    if idist.get_rank() == 0 and not tcfg.fastai_lr.enable:
        plot_lr_schedule_with_phases(tcfg, steps_per_epoch, save_dir=os.getcwd())

    print(f"[Rank {local_rank}] Starting trainer.run()...")
    if tcfg.fastai_lr.enable:
        lr_finder = FastaiLRFinder()
        with lr_finder.attach(trainer, to_save, output_transform=lambda x:x["loss"], start_lr=1e-6, end_lr=1e-1) as trainer_with_finder:
            trainer_with_finder.run(train_loader)
        ax = lr_finder.plot()
        ax.set_title("FastAI LR Finder")
        fig_path = os.path.join(os.getcwd(), "lr_finder_plot.png")
        ax.figure.savefig(fig_path)
        print(lr_finder.lr_suggestion())

    trainer.run(train_loader, max_epochs=tcfg.epochs)


# ------------------------------------------------------------------ #
# Hydra entry-point                                                    #
# ------------------------------------------------------------------ #

@hydra.main(version_base=None, config_path="../../cfg", config_name="first")
def main(cfg: DictConfig):
    backend = "nccl"

    if "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend)
        try:
            run_training(local_rank, cfg)
        finally:
            dist.destroy_process_group()
    else:
        print("Use torchrun for multi-GPU")
        exit(1)


if __name__ == "__main__":
    main()
