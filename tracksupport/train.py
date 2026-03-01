import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import logging
import os

from src.models import VJEPATracker
from src.datasets.mvfoul import MultiViewDataset

log = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, criterion_action, criterion_offence, device, epoch, cfg):
    model.train()
    running_loss = 0.0
    correct_action = 0
    correct_offence = 0
    total = 0
    all_target_action = []
    all_pred_action = []
    all_target_offence = []
    all_pred_offence = []
    global_step = epoch * len(dataloader)

    for i, (labels_offence, labels_action, videos, n_actions) in enumerate(tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)):
        try:
            videos = videos.to(device)
        except Exception as e:
            log.warning(f"Failed to move videos to device: {e}")
            continue
        labels_action = labels_action.to(device).squeeze()
        labels_offence = labels_offence.to(device).squeeze()
        target_action = labels_action.argmax(dim=1)
        target_offence = labels_offence.argmax(dim=1)


        pred_action, pred_offence, _ = model(videos)

        loss_action = criterion_action(pred_action, labels_action)
        loss_offence = criterion_offence(pred_offence, labels_offence)
        loss = loss_action + loss_offence

        loss = loss / cfg.training.grad_accum_steps
        loss.backward()

        if (i + 1) % cfg.training.grad_accum_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        batch_loss = loss.item() * cfg.training.grad_accum_steps
        running_loss += batch_loss * videos.size(0)

        # accuracy
        pred_action_cls = pred_action.argmax(dim=1)
        pred_offence_cls = pred_offence.argmax(dim=1)
        correct_action += (pred_action_cls == target_action).sum().item()
        correct_offence += (pred_offence_cls == target_offence).sum().item()
        total += videos.size(0)

        all_target_action.append(target_action.cpu())
        all_pred_action.append(pred_action_cls.cpu())
        all_target_offence.append(target_offence.cpu())
        all_pred_offence.append(pred_offence_cls.cpu())

        step = global_step + i
        if cfg.training.wandb.enable and (step % cfg.training.wandb.log_frequency == 0):
            import wandb
            wandb.log({
                "train/loss": batch_loss,
                "train/loss_action": loss_action.item(),
                "train/loss_offence": loss_offence.item(),
                "train/step": step,
            }, step=step)

        if (i + 1) % 10 == 0:
            log.info(f"  [Epoch {epoch}] Step {i+1}/{len(dataloader)} | Loss: {batch_loss:.4f}")

    epoch_loss = running_loss / total
    epoch_acc_action = correct_action / total
    epoch_acc_offence = correct_offence / total

    all_target_action = torch.cat(all_target_action).numpy()
    all_pred_action = torch.cat(all_pred_action).numpy()
    all_target_offence = torch.cat(all_target_offence).numpy()
    all_pred_offence = torch.cat(all_pred_offence).numpy()
    epoch_bal_acc_action = balanced_accuracy_score(all_target_action, all_pred_action)
    epoch_bal_acc_offence = balanced_accuracy_score(all_target_offence, all_pred_offence)

    return epoch_loss, epoch_acc_action, epoch_acc_offence, epoch_bal_acc_action, epoch_bal_acc_offence


@torch.no_grad()
def validate(model, dataloader, criterion_action, criterion_offence, device):
    model.eval()
    running_loss = 0.0
    correct_action = 0
    correct_offence = 0
    total = 0
    all_target_action = []
    all_pred_action = []
    all_target_offence = []
    all_pred_offence = []

    for labels_offence, labels_action, videos, n_actions in tqdm(dataloader, desc="Validation", leave=False):
        videos = videos.to(device)
        labels_action = labels_action.to(device)
        labels_offence = labels_offence.to(device)

        target_action = labels_action.argmax(dim=1)
        target_offence = labels_offence.argmax(dim=1)

        pred_action, pred_offence, _ = model(videos)

        loss_action = criterion_action(pred_action, target_action)
        loss_offence = criterion_offence(pred_offence, target_offence)
        loss = loss_action + loss_offence

        running_loss += loss.item() * videos.size(0)

        pred_action_cls = pred_action.argmax(dim=1)
        pred_offence_cls = pred_offence.argmax(dim=1)
        correct_action += (pred_action_cls == target_action).sum().item()
        correct_offence += (pred_offence_cls == target_offence).sum().item()
        total += videos.size(0)

        all_target_action.append(target_action.cpu())
        all_pred_action.append(pred_action_cls.cpu())
        all_target_offence.append(target_offence.cpu())
        all_pred_offence.append(pred_offence_cls.cpu())

    epoch_loss = running_loss / total
    epoch_acc_action = correct_action / total
    epoch_acc_offence = correct_offence / total

    all_target_action = torch.cat(all_target_action).numpy()
    all_pred_action = torch.cat(all_pred_action).numpy()
    all_target_offence = torch.cat(all_target_offence).numpy()
    all_pred_offence = torch.cat(all_pred_offence).numpy()
    epoch_bal_acc_action = balanced_accuracy_score(all_target_action, all_pred_action)
    epoch_bal_acc_offence = balanced_accuracy_score(all_target_offence, all_pred_offence)

    return epoch_loss, epoch_acc_action, epoch_acc_offence, epoch_bal_acc_action, epoch_bal_acc_offence


@hydra.main(version_base=None, config_path="cfg", config_name="first")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ---- wandb ----
    if cfg.training.wandb.enable:
        import wandb
        wandb.init(
            project=cfg.training.wandb.project,
            entity=cfg.training.wandb.entity,
            name=cfg.training.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ---- datasets ----
    tcfg = cfg.training
    train_dataset = MultiViewDataset(
        path=tcfg.data_path,
        start=tcfg.start_frame,
        end=tcfg.end_frame,
        fps=tcfg.fps,
        split="train",
        num_views=tcfg.num_views,
        num_frames=cfg.num_frames,
    )
    val_dataset = MultiViewDataset(
        path=tcfg.data_path,
        start=tcfg.start_frame,
        end=tcfg.end_frame,
        fps=tcfg.fps,
        split="val",
        num_views=tcfg.num_views,
        num_frames=cfg.num_frames,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=tcfg.train_batch_size,
        shuffle=True,
        num_workers=tcfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tcfg.val_batch_size,  # must be 1 for validation
        shuffle=False,
        num_workers=tcfg.num_workers,
        pin_memory=True,
    )

    # ---- model ----
    model = VJEPATracker(cfg)
    model.to(device)

    # freeze the pretrained video encoder
    for param in model.embedding.parameters():
        param.requires_grad = False

    # ---- optimizer & loss ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
    )

    # use class weights from dataset for balanced CE
    w_action = train_dataset.weights_action.to(device).squeeze()
    w_offence = train_dataset.weights_offence_severity.to(device).squeeze()
    criterion_action = nn.CrossEntropyLoss(weight=w_action, label_smoothing=tcfg.label_smoothing)
    criterion_offence = nn.CrossEntropyLoss(weight=w_offence, label_smoothing=tcfg.label_smoothing)

    # ---- training loop ----
    best_val_loss = float("inf")
    save_dir = os.getcwd()  # hydra changes cwd to output dir

    for epoch in range(tcfg.epochs):
        log.info(f"===== Epoch {epoch+1}/{tcfg.epochs} =====")

        train_loss, train_acc_action, train_acc_offence, train_bal_acc_action, train_bal_acc_offence = train_one_epoch(
            model, train_loader, optimizer, criterion_action, criterion_offence, device, epoch, cfg
        )
        log.info(
            f"Train | Loss: {train_loss:.4f} | Action Acc: {train_acc_action:.4f} | Offence Acc: {train_acc_offence:.4f}"
            f" | Action Bal Acc: {train_bal_acc_action:.4f} | Offence Bal Acc: {train_bal_acc_offence:.4f}"
        )

        val_loss, val_acc_action, val_acc_offence, val_bal_acc_action, val_bal_acc_offence = validate(
            model, val_loader, criterion_action, criterion_offence, device
        )
        log.info(
            f"Val   | Loss: {val_loss:.4f} | Action Acc: {val_acc_action:.4f} | Offence Acc: {val_acc_offence:.4f}"
            f" | Action Bal Acc: {val_bal_acc_action:.4f} | Offence Bal Acc: {val_bal_acc_offence:.4f}"
        )

        if cfg.training.wandb.enable:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "train/epoch_acc_action": train_acc_action,
                "train/epoch_acc_offence": train_acc_offence,
                "train/epoch_bal_acc_action": train_bal_acc_action,
                "train/epoch_bal_acc_offence": train_bal_acc_offence,
                "val/epoch_loss": val_loss,
                "val/epoch_acc_action": val_acc_action,
                "val/epoch_acc_offence": val_acc_offence,
                "val/epoch_bal_acc_action": val_bal_acc_action,
                "val/epoch_bal_acc_offence": val_bal_acc_offence,
            }, step=(epoch + 1) * len(train_loader))

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc_action": val_acc_action,
                "val_acc_offence": val_acc_offence,
            }, ckpt_path)
            log.info(f"Saved best model to {ckpt_path} (val_loss={val_loss:.4f})")

    # save final model
    torch.save({
        "epoch": tcfg.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(save_dir, "last_model.pt"))
    log.info("Training complete.")

    if cfg.training.wandb.enable:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
