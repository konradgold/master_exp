import hydra
from omegaconf import DictConfig
from torch import nn
import torch
import warnings

import torchvision

from tracker.basic_tracker import Tracker
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from torchvision.io.video import read_video
from transformers import AutoModel, AutoVideoProcessor
from classification.attention_pooling import AttentivePooler
from classification.mvfoulhead import EmbedMVAggregate
from peft import LoraConfig, get_peft_model
from tracker.modules import TrackTention

class VJEPATracker(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.tracker = Tracker(cfg)
        self.tracktention = TrackTention(cfg)
        self.processor = AutoVideoProcessor.from_pretrained(cfg.model_hf)
        self.embedding = AutoModel.from_pretrained(cfg.model_hf).encoder

        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            init_lora_weights=cfg.lora.init_weights,
            target_modules=["query", "value"]
        )
        self.embedding = get_peft_model(self.embedding, lora_config)

        self.num_frames = cfg.num_frames
        self.wpatches = cfg.wpatches
        self.hpatches = cfg.hpatches
        self.max_track_tokens = cfg.max_track_tokens

        self.aggregation = AttentivePooler(cfg.modules)
        self.classifier = EmbedMVAggregate(agr_type=cfg.agr_type, feat_dim=cfg.embedding_dim, return_attention=cfg.return_attention)

        self._use_lora = hasattr(self.embedding, 'peft_config')
        self._phase = cfg.training.phase

        # Permanently freeze pretrained weights; LoRA adapters start
        # frozen or unfrozen depending on the initial phase.
        self._freeze_pretrained()
        self._set_backbone_grad()

    # ------------------------------------------------------------------
    # Freeze / phase helpers
    # ------------------------------------------------------------------

    def _freeze_pretrained(self):
        """Freeze the original (non-LoRA) encoder weights, the RTDETR
        detector, and the processor.  The Tracker's learnable
        ``positional_embedding`` is kept trainable."""
        # Freeze original V-JEPA encoder weights (LoRA params are handled
        # separately by _set_backbone_grad).
        for name, p in self.embedding.named_parameters():
            if "lora_" in name:
                continue            # handled by phase logic
            p.requires_grad = False

        # Freeze RTDETR detector inside Tracker, keep positional_embedding
        for name, p in self.tracker.named_parameters():
            if name == "positional_embedding":
                continue
            p.requires_grad = False

    def _set_backbone_grad(self):
        """Toggle LoRA adapter parameters based on current phase.

        * ``train``  – head only, LoRA adapters frozen.
        * ``pretrain`` – LoRA adapters trainable (backbone fine-tuning).
        """
        lora_trainable = self._phase == "pretrain"

        if self._use_lora:
            lora_params = 0
            for name, p in self.embedding.named_parameters():
                if "lora_" in name:
                    p.requires_grad = lora_trainable
                    lora_params += p.numel()
            state = "TRAINABLE" if lora_trainable else "FROZEN"
            print(f"[VJEPATracker] LoRA adapters {state} "
                  f"({lora_params:,} params)")

        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters()
                     if not p.requires_grad)
        print(f"[VJEPATracker] Frozen: {frozen:,}  |  Trainable: {trainable:,}")

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        assert phase in ("pretrain", "train", "test")
        self._phase = phase
        self._set_backbone_grad()

    def train(self, mode: bool = True):
        # Skip calling .train() on the Ultralytics RTDETR model,
        # which overrides .train() to launch a training procedure.
        self.training = mode
        for name, module in self._modules.items():
            if name in ("processor", "embedding", "tracker"):
                # Don't call module.train() (RTDETR would start its own
                # training loop) and don't toggle requires_grad — it was
                # already set once in __init__.
                continue
            module.train(mode)
        return self

    def forward(self, x):
        b, v, t, c, h, w = x.shape
        x = x.view(b*v, t, c, h, w)
        x_list = [x[i] for i in range(x.shape[0])]
        vframes = self.processor(videos=x_list, return_tensors="pt")['pixel_values_videos']
        assert vframes.shape[:3] == (b*v, t, c), f"Expected shape {(b*v, t, c, h, w)}, got {vframes.shape}"

        vframes_unscaled = (vframes-vframes.min())/(vframes.max()-vframes.min())
        track_tokens, tracks, frames = self.tracker(vframes_unscaled)
        frames = torch.Tensor(frames).to(vframes.device).long()

        vframes = torch.gather(input=vframes, dim=1, index=frames.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *vframes.shape[2:]))

        embedded = self.embedding(vframes, skip_predictor=True)['last_hidden_state']
        embedded = embedded.view(b*v, self.num_frames//2, self.wpatches, self.hpatches, -1)
        track_tokens = track_tokens[:,::2].view(b*v, self.num_frames//2, self.max_track_tokens, -1)
        tracks: torch.Tensor = tracks[:,::2]

        track_tokens, tokens = self.tracktention(track_tokens, tracks, embedded)
        aggregated = self.aggregation(tokens.view(b, -1, tokens.shape[-1]))
        pred_action, pred_offence_severity = self.classifier(aggregated)
        assert pred_action.shape[0] == b, f"Expected batch size {b} for pred_action, got {pred_action.shape[0]}"
        assert pred_offence_severity.shape[0] == b, f"Expected batch size {b} for pred_offence_severity, got {pred_offence_severity.shape[0]}"
        return pred_action, pred_offence_severity, track_tokens
    

class MAEFinder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
        self.processor = VideoMAEImageProcessor.from_pretrained(cfg.model_hf, num_labels=cfg.num_classes)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            cfg.model_hf,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        self.num_frames = cfg.num_frames


        self.classifier = EmbedMVAggregate(agr_type=cfg.agr_type, feat_dim=cfg.embedding_dim, return_attention=cfg.return_attention)
        self._phase = cfg.training.phase

    def _set_backbone_grad(self):
        """Freeze backbone in 'train' phase, unfreeze in 'pretrain' phase."""
        freeze = (self._phase == "train")
        for param in self.model.parameters():
            param.requires_grad = not freeze
        if freeze:
            print(f"[MViTracker] Backbone FROZEN ({sum(1 for p in self.model.parameters()):,} params)")
        else:
            print(f"[MAEFinder] Backbone UNFROZEN (all params trainable)")

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str = "train"):
        assert phase in ["pretrain", "train", "test"]
        self._phase = phase
        self._set_backbone_grad()

    def train(self, mode: bool = True):
        # Skip calling .train() on the Ultralytics RTDETR model,
        # which overrides .train() to launch a training procedure.
        self.training = mode
        for name, module in self._modules.items():
            if name == "model" and self.phase == "train":
                continue
            module.train(mode)
        return self


class MAETracker(nn.Module):
    """VideoMAE backbone with TrackTention blocks interleaved at
    configurable layer indices.

    Instead of applying TrackTention only after the full encoder,
    this model injects TrackTention blocks between specific encoder
    layers so that spatial track information is fused iteratively
    throughout the backbone.

    Config fields (on top of standard fields):
        tracktention_layer_indices: list[int]
            Layer indices (0-based) *after* which to insert a
            TrackTention block.  E.g. ``[3, 7, 11]`` for a 12-layer
            VideoMAE inserts TrackTention after layers 3, 7, and 11.
        tubelet_size: int
            Temporal tubelet size of the VideoMAE patch embedding.
            Needed to compute ``T_tubes = num_frames / tubelet_size``.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        from transformers import VideoMAEVideoProcessor, VideoMAEModel

        # ---- backbone ------------------------------------------------
        self.processor = VideoMAEVideoProcessor.from_pretrained(cfg.model_hf)
        self.encoder = VideoMAEModel.from_pretrained(cfg.model_hf)

        # Optional: apply LoRA to the encoder
        

        # ---- tracker -------------------------------------------------
        self.tracker = Tracker(cfg)

        # ---- geometry constants --------------------------------------
        self.num_frames = cfg.num_frames
        self.tubelet_size = cfg.tubelet_size
        self.t_tubes = self.num_frames // self.tubelet_size
        self.hpatches = self.encoder.config.image_size // self.encoder.config.patch_size
        self.wpatches = self.encoder.config.image_size // self.encoder.config.patch_size
        self.max_track_tokens = cfg.max_track_tokens
        self.embedding_dim = cfg.embedding_dim

        if cfg.lora.enable:
            lora_config = LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                init_lora_weights=cfg.lora.init_weights,
                target_modules=["query", "value"],
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
        self._use_lora = hasattr(self.encoder, "peft_config")

        # ---- interleaved TrackTention blocks -------------------------
        self.tracktention_indices = sorted(cfg.tracktention_layer_indices)
        self.tracktention_blocks = nn.ModuleDict()
        for idx in self.tracktention_indices:
            self.tracktention_blocks[str(idx)] = TrackTention(cfg)

        # ---- classification head -------------------------------------
        self.aggregation = AttentivePooler(cfg.modules)
        self.classifier = EmbedMVAggregate(
            agr_type=cfg.agr_type,
            feat_dim=cfg.embedding_dim,
            return_attention=cfg.return_attention,
        )

        self._phase = cfg.training.phase
        self._freeze_pretrained()
        self._set_backbone_grad()

    # ------------------------------------------------------------------
    # Freeze / phase helpers
    # ------------------------------------------------------------------

    def _freeze_pretrained(self):
        """Freeze non-LoRA encoder weights and the RTDETR detector."""
        for name, p in self.encoder.named_parameters():
            if "lora_" in name:
                continue
            p.requires_grad = False
        for name, p in self.tracker.named_parameters():
            if name == "positional_embedding":
                continue
            p.requires_grad = False

    def _set_backbone_grad(self):
        lora_trainable = self._phase == "pretrain"
        if self._use_lora:
            lora_params = 0
            for name, p in self.encoder.named_parameters():
                if "lora_" in name:
                    p.requires_grad = lora_trainable
                    lora_params += p.numel()
            state = "TRAINABLE" if lora_trainable else "FROZEN"
            print(f"[MAETracker] LoRA adapters {state} ({lora_params:,} params)")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"[MAETracker] Frozen: {frozen:,}  |  Trainable: {trainable:,}")

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str):
        assert phase in ("pretrain", "train", "test")
        self._phase = phase
        self._set_backbone_grad()

    def train(self, mode: bool = True):
        self.training = mode
        for name, module in self._modules.items():
            if name in ("processor", "encoder", "tracker"):
                continue
            module.train(mode)
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _interleaved_encode(self, pixel_values, track_tokens, tracks):
        """Run the VideoMAE encoder with TrackTention injected after
        specified layers.

        Args:
            pixel_values: preprocessed video tensor for VideoMAE.
            track_tokens: ``(B, T_tubes, M, D)`` from the Tracker.
            tracks: ``(B, T_tubes, M)`` patch-index positions.

        Returns:
            hidden_states: ``(B, T_tubes, H, W, D)`` final spatial map.
            track_tokens: ``(B, T_tubes, M, D)`` updated track tokens.
        """
        # Get the underlying encoder (handles PeftModel wrapping)
        base_encoder = self.encoder
        if hasattr(base_encoder, "base_model"):
            base_encoder = base_encoder.base_model
        if hasattr(base_encoder, "model"):
            base_encoder = base_encoder.model

        # Run the VideoMAE patch + positional embedding
        # bool_masked_pos=None disables masking (inference / fine-tuning mode)
        embedding_output = base_encoder.embeddings(pixel_values, bool_masked_pos=None)
        hidden_states = embedding_output

        # Get encoder layers
        encoder = base_encoder.encoder
        layers = encoder.layer

        BV = hidden_states.shape[0]

        for i, layer_module in enumerate(layers):
            hidden_states = layer_module(hidden_states)

            if i in self.tracktention_indices:
                # Reshape (BV, N, D) → (BV, T_tubes, H, W, D)
                hidden_states = hidden_states.view(
                    BV, self.t_tubes, self.hpatches, self.wpatches, self.embedding_dim
                )
                track_tokens, hidden_states = self.tracktention_blocks[str(i)](
                    track_tokens, tracks, hidden_states
                )
                # Flatten back to (BV, N, D) for next encoder layer
                hidden_states = hidden_states.reshape(BV, -1, self.embedding_dim)

        # Apply final layer norm if present
        if hasattr(encoder, "layernorm") and encoder.layernorm is not None:
            hidden_states = encoder.layernorm(hidden_states)
        elif hasattr(base_encoder, "layernorm") and base_encoder.layernorm is not None:
            hidden_states = base_encoder.layernorm(hidden_states)

        # Final reshape to spatial
        hidden_states = hidden_states.view(
            BV, self.t_tubes, self.hpatches, self.wpatches, self.embedding_dim
        )
        return hidden_states, track_tokens

    def forward(self, x):
        b, v, t, c, h, w = x.shape
        x = x.view(b * v, t, c, h, w)

        # Preprocess for VideoMAE
        x_list = [x[i] for i in range(x.shape[0])]
        vframes = self.processor(
            videos=x_list, return_tensors="pt" 
        )["pixel_values"].to(x.device)



        # Run tracker on normalised frames to get tracks
        vframes_unscaled = (vframes - vframes.min()) / (vframes.max() - vframes.min())
        track_tokens, tracks, frames = self.tracker(vframes_unscaled)
        frames = torch.tensor(frames, device=vframes.device, dtype=torch.long)

        # Select tracked frames from the preprocessed video
        vframes = torch.gather(
            vframes, 1,
            frames.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                  .expand(-1, -1, *vframes.shape[2:]),
        )

        # Subsample to tubelet temporal resolution
        # track_tokens: (BV, num_frames, M, D) → (BV, T_tubes, M, D)
        track_tokens = track_tokens[:, :: self.tubelet_size]
        track_tokens = track_tokens.view(
            b * v, self.t_tubes, self.max_track_tokens, -1
        )
        tracks = tracks[:, :: self.tubelet_size]

        # Run interleaved encoder
        tokens, track_tokens = self._interleaved_encode(
            vframes, track_tokens, tracks
        )

        # Aggregate & classify
        aggregated = self.aggregation(tokens.reshape(b, -1, tokens.shape[-1]))
        pred_action, pred_offence_severity = self.classifier(aggregated)
        assert pred_action.shape[0] == b
        assert pred_offence_severity.shape[0] == b
        return pred_action, pred_offence_severity, track_tokens


class MViTClassifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        weights = torchvision.models.video.MViT_V2_S_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.num_frames = cfg.num_frames

        self.model = torchvision.models.video.mvit_v2_s(weights=weights)
        self.model.head = nn.Identity()
        self.classifier = EmbedMVAggregate(agr_type=cfg.agr_type, feat_dim=cfg.embedding_dim, return_attention=cfg.return_attention)
        self._phase = cfg.training.phase

        # Freeze/unfreeze backbone based on phase
        self._set_backbone_grad()
    
    def _set_backbone_grad(self):
        """Freeze backbone in 'train' phase, unfreeze in 'pretrain' phase."""
        freeze = (self._phase == "train")
        for param in self.model.parameters():
            param.requires_grad = not freeze
        if freeze:
            print(f"[MViTracker] Backbone FROZEN ({sum(1 for p in self.model.parameters()):,} params)")
        else:
            print(f"[MViTracker] Backbone UNFROZEN (all params trainable)")

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase: str = "train"):
        assert phase in ["pretrain", "train", "test"]
        self._phase = phase
        self._set_backbone_grad()

    def train(self, mode: bool = True):
        # Skip calling .train() on the Ultralytics RTDETR model,
        # which overrides .train() to launch a training procedure.
        self.training = mode
        for name, module in self._modules.items():
            if name == "model" and self.phase == "train":
                continue
            module.train(mode)
        return self

    def forward(self, x):
        b, v, t, c, h, w = x.shape
        x = x.reshape(b*v, t, c, h, w).contiguous().squeeze(0)  # (B*V, T, C, H, W)
        assert x.shape == (b*v, t, c, h, w), f"Expected shape {(b*v, t, c, h, w)}, got {x.shape}"
        if t > self.num_frames:
            x = x[:, torch.arange(0, t, t//self.num_frames)[-self.num_frames:]] # (B*V, num_frames, C, H, W)
        x = self.preprocess(x)
        if x.dim() == 4:
            x = x.unsqueeze(0)
        # Skip gradient computation through backbone when frozen
        if self._phase == "train":
            with torch.no_grad():
                x = self.model(x)
        else:
            x = self.model(x)
        x = x.view(b, v, -1)
        pred_action, pred_offence_severity = self.classifier(x)
        assert pred_action.shape[0] == b, f"Expected batch size {b} for pred_action, got {pred_action.shape[0]}"
        assert pred_offence_severity.shape[0] == b, f"Expected batch size {b} for pred_offence_severity, got {pred_offence_severity.shape[0]}"
        return pred_action, pred_offence_severity, None



# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "vjepa": VJEPATracker,
    "mvit": MViTClassifier,
    "mae": MAEFinder,
    "mae_tracked": MAETracker,
}


def build_model(cfg: DictConfig) -> nn.Module:
    """Instantiate a model by ``cfg.model_type`` using the registry."""
    model_type = cfg.model_type
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](cfg)


@hydra.main(version_base=None, config_path="../cfg", config_name="first")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VJEPATracker(cfg)
    model.to(device)
    vid_path = "/sc/home/konrad.goldenbaum/data/SoccerData/mvfouls/train/action_43/clip_1.mp4"
    frames = read_video(vid_path, start_pts=0, end_pts=None, pts_unit="sec", output_format="TCHW")[0]
    frames = frames.unsqueeze(0)
    frames = torch.cat([frames, frames], dim=0)
    frames = frames.unsqueeze(0)
    frames = frames.to(device)

    pred_action, pred_offence_severity, track_tokens = model(frames)
    print("Pred action shape:", pred_action.shape)
    print("Pred offence severity shape:", pred_offence_severity.shape)
    print("Track tokens shape:", track_tokens.shape)

if __name__ == "__main__":
    main()
