import torch.nn as nn
import torch
from omegaconf import DictConfig
from ultralytics import RTDETR
import hydra
from torchvision.io import read_video
from transformers import AutoVideoProcessor

class Tracker(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.max_track_tokens = cfg.max_track_tokens
        self.hpatches = cfg.hpatches
        self.wpatches = cfg.wpatches
        self.num_patches = self.hpatches * self.wpatches
        self.embedding_dim = cfg.embedding_dim
        self.num_frames = cfg.num_frames
        self.positional_embedding = nn.Parameter(torch.randn(self.num_patches, self.embedding_dim))
        self.tracker: RTDETR = RTDETR(cfg.track_model)
        self.tracker.overrides["verbose"] = False

    def train(self, mode: bool = True):
        # Skip calling .train() on the Ultralytics RTDETR model,
        # which overrides .train() to launch a training procedure.
        self.training = mode
        for name, module in self._modules.items():
            if name == "tracker":
                continue
            module.train(mode)
        return self

    def forward(self, x: torch.Tensor):
        B, T, H, W, C = x.shape
        out = torch.zeros(B, self.num_frames, self.max_track_tokens, self.embedding_dim, device=x.device)
        all_tracks = None
        all_frames = []
        for b in range(B):
            tracks, frames = self._find_tracks(x[b])
            all_frames.append(frames)
            # Contract: at track t, the number is either the patch index (0 to num_patches-1, t = i*self.hpatches + j) or -1 for padding
            if tracks.ndim == 2:
                tracks = tracks.unsqueeze(0)  # Add batch dimension if missing
            all_tracks = tracks if all_tracks is None else torch.cat((all_tracks, tracks), dim=0)

            valid_mask = tracks >= 0
            valid_tracks = tracks[valid_mask]
            assert valid_tracks.max() < self.num_patches, f"Track index {valid_tracks.max()} is out of bounds for {self.num_patches} patches"
            out[b, valid_mask.squeeze()] = self.positional_embedding[valid_tracks.long()]
            
        return out, all_tracks, all_frames

    def _find_tracks(self, x: torch.Tensor):
        T, C, H, W = x.shape
        results = self.tracker.track(
                source=x,
                tracker="bytetrack.yaml",
                persist=True,
                stream=True,
                verbose=False,
            )
        centers = []
        track_first = {}
        track_last = {}

        for frame_idx, r in enumerate(results):
            boxes = r.boxes
            if boxes.id is None:
                continue

            xywhn = boxes.xywhn.cpu().tolist()

            ids = boxes.id.int().cpu().tolist()

            for tid, (cxn, cyn, w_n, h_n) in zip(ids, xywhn):
                tid = int(tid)

                # lifetime
                if tid not in track_first:
                    track_first[tid] = frame_idx
                track_last[tid] = frame_idx

                # normalized center directly
                w_id = int(cxn*self.wpatches) if int(cxn*self.wpatches) < self.wpatches else self.wpatches - 1
                assert w_id < self.wpatches, f"Calculated patch index {w_id} is out of bounds for {self.wpatches} patches"
                h_id = int(cyn*self.hpatches) if int(cyn*self.hpatches) < self.hpatches else self.hpatches - 1
                assert h_id < self.hpatches, f"Calculated patch index {h_id} is out of bounds for {self.hpatches} patches"

                centers.append((frame_idx, tid, h_id * self.wpatches + w_id))

        lifetimes = {
            tid: track_last[tid] - track_first[tid] + 1
            for tid in track_first.keys()
        }

        # sort by lifetime descending
        longest = sorted(lifetimes.items(), key=lambda kv: kv[1], reverse=True)

        
        centers_tid = dict()
        early_longest = 0
        latest_longest = len(centers)


        longest_trace = longest[0][1] if longest else 0
        for tid, length in longest[:self.max_track_tokens]:
            if track_first[tid] > early_longest and track_first[tid] < (T-self.num_frames)//2:
                early_longest = track_first[tid]
            if track_last[tid] < latest_longest and length > longest_trace//3 and track_last[tid] > (T + self.num_frames)//2:
                latest_longest = track_last[tid]

        if latest_longest - early_longest < self.num_frames*1.5:
            if T - latest_longest < early_longest:
                early_longest = max(0, latest_longest - self.num_frames*1.5)
            if latest_longest - early_longest < self.num_frames*1.5:
                latest_longest = min(T-1, early_longest + self.num_frames*1.5)
        else:
            early_longest = max(0, latest_longest - self.num_frames*1.5)

        centers_tid = dict()

        while len(centers_tid) <= self.num_frames:
            centers_tid = dict()
            centers_limit=[center for center in centers if center[0] >= early_longest and center[0] <= latest_longest]
            for tid, length in longest[:self.max_track_tokens]:
                for frame_idx, t, pid in centers_limit:
                    if tid == t:
                        centers_tid.setdefault(frame_idx, []).append((tid, pid))
            early_longest = max(0, early_longest-1)
            latest_longest = min(T-1, latest_longest+1)
        if len(centers_tid) <= self.num_frames:
            raise ValueError(f"Could not find enough tracks within the frame limit. Found {len(centers_tid)} frames with tracks, but need at least {self.num_frames}. Consider adjusting the frame limit or tracking parameters.")
        
        for frame_idx, tracks in centers_tid.items():
            centers_tid[frame_idx] = []
            for i, (tid, pid) in enumerate(sorted(tracks, key=lambda x: x[0])):
                centers_tid[frame_idx].append((i, pid))
        
        
        frames = sorted(list(centers_tid.keys()))
        if len(frames) > self.num_frames:
            # evenly spaced frames
            indices = torch.linspace(0, len(frames) - 1, self.num_frames).long()
            frames = [frames[i] for i in indices]
            
        result_tensor = torch.ones(len(frames), self.max_track_tokens, device=x.device) * -1
        
        for frame_idx in frames:
            for (idx, pid) in centers_tid[frame_idx]:
                assert pid < self.num_patches, f"Patch index {pid} is out of bounds for {self.num_patches} patches"
                result_tensor[frames.index(frame_idx), idx] = pid
        
        return result_tensor, frames
