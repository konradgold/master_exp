from typing import Tuple
from transformers import AutoImageProcessor, AutoModel
import torch
import torch.nn.functional as F
from difftopk import topk, get_sorting_network

from torch import nn

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


def lapsum_topk(logits, k):  # (B,V,T) → probs sum=k, sparse
    logits = logits.clone()
    for _ in range(20):  # Iterative solve (PAV-like)
        probs = torch.sigmoid(logits)
        shift = torch.log(k / probs.sum(-1, keepdim=True) - 1)
        logits = logits + shift
    return torch.sigmoid(logits)


class SimpleFrameSelector(nn.Module):
    def __init__(self, num_frames: int, num_queries: int = 1, num_probes: int = 4, embed_dim: int = 768):
        super().__init__()
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.num_probes = num_probes
        self.embed_dim = embed_dim
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m")  # Freeze the processor
        self.embedding_model = AutoModel.from_pretrained("facebook/dinov3-convnext-tiny-pretrain-lvd1689m").requires_grad_(False)  # Freeze the embedding model
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim)
        
        
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.scale = embed_dim ** -0.5

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        shapes = videos.shape
        B, V, T, C, H, W = shapes
        K: int = self.num_frames
        assert T >= self.num_frames, f"Number of frames {T} is less than required {K}"

        # 1. Select Probes
        probe_idxs = torch.linspace(0, T-1, self.num_probes, device=videos.device).long()
        probes = videos[:, :, probe_idxs] # (B, V, num_probes, C, H, W)

        scores = self._encode_probes(probes, shapes)  # (B, V, num_probes) 

        interp_scores = self.interpolate_scores(scores, probe_idxs, shapes)  # (B, V, T)

        _, topk_weights = topk(get_sorting_network("bitonic", n=T, k=K, device=videos.device), vectors=interp_scores.view(-1, T), k=K)

        topk_weights = topk_weights.view(B, V, T, K) 

        topk_weights = F.normalize(topk_weights, p=1, dim=-1).permute(0, 1, 3, 2) # (B, V, K, T)
        soft_selected = torch.einsum('bvtchw,bvkt->bvkchw', videos.float(), topk_weights)

        with torch.no_grad():
            idx = topk_weights.argmax(dim=-1)  # (B,V,K), time index per slot
            idx_exp = idx.view(B, V, K, 1, 1, 1).expand(-1, -1, -1, C, H, W)
            hard_selected = torch.gather(videos, dim=2, index=idx_exp)

        selected_frames = hard_selected.detach() + (soft_selected - soft_selected.detach())

        return selected_frames, interp_scores # (B, V, K, C, H, W)

    def _encode_probes(self, probes: torch.Tensor, shapes: Tuple[int, int, int]) -> torch.Tensor:
        B, V, T, C, H, W = shapes
        flat_probes = probes.view(B * V * self.num_probes, C, H, W)
        encoded_probes = self.embedding_model(**self.processor(images=flat_probes, return_tensors="pt").to(probes.device)).pooler_output.reshape(B*V, self.num_probes, -1) # (B*V,num_probes, embed_dim)
    

        q = self.q(self.queries)  # (1, num_queries, embed_dim)
        k = self.kv(encoded_probes) # (B*V, num_probes, embed_dim)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B*V, num_queries, num_probes)
        att = att.softmax(dim=-1)  # (B*V, num_queries, num_probes)
        return att.sum(dim=1).view(B, V, self.num_probes)

    def interpolate_scores(self, scores, probe_idxs: torch.Tensor, shapes):
        B, V, T, C, H, W = shapes
        # probe_idxs: (num_probes,) on same device
        # scores: (B, V, num_probes) on same device

        # 1. Prepare x (t) and y
        # t: (num_probes,) -> broadcast later
        t = probe_idxs.float()                # (P,)
        # y: (B, V, P, 1) -> "channels" dim = 1
        y = scores.unsqueeze(-1)              # (B, V, P, 1)

        # 2. Compute spline coeffs for each (B,V)
        # torchcubicspline expects (..., P, C) with t shape (P,)
        coeffs = natural_cubic_spline_coeffs(t, y)  # coeffs has batch dims (B, V, ..., C)

        spline = NaturalCubicSpline(coeffs)

        # 3. Evaluate at all integer times 0..T-1
        full_idxs = torch.arange(T, device=scores.device).float()  # (T,)               # (B, V, T)

        interp = spline.evaluate(full_idxs)   # (B, V, T, 1)
        interp_scores = interp.squeeze(-1)  # (B, V, T)

        return interp_scores
    
    def assert_correctness(self, selected_frames, videos, interp_scores):
        # Check that selected frames are a convex combination of input frames
        B, V, K, C, H, W = selected_frames.shape
        T = videos.shape[2]

        idx = torch.argsort(interp_scores, dim=-1, descending=True)[:, :, :self.num_frames]  # (B, V, K)
        faults = 0
        mean_fault_distance = 0.0
        for b in range(B):
            for v in range(V):
                selected_idxs = idx[b, v]  # (K,)
                assert torch.all(selected_idxs < T), f"Selected index out of bounds: {selected_idxs} >= {T}"
                selected_frames_bv = selected_frames[b, v]  # (K, C, H, W)
                videos_bv = videos[b, v]  # (T, C, H, W)
                for k in range(K):
                    frame_k = selected_frames_bv[k]  # (C, H, W)
                    idx_k = selected_idxs[k]
                    video_frame_k = videos_bv[idx_k]  # (C, H, W)
                    if not torch.allclose(frame_k, video_frame_k, atol=1e-5):
                        faults += 1
                        mean_fault_distance += torch.norm(frame_k - video_frame_k).item()
        if faults > 0:
            print(f"Warning: {faults} selected frames do not match the corresponding video frames based on interp scores. This may indicate an issue with the selection process.")
            print(f"Mean fault distance: {mean_fault_distance / faults}")
        
if __name__ == "__main__":
  # Disable gradients for testing:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selector = SimpleFrameSelector(num_frames=4, num_queries=1, num_probes=4).to(device).train()
    videos = torch.randn(2, 3, 16, 3, 224, 224, device=device)  # (B, V, T, C, H, W)
    selected_frames,_ = selector(videos)
    print(selected_frames.shape)
    assert selected_frames.shape == (2, 3, 4, 3, 224, 224), "Output shape mismatch"

    truth = torch.randn(2, 3, 4, 3, 224, 224, device=device)

    mse = nn.MSELoss()(truth, selected_frames)
    print(f"MSE Loss: {mse.item()}")

    mse.backward()  # Check if gradients can be computed without error
    print("Backward pass successful, gradients computed.")





