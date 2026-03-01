import torch.nn as nn
import torch
from omegaconf import DictConfig
from positional_encodings.torch_encodings import PositionalEncoding1D
from src.tracker.basic_tracker import Tracker


class AttentionalSampler(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.max_track_tokens = cfg.max_track_tokens
        self.embedding_dim = cfg.embedding_dim
        self.num_heads = cfg.num_heads
        self.hpatches = cfg.hpatches
        self.wpatches = cfg.wpatches
        self.window_decay = cfg.window_decay
        self.use_spda = cfg.use_sdpa

        self.LN = nn.LayerNorm(self.embedding_dim)

        self.q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.k = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, t: torch.Tensor, mv: torch.Tensor, positions: torch.Tensor = None):
        B, T, M, D = t.shape
        B, T, H, W, D = mv.shape
        assert M == self.max_track_tokens, f"Expected {self.max_track_tokens} track tokens, got {M}"
        

        q = self.q(t)
        k = self.k(mv)
        k = k.view(B*T, H, W, D)
        k = self.rope(k, square=True)
        q = self.rope(q, square=False, positions=positions)
        q=q.view(B*T, M, D)
        k = k.view(B*T, H*W, D)

        q = self.LN(q)
        k = self.LN(k)

        att = q@k.transpose(-2,-1)/ (self.embedding_dim ** 0.5)
        bias = self._compute_bias(positions)
        att = torch.softmax(att + bias, dim=-1)
        attn_output = att@mv.view(B*T, H*W,D)
        
        return attn_output.view(B, T, M, D)
    

    def rope(self, x: torch.Tensor, square: bool = False, positions: torch.Tensor = None):
        # Placeholder for ROPE implementation
        if square:
            return self._square_rope(x)
        return self._flat_rope(x, positions)
    
    def _square_rope(self, x: torch.Tensor):
        # Placeholder for square ROPE implementation
        BT, H, W, D = x.shape
        for i in range(H*W):
            x[:, i//W, i%W, :] = self._apply_rope(x[:, i//W, i%W, :], i//W, i%W)
        return x
    
    def _flat_rope(self, x: torch.Tensor, positions: torch.Tensor):
        # Placeholder for flat ROPE implementation
        B, T, M, D = x.shape
        for i in range(B):
            for j in range(T):
                for k in range(M):
                    x[i, j, k] = self._apply_flat_rope(x[i, j, k], positions[i, j, k]//self.wpatches, positions[i, j, k]%self.wpatches)
        return x

    def _apply_rope(self, x: torch.Tensor, h_idx: int, w_idx: int):
        # Placeholder for applying ROPE to a single token
        theta = 100 ** (-4*torch.arange(1, self.embedding_dim//4+1, device=x.device) / self.embedding_dim)
        x = x.view(-1, self.embedding_dim//2, 2)
        x[:, ::2, 0] = x[:, ::2, 0] * torch.cos(theta*h_idx) - x[:, 1::2, 0] * torch.sin(theta*h_idx)
        x[:, 1::2, 0] = x[:, 1::2, 0] * torch.cos(theta*h_idx) - x[:, ::2, 0] * torch.sin(theta*h_idx)
        x[:, ::2, 1] = x[:, ::2, 1] * torch.cos(theta*w_idx) + x[:, 1::2, 1] * torch.sin(theta*w_idx)
        x[:, 1::2, 1] = x[:, 1::2, 1] * torch.cos(theta*w_idx) + x[:, ::2, 1] * torch.sin(theta*w_idx)
        return x.view(-1, self.embedding_dim)
    
    def _apply_flat_rope(self, x: torch.Tensor, h: int, w: int):
        # Placeholder for applying flat ROPE to a single token
        theta = 100 ** (-4*torch.arange(1, self.embedding_dim//4+1, device=x.device) / self.embedding_dim)
        x = x.view(self.embedding_dim//2, 2)
        x[::2, 0] = x[::2, 0] * torch.cos(theta*h) - x[1::2, 0] * torch.sin(theta*h)
        x[1::2, 0] = x[1::2, 0] * torch.cos(theta*h) - x[::2, 0] * torch.sin(theta*h)
        x[::2, 1] = x[::2, 1] * torch.cos(theta*w) + x[1::2, 1] * torch.sin(theta*w)
        x[1::2, 1] = x[1::2, 1] * torch.cos(theta*w) + x[::2, 1] * torch.sin(theta*w)
        return x.view(self.embedding_dim)
        
    
    def _compute_bias(self, positions: torch.Tensor):
        B, T, M = positions.shape
        P = self.hpatches * self.wpatches
        bias = torch.empty(B, T, M, P, device=positions.device, dtype=torch.float32)
        pos_coords = torch.stack((
            positions // self.wpatches,
            positions %  self.wpatches
        ), dim=-1).float()
        p = torch.arange(P, device=positions.device)
        patch_coords = torch.stack((
            p // self.wpatches,
            p %  self.wpatches
        ), dim=-1)
        diff = pos_coords[..., None, :] - patch_coords[None, None, None, :, :]
        dist = torch.linalg.vector_norm(diff, dim=-1) 
        bias = - dist / (2 * self.window_decay ** 2)

        bias[pos_coords[..., 0] < 0] = -1e9
        return bias.view(B*T, M, self.hpatches*self.wpatches)



class AttentionSplatting(AttentionalSampler):

    def forward(self, t: torch.Tensor, mv: torch.Tensor, positions: torch.Tensor):
        B, T, M, D = t.shape
        B, T, H, W, D = mv.shape
        assert M == self.max_track_tokens, f"Expected {self.max_track_tokens} track tokens, got {M}"
        

        q = self.q(mv)
        k = self.k(t)
        q = q.view(B*T, H, W, D)
        k = self.rope(k, square=False,  positions=positions)
        q = self.rope(q, square=True)
        k= k.view(B*T, M, D)
        q = q.view(B*T, H*W, D)

        q = self.LN(q)
        k = self.LN(k)

        att = q@k.transpose(-2,-1)/ (self.embedding_dim ** 0.5)
        bias = self._compute_bias(positions)
        att = torch.softmax(att + bias.transpose(-1,-2), dim=-1)
        attn_output =att@t.reshape(B*T, M, D)
        
        return attn_output.view(B, T, H, W, D)
        
    
class TrackTransformer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.max_track_tokens = cfg.max_track_tokens
        self.embedding_dim = cfg.embedding_dim
        self.num_heads = cfg.num_heads
        self.num_layers = cfg.TrackTransformer.num_layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.num_heads, batch_first=True),
            num_layers=self.num_layers
        )
        self.positional_encoding = PositionalEncoding1D(self.embedding_dim)

    def forward(self, x: torch.Tensor):
        B, T, M, D = x.shape
        assert M == self.max_track_tokens, f"Expected {self.max_track_tokens} track tokens, got {M}"
        x = x.permute(0, 2, 1, 3).reshape(B*M, T, D)  # (B * M, T, D)
        x = x + self.positional_encoding(x)
        x = self.transformer(src=x)

        x = x.view(B, M, T, D).permute(0, 2, 1, 3)  # (B, T, M, D)
        return x

class TrackTention(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.attentional_sampler = AttentionalSampler(cfg)
        self.attention_splatting = AttentionSplatting(cfg)
        self.transformer = TrackTransformer(cfg)


    def forward(self, track_tokens, tracks, embedded):

        track_tokens = self.attentional_sampler(track_tokens, embedded, tracks)
        track_tokens = self.transformer(track_tokens)
        embedded = self.attention_splatting(track_tokens, embedded, tracks)

        return track_tokens, embedded


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("cfg/first.yaml")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tt = TrackTention(cfg)
    x = torch.randn(2, 128, 16, 16, 1024)

    # Direct
    direct_tokens, direct_mv = tt(x)

    # Manual using the SAME modules and SAME call order
    tracker = tt.tracker
    at = tt.attentional_sampler
    tf = tt.transformer
    att = tt.attention_splatting

    track_tokens, tracks, frames = tracker(x)
    mv = torch.gather(
        x, 1,
        frames.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            .expand(-1, -1, *x.shape[2:])
    )
    track_tokens = at(track_tokens, mv, tracks)
    track_tokens = tf(track_tokens)
    mv = att(track_tokens, mv, tracks)

    assert mv.shape == direct_mv.shape, f"Expected mv shape {direct_mv.shape}, got {mv.shape}"
    assert track_tokens.shape == direct_tokens.shape, f"Expected track_tokens shape {direct_tokens.shape}, got {track_tokens.shape}"

