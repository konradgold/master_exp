
import torch
from einops import rearrange

B, T, C, H, W = 2, 16, 3, 8, 8
patch_size = 2
x = torch.randn(B, T, C, H, W)

proj = torch.nn.Linear(C * patch_size * patch_size, 7, bias=False)
torch.manual_seed(0)
proj.weight = torch.nn.init.xavier_uniform_(proj.weight)

x = rearrange(x, 'b t c (nh ph) (nw pw) -> b t (nh nw) (ph pw c)', ph=patch_size, pw=patch_size)
print(x.shape)  # Should print torch.Size([2, 16, 16, 12])

x_n = proj(x)

matrix = proj.weight.data
print(matrix.shape)  # Should print torch.Size([7, 12])
matrix[torch.pow(matrix, 2) < 0.01] = 0.
print((matrix==0.).sum())  # Check how many zeros



print(x.shape)  # Should print torch.Size([2, 16, 16, 7])