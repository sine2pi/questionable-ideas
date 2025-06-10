import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from einops import rearrange, repeat
from rotary import RotaryEmbedding

class LightForceRotary(nn.Module):
    """
    Simplified force-aware rotary embedding that's efficient for practical use.
    This is a distillation of the gravitational model into a form that:
    1. Has O(n) computation complexity instead of O(n²)
    2. Requires minimal prior knowledge
    3. Can be used as a drop-in replacement for standard RoPE
    4. Retains the core benefit of content-aware positioning
    """
    def __init__(self, dim, theta=10000, learned_freq=False, heads=None,
                 force_factor=0.2, use_quaternion=False, static_init=True):
        """
        Initialize the lightweight force-based rotary embedding.
        Args:
            dim: Dimension of the embedding
            theta: Base value for frequency
            learned_freq: Whether frequencies should be learnable
            heads: Optional number of heads (for per-head scaling)
            force_factor: How much content influences position (0-1)
            use_quaternion: Whether to use quaternion rotation
            static_init: Whether to use static initialization
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.force_factor = force_factor
        self.use_quaternion = use_quaternion
        self.rotary = RotaryEmbedding(
            dim=dim, 
            theta=theta,
            learned_freq=learned_freq,
            use_quaternion=use_quaternion
        )
        self.content_mapper = nn.Linear(dim, 1)
        if heads is not None:
            self.head_factors = nn.Parameter(torch.ones(heads))
        if static_init and not learned_freq:
            nn.init.zeros_(self.content_mapper.bias)
            nn.init.orthogonal_(self.content_mapper.weight, gain=0.1)
    def forward(self, x, seq_dim=1):
        """
        Apply lightweight force-rotary embedding.
        Args:
            x: Input tensor [batch, seq_len, dim] or [batch, heads, seq_len, head_dim]
            seq_dim: Dimension containing sequence (default: 1)
        Returns:
            Rotated tensor with same shape as input
        """
        orig_shape = x.shape
        device = x.device
        if len(orig_shape) == 3:
            batch, seq_len, dim = orig_shape
            is_heads_format = False
            content = x
        else:
            batch, heads, seq_len, head_dim = orig_shape
            is_heads_format = True
            content = x.transpose(1, 2).reshape(batch, seq_len, -1)
        std_positions = torch.arange(seq_len, device=device).float()
        with torch.no_grad():
            offsets = self.content_mapper(content).squeeze(-1)
            offsets = torch.tanh(offsets) * seq_len * 0.1
            offsets = offsets * self.force_factor
        if is_heads_format and self.heads is not None:
            head_scale = self.head_factors.view(1, heads, 1)
            head_offsets = offsets.unsqueeze(1) * head_scale
            result = torch.zeros_like(x)
            for h in range(heads):
                head_positions = std_positions + head_offsets[:, h]
                for b in range(batch):
                    freqs = self.rotary.forward(head_positions[b])
                    result[b, h] = self.rotary.apply_rotary(
                        freqs, x[b, h], seq_dim=-2
                    )
            return result
        else:
            positions = std_positions + offsets
            result = torch.zeros_like(x)
            if self.use_quaternion:
                for b in range(batch):
                    sorted_indices = torch.argsort(positions[b])
                    reverse_indices = torch.argsort(sorted_indices)
                    if is_heads_format:
                        for h in range(heads):
                            x_sorted = x[b, h][sorted_indices]
                            x_rotated = self.rotary.rotate_(x_sorted.unsqueeze(0)).squeeze(0)
                            result[b, h] = x_rotated[reverse_indices]
                    else:
                        x_sorted = x[b][sorted_indices]
                        x_rotated = self.rotary.rotate_(x_sorted.unsqueeze(0)).squeeze(0)
                        result[b] = x_rotated[reverse_indices]
            else:
                for b in range(batch):
                    freqs = self.rotary.forward(positions[b])
                    if is_heads_format:
                        for h in range(heads):
                            result[b, h] = self.rotary.apply_rotary(
                                freqs, x[b, h], seq_dim=-2
                            )
                    else:
                        result[b] = self.rotary.apply_rotary(
                            freqs, x[b], seq_dim=-2
                        )
            return result
def test_light_force_rotary():
    """
    Test function for the lightweight force-aware rotary embedding.
    """
    batch_size = 2
    seq_len = 16
    dims = 64
    x = torch.randn(batch_size, seq_len, dims)
    light_force_rotary = LightForceRotary(
        dim=dims,
        theta=10000,
        learned_freq=False,
        force_factor=0.2
    )
    rotated = light_force_rotary(x)
    assert rotated.shape == x.shape, f"Shape mismatch: {rotated.shape} vs {x.shape}"
    print("Light force-aware rotary embedding test passed!")
    return rotated
if __name__ == "__main__":
    test_light_force_rotary()
