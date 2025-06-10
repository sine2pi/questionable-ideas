

class ForceDirectedAttention(nn.Module):
    """
    Physics-inspired attention where tokens exert forces on each other.
    
    Each token emits force vectors and has receptivity to forces, with direction-dependent
    strength modulation.
    """
    def __init__(self, dims, heads):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads

        # Each token learns how to emit force vectors
        self.force_emitter = nn.Linear(dims, dims)

        # Each token learns its response to incoming forces
        self.force_receptor = nn.Linear(dims, dims)

        # Direction-dependent strength modulation
        self.direction_modulator = nn.Parameter(torch.randn(heads, dims))

        # Standard projections
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)

    def forward(self, x, mask=None):
        batch, seq_len = x.shape[:2]

        # Calculate emission and reception properties
        emissions = self.force_emitter(x)
        receptivity = self.force_receptor(x)

        # Standard projections
        v = self.v_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Calculate force vectors between token pairs
        token_i = emissions.unsqueeze(2)  # [batch, seq, 1, dim]
        token_j = receptivity.unsqueeze(1)  # [batch, 1, seq, dim]

        # Force direction vector (not just magnitude)
        force_directions = token_i - token_j  # [batch, seq, seq, dim]
        force_magnitudes = torch.norm(force_directions, dim=-1, keepdim=True)
        normalized_forces = force_directions / (force_magnitudes + 1e-8)

        # Direction-dependent strength
        direction_scores = torch.zeros(batch, self.heads, seq_len, seq_len, device=x.device)
        
        for h in range(self.heads):
            head_modulator = self.direction_modulator[h].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            dot_products = normalized_forces * head_modulator
            head_scores = dot_products.sum(dim=-1)
            direction_scores[:, h] = head_scores

        # Reshape force_magnitudes for broadcasting
        broadcast_magnitudes = force_magnitudes.squeeze(-1).unsqueeze(1)
        
        # Combined force effect (analogous to attention scores)
        force_field = direction_scores * torch.exp(-broadcast_magnitudes)
        
        if mask is not None:
            # Expand mask for heads dimension if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            force_field = force_field + mask

        # Convert force field to attention weights
        weights = F.softmax(force_field, dim=-1)

        # Apply weights to values
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)

        return self.output(output)
