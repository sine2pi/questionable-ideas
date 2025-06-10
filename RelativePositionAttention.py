
class RelativePositionAttention(nn.Module):
    """
    Attention mechanism that explicitly models relative positions between tokens.
    
    This adds learnable position embeddings to the standard attention mechanism,
    allowing the model to better understand spatial relationships.
    """
    def __init__(self, dims, heads, max_dist=128):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads

        # Create learnable relative position embeddings
        self.rel_pos_embedding = nn.Parameter(
            torch.randn(2 * max_dist - 1, self.head_dim) * 0.02
        )
        self.max_dist = max_dist

    def forward(self, q, k, v, mask=None):
        batch, heads, seq_len, head_dim = q.shape

        # Calculate content-based attention
        content_score = torch.matmul(q, k.transpose(-1, -2))

        # Calculate relative position scores
        rel_pos_indices = torch.arange(seq_len, device=q.device).unsqueeze(1) - torch.arange(seq_len, device=q.device).unsqueeze(0)
        rel_pos_indices = rel_pos_indices + self.max_dist - 1  # Shift to positive indices
        rel_pos_indices = torch.clamp(rel_pos_indices, 0, 2 * self.max_dist - 2)
        rel_embeddings = self.rel_pos_embedding[rel_pos_indices]

        # Calculate position-aware attention
        q_reshaped = q.permute(2, 0, 1, 3).reshape(seq_len, batch * heads, head_dim)
        rel_attn = torch.bmm(q_reshaped, rel_embeddings.transpose(1, 2))
        rel_attn = rel_attn.reshape(seq_len, batch, heads, seq_len).permute(1, 2, 0, 3)

        # Combine content and position information
        attn_score = content_score + rel_attn

        if mask is not None:
            attn_score = attn_score + mask

        attn_weights = F.softmax(attn_score, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        return out
        
