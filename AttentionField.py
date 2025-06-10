
class AttentionField(nn.Module):
    """
    Models attention as field interactions between tokens.
    
    Each token can emit different types of fields and has different sensitivities
    to those fields, with field strength decaying based on token distance.
    """
    def __init__(self, dims, heads, field_types=4):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads

        # Each token can emit different types of fields
        self.field_emitters = nn.Linear(dims, field_types * heads)

        # Each token has different sensitivities to fields
        self.field_receivers = nn.Linear(dims, field_types * heads)

        # Field decay parameters (learnable)
        self.field_decay = nn.Parameter(torch.ones(heads, field_types))
        
        # Output projection
        self.out_proj = nn.Linear(dims, dims)

    def forward(self, x, mask=None):
        batch, seq_len = x.shape[:2]

        # Calculate field emissions and sensitivities
        emitted = self.field_emitters(x).view(batch, seq_len, self.heads, -1)
        sensitivity = self.field_receivers(x).view(batch, seq_len, self.heads, -1)

        # Calculate positions
        positions = torch.arange(seq_len, device=x.device).float()
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)

        # Field strength decays with distance
        field_strength = torch.exp(-torch.abs(rel_pos).unsqueeze(0).unsqueeze(-1) *
                               self.field_decay.unsqueeze(1).unsqueeze(1))

        # Calculate attention as field interaction
        attention = torch.einsum('bshf,bthf,hfst->bsht',
                            sensitivity, emitted, field_strength)

        if mask is not None:
            attention = attention + mask

        weights = F.softmax(attention, dim=-1)
        
        # Use weights to combine value projections
        v = x.view(batch, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        output = torch.matmul(weights, v)
        output = output.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        
        return self.out_proj(output)

