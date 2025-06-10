
class AttentionField(nn.Module):
    def __init__(self, dims, heads, field_types=4):
        super().__init__()
        self.dims = dims
        self.heads = heads

        self.field_emitters = nn.Linear(dims, field_types * heads)

        self.field_receivers = nn.Linear(dims, field_types * heads)

        self.field_decay = nn.Parameter(torch.ones(heads, field_types))

    def forward(self, x, mask=None):
        batch, seq_len = x.shape[:2]

        emitted = self.field_emitters(x).view(batch, seq_len, self.heads, -1)
        sensitivity = self.field_receivers(x).view(batch, seq_len, self.heads, -1)

        positions = torch.arange(seq_len, device=x.device).float()
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)

        field_strength = torch.exp(-torch.abs(rel_pos).unsqueeze(0).unsqueeze(-1) *
                               self.field_decay.unsqueeze(1).unsqueeze(1))

        attention = torch.einsum('bshf,bthf,hfst->bsht',
                            sensitivity, emitted, field_strength)

        if mask is not None:
            attention = attention + mask

        weights = F.softmax(attention, dim=-1)
