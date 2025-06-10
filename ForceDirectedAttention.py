
class ForceDirectedAttention(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.dims = dims
        self.heads = heads

        self.force_emitter = nn.Linear(dims, dims)

        self.force_receptor = nn.Linear(dims, dims)

        self.direction_modulator = nn.Parameter(torch.randn(heads, dims))

        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)

    def forward(self, x):
        batch, seq_len = x.shape[:2]

        emissions = self.force_emitter(x)
        receptivity = self.force_receptor(x)

        q = self.q_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)

        token_i = emissions.unsqueeze(2)
        token_j = receptivity.unsqueeze(1)

        force_directions = token_i - token_j
        force_magnitudes = torch.norm(force_directions, dim=-1, keepdim=True)
        normalized_forces = force_directions / (force_magnitudes + 1e-8)

        direction_scores = torch.einsum('bstn,hd->bshtn',
                                     normalized_forces,
                                     self.direction_modulator)

        force_field = direction_scores * torch.exp(-force_magnitudes)

        attention = torch.sum(force_field, dim=-1)
        weights = F.softmax(attention, dim=-1)

        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)

def calculate_forces(emissions, receptivity, positions, decay_factor=2, epsilon=1e-8):
    force_directions = emissions.unsqueeze(2) - receptivity.unsqueeze(1)

    distances = torch.norm(force_directions, dim=-1, keepdim=True)

    normalized_forces = force_directions / (distances + epsilon)

    charges = (emissions.unsqueeze(2) * receptivity.unsqueeze(1)).sum(dim=-1, keepdim=True)
    magnitudes = charges / (distances ** decay_factor + epsilon)

    forces = normalized_forces * magnitudes

    return forces
