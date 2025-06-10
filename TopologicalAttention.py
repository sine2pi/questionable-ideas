
class TopologicalAttention(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        self.dims = dims
        self.heads = heads

        self.edge_projector = nn.Linear(dims * 2, heads)

    def forward(self, x, adjacency_hint=None):
        batch, seq_len = x.shape[:2]

        xi = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        xj = x.unsqueeze(1).expand(-1, seq_len, -1, -1)

        pairs = torch.cat([xi, xj], dim=-1)

        edge_logits = self.edge_projector(pairs).permute(0, 3, 1, 2)

        if adjacency_hint is not None:
            edge_logits = edge_logits + adjacency_hint

        edge_weights = torch.sigmoid(edge_logits)

        indirect = torch.matmul(edge_weights, edge_weights) / seq_len

        attention_weights = 0.8 * edge_weights + 0.2 * indirect

        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
