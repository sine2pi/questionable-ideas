
class ForceFieldEmbedding(nn.Module):
    """
    Computes token-to-token force vectors that can be used to derive
    content-aware relative positions.
    """
    def __init__(self, dims, force_dim=None, decay_factor=2.0):
        """
        Initialize the force field embedding.
        Args:
            dims: Dimension of token embeddings
            force_dim: Dimension of force vectors (defaults to dims)
            decay_factor: Controls how quickly forces decay with distance
        """
        super().__init__()
        self.dims = dims
        self.force_dim = force_dim if force_dim is not None else dims
        self.decay_factor = decay_factor
        self.emit_proj = nn.Linear(dims, self.force_dim)
        self.receive_proj = nn.Linear(dims, self.force_dim)
        self.charge_proj = nn.Linear(dims, 1)
        self.emit_act = nn.SiLU()
        self.receive_act = nn.SiLU()
    def compute_force_field(self, x):
        """
        Compute the force field between all token pairs.
        Args:
            x: Token embeddings of shape [batch, seq_len, dims]
        Returns:
            force_vectors: Force vectors of shape [batch, seq_len, seq_len, force_dim]
            force_strengths: Strength of force between each token pair [batch, seq_len, seq_len]
        """
        batch, seq_len = x.shape[:2]
        emissions = self.emit_act(self.emit_proj(x))
        receptions = self.receive_act(self.receive_proj(x))
        charges = self.charge_proj(x).squeeze(-1)
        token_vectors = (emissions[:, :, None, :] - receptions[:, None, :, :])
        vector_magnitudes = torch.norm(token_vectors, dim=-1)
        charge_pairs = charges[:, :, None] * charges[:, None, :]
        force_strengths = charge_pairs / (vector_magnitudes + 1e-8).pow(self.decay_factor)
        normalized_vectors = token_vectors / (vector_magnitudes[..., None] + 1e-8)
        force_vectors = normalized_vectors * force_strengths[..., None]
        return force_vectors, force_strengths
    def derive_relative_positions(self, x):
        """
        Derive relative position offsets from token content.
        Args:
            x: Token embeddings of shape [batch, seq_len, dims]
        Returns:
            pos_offsets: Relative position offsets [batch, seq_len]
        """
        _, force_strengths = self.compute_force_field(x)
        influence_weights = F.softmax(force_strengths, dim=-1)
        seq_len = x.shape[1]
        target_positions = torch.arange(seq_len, device=x.device).float()
        content_positions = torch.matmul(influence_weights, target_positions)
        standard_positions = target_positions.unsqueeze(0).expand(x.shape[0], -1)
        pos_offsets = content_positions - standard_positions
        return pos_offsets
    def forward(self, x):
        """
        Compute force-based positional features for input tokens.
        Args:
            x: Token embeddings of shape [batch, seq_len, dims]
        Returns:
            pos_features: Positional features derived from forces
        """
        force_vectors, force_strengths = self.compute_force_field(x)
        incoming_forces = force_vectors.sum(dim=1)
        outgoing_forces = force_vectors.sum(dim=2)
        pos_features = torch.cat([incoming_forces, outgoing_forces], dim=-1)
        return pos_features
