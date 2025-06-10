class TopologicalAttention(nn.Module):
    """
    Attention based on learned graph structure with multi-hop paths.
    
    Creates a graph structure between tokens and considers both direct and indirect
    (multi-hop) connections.
    """
    def __init__(self, dims, heads, max_hops=2):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        self.max_hops = max_hops

        # Project to edge features that define the topology
        self.edge_projector = nn.Linear(dims * 2, heads)
        
        # Hop weights for combining different path lengths
        self.hop_weights = nn.Parameter(torch.ones(max_hops + 1) / (max_hops + 1))
        
        # Standard projections
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)

    def forward(self, x, adjacency_hint=None, mask=None):
        batch, seq_len = x.shape[:2]

        # Project to values
        v = self.v_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Create all pairs of token representations
        xi = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        xj = x.unsqueeze(1).expand(-1, seq_len, -1, -1)

        # Concatenate to get pair representations
        pairs = torch.cat([xi, xj], dim=-1)

        # Project to get edge weights/topology
        edge_logits = self.edge_projector(pairs).permute(0, 3, 1, 2)

        if adjacency_hint is not None:
            # Use syntactic or semantic hints about connectivity
            edge_logits = edge_logits + adjacency_hint

        # Convert to probabilities
        direct_edges = torch.sigmoid(edge_logits)

        # Calculate multi-hop paths
        topo_paths = [direct_edges]
        current_paths = direct_edges

        for hop in range(1, self.max_hops + 1):
            # Calculate paths with one more hop
            next_hop = torch.matmul(current_paths, direct_edges) / (seq_len ** 0.5)
            topo_paths.append(next_hop)
            current_paths = next_hop

        # Combine paths of different hop lengths
        hop_weights = F.softmax(self.hop_weights, dim=0)
        attention = sum(w * path for w, path in zip(hop_weights, topo_paths))

        if mask is not None:
            # Apply mask if provided
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attention = attention + mask

        # Normalize and use for value aggregation
        weights = F.softmax(attention, dim=-1)
        
        # Apply weights to values
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)

        return self.output(output)
