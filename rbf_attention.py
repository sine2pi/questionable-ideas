
def rbf_attention(q, k, v, mask=None, sigma=1.0):
    """
    Attention using radial basis function kernel instead of dot product.
    
    Creates a different similarity metric based on distances in embedding space.
    """
    # Compute squared distances between queries and keys
    q_norm = q.pow(2).sum(dim=-1, keepdim=True)
    k_norm = k.pow(2).sum(dim=-1, keepdim=True)
    qk = torch.matmul(q, k.transpose(-1, -2))
    dist_sq = q_norm + k_norm.transpose(-1, -2) - 2 * qk

    # Apply RBF kernel
    sim = torch.exp(-dist_sq / (2 * sigma))

    if mask is not None:
        sim = sim.masked_fill(mask == float('-inf'), 0)

    weights = sim / (sim.sum(dim=-1, keepdim=True) + 1e-8)
    return torch.matmul(weights, v)
