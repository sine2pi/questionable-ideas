
class rbfEnhancedAttention(nn.Module):
    def __init__(self, dims, heads, rbf_sigma=1.0, rbf_ratio=0.0):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        self.scale = self.head_dim ** -0.5
        
        # Standard projections
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)
        
        # RBF parameters
        self.rbf_sigma = nn.Parameter(torch.tensor(rbf_sigma))
        self.rbf_ratio = nn.Parameter(torch.tensor(rbf_ratio))  # Blend ratio (0 = standard, 1 = pure RBF)
    
    def forward(self, x, mask=None):
        batch, seq_len = x.shape[:2]
        
        # Project to q, k, v
        q = self.q_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # 1. Standard dot-product attention
        dot_attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 2. RBF attention
        q_norm = q.pow(2).sum(dim=-1, keepdim=True)
        k_norm = k.pow(2).sum(dim=-1, keepdim=True)
        qk = torch.matmul(q, k.transpose(-1, -2))
        dist_sq = q_norm + k_norm.transpose(-1, -2) - 2 * qk
        rbf_attn = torch.exp(-dist_sq / (2 * self.rbf_sigma.pow(2)))
        
        # Blend the two attention mechanisms
        blend_ratio = torch.sigmoid(self.rbf_ratio)  # Ensure 0-1 range
        attn_scores = (1 - blend_ratio) * dot_attn + blend_ratio * rbf_attn
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and value aggregation
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.dims)
        
        return self.output(out)
