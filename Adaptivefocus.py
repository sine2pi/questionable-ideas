
class Adaptivefocus(nn.Module):
    """
    Attention with dynamically adjusted span size.
    
    The model learns to focus on varying context window sizes based on content.
    """
    def __init__(self, dims, heads, max_dist=512, sharpen=True, temp_scale=0.01):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.head_dim = dims // heads
        self.max_dist = max_dist
        self.sharpen = sharpen
        self.temp_scale = temp_scale
        self.span_scale = nn.Parameter(torch.tensor(1.0))
        
        # Standard projections
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)

    def forward(self, x, mask=None):
        batch, seq_len = x.shape[:2]
        
        # Project q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Determine effective span
        span_mean = torch.sigmoid(self.span_scale).item()
        span_len = min(int(seq_len * span_mean), seq_len)
        
        # Use only a subset of tokens based on span
        q_span = q[:, :span_len]
        k_span = k[:, :span_len]
        v_span = v[:, :span_len]
        
        # Reshape for attention
        q_heads = q_span.view(batch, span_len, self.heads, self.head_dim).transpose(1, 2)
        k_heads = k_span.view(batch, span_len, self.heads, self.head_dim).transpose(1, 2)
        v_heads = v_span.view(batch, span_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Adjust temperature based on span
        if self.sharpen:
            temperature = 1.0 + self.temp_scale * (1.0 - span_mean)
        else:
            temperature = 0.5 + self.temp_scale * span_mean
            
        # Scale for attention
        scale = (self.head_dim ** -0.5) / temperature
        
        # Compute attention scores
        scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) * scale
        
        if mask is not None:
            # Adjust mask for span
            if mask.dim() == 2:
                mask = mask[:span_len, :span_len]
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            
        # Convert to weights and apply to values
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v_heads)
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(batch, span_len, self.dims)
        
        # Ensure output has same sequence length as input by padding
        if span_len < seq_len:
            padding = torch.zeros(batch, seq_len - span_len, self.dims, device=x.device)
            output = torch.cat([output, padding], dim=1)
            
        return self.output(output)
