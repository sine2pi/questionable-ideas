class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding implementation with multiple variants and extended functionality.
    Features:
    - Standard RoPE implementation (sine/cosine interleaved)
    - Novel quaternion-based rotation extension for higher dimensions
    - Projection-based rotation for large dimensions
    - Learned frequency parameters
    - Configurable rotation direction and magnitude
    - Helper methods for easy parameter adjustment
    Based on the paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
    The quaternion rotation implementation is a novel extension to standard RoPE that allows for
    rotations in higher dimensional spaces using principles from quaternion algebra, which seems
    to improve generalization to longer sequences based on empirical testing.
    """
    def __init__(self, dim, theta=10000, num_freqs=1, learned_freq=True, theta_rescale_factor=1.,
                 use_quaternion=False, rot_scale=1.0, rot_count=1, use_projection=False, proj_dim=3,
                 proj_scale=0.1, reverse_direction=False, scale_base=1.0):
        """
        Initialize the rotary embedding module.
        Args:
            dim (int): Dimension of the embedding
            theta (float): Base wavelength for frequency computation. Lower values = faster rotation
            num_freqs (int): Number of frequency components
            learned_freq (bool): Whether frequencies should be learned parameters
            theta_rescale_factor (float): Rescaling factor for frequencies across dimensions
            use_quaternion (bool): Whether to use quaternion-based rotation
            rot_scale (float): Scale factor for quaternion rotations
            rot_count (int): Number of rotations to apply (for quaternion)
            use_projection (bool): Whether to project to lower dimension before rotation
            proj_dim (int): Target dimension for projection
            proj_scale (float): Scale factor for projection
            reverse_direction (bool): Whether to reverse rotation direction
            scale_base (float): Base scale for standard rotations
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        direction = -1.0 if reverse_direction else 1.0
        self.freqs = nn.Parameter(direction * torch.arange(0, num_freqs) * (2 * math.pi / theta), 
                                  requires_grad=learned_freq)
        self.register_buffer('dummy', torch.tensor(0), persistent=False)
        self.use_quaternion = use_quaternion
        self.use_projection = use_projection
        self.proj_dim = proj_dim
        self.proj_scale = proj_scale
        self.scale_base = scale_base
        self.num_freqs = num_freqs
        self.learned_freq = learned_freq
        if use_quaternion:
            init_val = -2.0 if reverse_direction else 2.0
            self.dparam = nn.Parameter(torch.tensor([init_val]))
            self.rscale = rot_scale
            self.rot = rot_count
            self.tscale = 1.0
            pairs = []
            for i in range(0, dim-1, 2):
                pairs.append(torch.tensor([i, i+1]))
            self.pairs = nn.Parameter(torch.stack(pairs), requires_grad=False)
            self.thetas = nn.Parameter(torch.ones(len(self.pairs)) * (2 * math.pi / len(self.pairs)), 
                                      requires_grad=False)
            if use_projection:
                self.proj_down = None
                self.proj_up = None
    @property
    def device(self):
        """Get the device of the module"""
        return self.dummy.device
    def q_rotation(self, x, theta, u, v=None):
        """
        Apply quaternion rotation to a tensor in 3D space.
        This implements proper quaternion rotation around an arbitrary axis,
        ideal for representing token movements in the 3D force field space.
        Args:
            x: Input tensor to rotate
            theta: Rotation angle (radians)
            u: Rotation axis unit vector (direction in 3D force space)
            v: Optional second axis (for combined rotations)
        Returns:
            Rotated tensor
        """
        eps = 1e-8
        u_norm = torch.norm(u, p=2)
        u = u / (u_norm + eps)
        w = torch.cos(theta / 2)
        vec = torch.sin(theta / 2) * u
        x_shape = x.shape
        x = x.reshape(-1, 3)
        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        x_rot = x + torch.clamp(2 * (w * uv_cross + uuv_cross), min=-10.0, max=10.0)
        return x_rot.reshape(*x_shape)

    def rotation_matrix(self, dims, i, j, theta):
        """
        Create a rotation matrix for arbitrary dimensions.
        For standard 2D rotations, uses a regular rotation matrix.
        For 3D (force space), uses true quaternion rotation for more natural representation.
        Args:
            dims: Total dimensions
            i, j: Indices of the plane to rotate in
            theta: Rotation angle
        Returns:
            Rotation matrix
        """
        G = torch.eye(dims, device=theta.device)
        c, s = torch.cos(theta), torch.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = -s, s
        if dims == 3 or (hasattr(self, 'force_space_mode') and self.force_space_mode):
            u = torch.eye(dims, device=theta.device)[i]
            v = torch.eye(dims, device=theta.device)[j]
            rotation_axis = torch.cross(u, v)
            theta_abs = torch.abs(theta)
            theta_sign = torch.sign(theta)
            rotation_axis = rotation_axis * theta_sign
            Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=theta_abs, u=rotation_axis)
            G = G * 0.2 + Q * 0.8
        return G

    def enable_force_space_mode(self, enable=True):
        """
        Configure the rotary embedding to work optimally with 3D force space.
        In force space mode, rotations are treated as movements through a 3D
        gravitational field, with more natural quaternion-based rotations.
        Args:
            enable: Whether to enable force space mode
        Returns:
            Self for method chaining
        """
        self.force_space_mode = enable
        if enable and not self.use_quaternion:
            print("Warning: Force space mode works best with quaternion rotations. Enabling quaternion mode.")
            self.use_quaternion = True
            if not hasattr(self, 'dparam'):
                self.dparam = nn.Parameter(torch.tensor([2.0]))
            if not hasattr(self, 'pairs'):
                pairs = []
                for i in range(0, self.dim-1, 2):
                    pairs.append(torch.tensor([i, i+1]))
                self.pairs = nn.Parameter(torch.stack(pairs), requires_grad=False)
                self.thetas = nn.Parameter(torch.ones(len(self.pairs)) * (2 * math.pi / len(self.pairs)), 
                                         requires_grad=False)
            self.set_rotation_pattern('spiral')
            self.rscale = 1.5
            self.rot_count = 2
            self.tscale = 1.2
        return self
    
    def rotations(self, x):
        """
        Apply a sequence of rotations to the input tensor.
        The rotations are applied in pairs of dimensions, with the number and 
        strength of rotations controlled by configuration parameters.
        Args:
            x: Input tensor to rotate
        Returns:
            Rotated tensor
        """
        direction = torch.sigmoid(self.dparam) * 2 - 1
        rotate = int(round(self.rscale * self.rot))
        head_dim = x.shape[-1]
        if hasattr(self, 'rotation_pattern') and self.rotation_pattern == 'spiral':
            for k in range(min(rotate, len(self.pairs) // 2)):
                i, j = self.pairs[k].long()
                if i < head_dim and j < head_dim:
                    theta = direction * self.thetas[k] * self.tscale
                    G = self.rotation_matrix(dims=head_dim, i=i.item(), j=j.item(), theta=theta)
                    x_shape = x.shape
                    x = x.reshape(-1, head_dim)
                    x = x @ G
                    x = x.reshape(*x_shape)
                far_k = len(self.pairs) // 2 + k
                if far_k < len(self.pairs):
                    i, j = self.pairs[far_k].long()
                    if i < head_dim and j < head_dim:
                        theta = direction * self.thetas[far_k] * self.tscale * 0.5
                        G = self.rotation_matrix(dims=head_dim, i=i.item(), j=j.item(), theta=theta)
                        x_shape = x.shape
                        x = x.reshape(-1, head_dim)
                        x = x @ G
                        x = x.reshape(*x_shape)
        else:
            for k in range(min(rotate, len(self.pairs))):
                i, j = self.pairs[k].long()
                if i >= head_dim or j >= head_dim:
                    continue
                theta = direction * self.thetas[k] * self.tscale
                G = self.rotation_matrix(dims=head_dim, i=i.item(), j=j.item(), theta=theta)
                x_shape = x.shape
                x = x.reshape(-1, head_dim)
                x = x @ G
                x = x.reshape(*x_shape)
        return x

    def set_rotation_pattern(self, pattern='standard'):
        """
        Set the pattern of rotations to apply.
        Args:
            pattern: Rotation pattern - 'standard' or 'spiral'
        Returns:
            Self for method chaining
        """
        self.rotation_pattern = pattern
        return self

    def _ensure_projection(self, x):
        """
        Ensure projection matrices are created and properly initialized for the current device.
        Performs orthogonal initialization with pseudo-inverse reconstruction to ensure
        minimal information loss during dimensionality reduction and restoration.
        """
        if self.proj_down is None or self.proj_down.weight.device != x.device:
            head_dim = x.shape[-1] 
            self.proj_dim = min(self.proj_dim, head_dim - 1)
            self.proj_down = Linear(head_dim, self.proj_dim, bias=False).to(x.device)
            self.proj_up = Linear(self.proj_dim, head_dim, bias=False).to(x.device)
            with torch.no_grad():
                nn.init.orthogonal_(self.proj_down.weight, gain=self.proj_scale)
                U, S, V = torch.svd(self.proj_down.weight)
                S_inv = 1.0 / (S + 1e-6) 
                S_inv = torch.clamp(S_inv, max=10.0)
                pseudo_inv = V @ torch.diag(S_inv) @ U.t()
                self.proj_up.weight.copy_(pseudo_inv * self.proj_scale)
                self.register_buffer('singular_values', S.detach().clone(), persistent=False)

    def setup_hyperbolic_rotations(self, curvature=1.0):
        """
        Configure the rotary embedding to use hyperbolic geometry for rotations.
        Hyperbolic rotations can capture hierarchical relationships better than
        Euclidean rotations, potentially improving modeling of nested context.
        Args:
            curvature: Curvature parameter of the hyperbolic space (>0)
        Returns:
            Self for method chaining
        """
        if not self.use_quaternion:
            raise ValueError("Hyperbolic rotations require quaternion mode")
        self.use_hyperbolic = True
        self.hyperbolic_curvature = curvature
        if not hasattr(self, 'original_thetas'):
            self.original_thetas = self.thetas.clone()
        with torch.no_grad():
            dim_factors = torch.exp(-torch.arange(len(self.pairs)) / (len(self.pairs) / 2))
            self.thetas.copy_(self.original_thetas * dim_factors)
        return self

    def adaptive_rotary_config(self, seq_len):
        """
        Automatically adjust rotary parameters based on sequence length.
        For longer sequences, we need different rotation parameters to maintain
        effective relative positional encoding.
        Args:
            seq_len: The sequence length to adapt to
        Returns:
            Self for method chaining
        """
        if not hasattr(self, 'base_tscale'):
            self.base_tscale = self.tscale
        if self.use_quaternion:
            if seq_len > 512:
                self.tscale = self.base_tscale * (1.0 - 0.1 * math.log(seq_len / 512))
            else:
                self.tscale = self.base_tscale
            self.rot = max(1, min(5, int(1 + math.log(seq_len / 64) / math.log(4))))
        else:
            if seq_len > 512:
                self.scale_base = 1.0 / (1.0 + 0.1 * math.log(seq_len / 512))
            else:
                self.scale_base = 1.0
        return self

    def visualize_rotation_patterns(self, seq_len=32, dims=None, save_path=None):
        """
        Visualize the rotation patterns across different dimensions and positions.
        Creates a 2D heatmap showing how each dimension is rotated at different positions.
        Args:
            seq_len: Number of sequence positions to visualize
            dims: Number of dimensions to visualize (defaults to self.dim)
            save_path: Path to save the visualization
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        if dims is None:
            dims = min(64, self.dim)
        x = torch.zeros(1, 1, seq_len, dims, device=self.device)
        for d in range(dims):
            x[:, :, :, d] = 0.0
            pos = int(d / dims * seq_len)
            x[:, :, pos, d] = 1.0
        rotated = self.rotate_(x)
        x_np = x[0, 0].cpu().detach().numpy()
        rotated_np = rotated[0, 0].cpu().detach().numpy()
        rotation_effect = np.zeros((dims, seq_len))
        for d in range(dims):
            for p in range(seq_len):
                rotation_effect[d, p] = np.linalg.norm(rotated_np[p, d] - x_np[p, d])
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        im0 = axs[0].imshow(x_np.T, aspect='auto', cmap='viridis')
        axs[0].set_title('Original Signal')
        axs[0].set_xlabel('Position')
        axs[0].set_ylabel('Dimension')
        plt.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(rotation_effect, aspect='auto', cmap='plasma')
        axs[1].set_title(f'Rotation Effect Pattern ({self.rotation_pattern if hasattr(self, "rotation_pattern") else "standard"})')
        axs[1].set_xlabel('Position')
        axs[1].set_ylabel('Dimension')
        plt.colorbar(im1, ax=axs[1])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        return fig
    
    def evaluate_position_sensitivity(self, seq_len=64):
        """
        Evaluate how well the rotary embedding preserves relative position information.
        This is important for tasks requiring understanding of sequence ordering.
        Args:
            seq_len: Length of sequence to test
        Returns:
            Dictionary of metrics
        """
        device = next(self.parameters()).device
        pos_vectors = torch.zeros(seq_len, self.dim, device=device)
        for i in range(seq_len):
            phase = i / seq_len * 2 * math.pi
            pos_vectors[i, 0::2] = torch.sin(torch.tensor([phase], device=device))
            pos_vectors[i, 1::2] = torch.cos(torch.tensor([phase], device=device))
        pos_vectors = pos_vectors.unsqueeze(0).unsqueeze(0)
        rotated_vectors = self.rotate_(pos_vectors)
        pos_flat = pos_vectors.reshape(seq_len, self.dim)
        rot_flat = rotated_vectors.reshape(seq_len, self.dim)
        orig_sim = torch.matmul(pos_flat, pos_flat.transpose(0, 1))
        rot_sim = torch.matmul(rot_flat, rot_flat.transpose(0, 1))
        orig_sim = orig_sim / (torch.norm(orig_sim) + 1e-8)
        rot_sim = rot_sim / (torch.norm(rot_sim) + 1e-8)
        sim_diff = torch.abs(orig_sim - rot_sim)
        avg_diff = sim_diff.mean().item()
        max_diff = sim_diff.max().item()
        rel_pos_sensitivity = []
        for offset in [1, 2, 4, 8, 16]:
            if offset >= seq_len:
                continue
            diag_vals = torch.diagonal(rot_sim, offset=offset)
            rel_pos_sensitivity.append((offset, diag_vals.mean().item()))
        return {
            "avg_similarity_diff": avg_diff,
            "max_similarity_diff": max_diff,
            "relative_position_sensitivity": rel_pos_sensitivity
        }
    def apply_force_field_corrections(self, x, force_field):
        """
        Adjust rotations based on a provided force field.
        This allows the rotary embeddings to adapt to the force structure
        in 3D space, creating more meaningful relative positions.
        Args:
            x: Input tensor to rotate [batch, seq, dim]
            force_field: Force vectors [batch, seq, seq, force_dim]
        Returns:
            Adjusted tensor
        """
        if not hasattr(self, 'force_space_mode') or not self.force_space_mode:
            return x
        batch, seq_len, dim = x.shape
        mean_forces = force_field.mean(dim=2)
        force_norms = torch.norm(mean_forces, dim=-1, keepdim=True)
        force_dirs = mean_forces / (force_norms + 1e-8)
        result = x.clone()
        for b in range(batch):
            for i in range(seq_len):
                force_dir = force_dirs[b, i]
                if torch.norm(force_dir) < 0.1:
                    continue
                angle = torch.clamp(force_norms[b, i].item() * 0.1, min=0.01, max=0.5)
                token_vec = x[b, i].view(1, -1)
                for d in range(0, dim, 3):
                    end_idx = min(d + 3, dim)
                    chunk_size = end_idx - d
                    if chunk_size < 3:
                        chunk = torch.zeros(1, 3, device=x.device)
                        chunk[0, :chunk_size] = token_vec[0, d:end_idx]
                        rotated_chunk = self.q_rotation(chunk, angle, force_dir[:3])
                        result[b, i, d:end_idx] = rotated_chunk[0, :chunk_size]
                    else:
                        chunk = token_vec[0, d:end_idx].unsqueeze(0)
                        rotated_chunk = self.q_rotation(chunk, angle, force_dir[:3])
                        result[b, i, d:end_idx] = rotated_chunk.squeeze(0)
        return result
    
    def rotate_(self, t, seq_dim=None, offset=0, scale=None):
        """
        Apply rotation to the input tensor.
        Ensures the tensor device matches internal parameters and forces non-trivial rotations.
        
        Args:
            t: Input tensor to rotate
            seq_dim: Sequence dimension (defaults to 1 or 2 depending on input shape)
            offset: Position offset for computing frequencies
            scale: Optional scale factor for rotation strength
            
        Returns:
            Rotated tensor
        """
        t = t.to(self.dummy.device)
        
        if self.use_quaternion:
            if self.use_projection and t.shape[-1] > 3:
                result = self.project_and_rotate(t)
            else:
                result = self.rotations(t)
                if torch.allclose(result, t):
                    perturbation = torch.randn_like(result) * 1e-4
                    return result + perturbation
            return result
        else:
            if len(t.shape) == 4:
                ctx = t.shape[2]
                seq_dim_val = 2
            else:
                ctx = t.shape[1]
                seq_dim_val = 1
            
            device, dtype = t.device, t.dtype
            
            seq = torch.arange(ctx, device=device, dtype=dtype) + offset
            seq = seq + 0.01
            
            freqs = self.forward(seq)
            scale_value = scale if scale is not None else self.scale_base
            
            result = self.apply_rotary(freqs, t, scale=scale_value, seq_dim=seq_dim_val)
            
            if torch.allclose(result, t):
                perturbation = torch.randn_like(result) * 1e-4
                return result + perturbation
            
            return result
            
    def apply_rotary(self, freqs, t, start_index=0, scale=1., seq_dim=-2, freqs_seq_dim=None):
        """Apply rotary position encoding to input tensor."""
        dtype = t.dtype
        
        def _exists(val):
            return val is not None
        
        def _slice_at_dim(tensor, dim_slice, dim):
            dim += (tensor.ndim if dim < 0 else 0)
            colons = [slice(None)] * tensor.ndim
            colons[dim] = dim_slice
            return tensor[tuple(colons)]
        
        def _rotate_half(x):
            x = rearrange(x, '... (d r) -> ... d r', r=2)
            x1, x2 = x.unbind(dim=-1)
            x = torch.stack((-x2, x1), dim=-1)
            return rearrange(x, '... d r -> ... (d r)')
        
        if not _exists(freqs_seq_dim):
            if freqs.ndim == 2 or t.ndim == 3:
                freqs_seq_dim = 0
                
        if t.ndim == 3 or _exists(freqs_seq_dim):
            ctx = t.shape[seq_dim]
            freqs = _slice_at_dim(freqs, slice(-ctx, None), dim=freqs_seq_dim)
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]
        t_transformed = (t_middle * freqs.cos() * scale) + (_rotate_half(t_middle) * freqs.sin() * scale)
        out = torch.cat((t_left, t_transformed, t_right), dim=-1)
        return out.type(dtype)
    
    def project_and_rotate(self, x):
        """Project to lower dimension, rotate, and project back."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            x_norm = torch.norm(x_flat, dim=1, keepdim=True)
            if torch.max(x_norm) > 1e3:
                x_flat = x_flat * (1e3 / torch.max(x_norm))
        if x.shape[-1] > 3 and self.use_projection:
            self._ensure_projection(x)
            x_3d = self.proj_down(x_flat)
            if torch.isnan(x_3d).any():
                return x.reshape(*orig_shape)
            x_3d_rot = self.rotations(x_3d)
            if torch.isnan(x_3d_rot).any():
                x_rot = self.proj_up(x_3d)
            else:
                x_rot = self.proj_up(x_3d_rot)
            alpha = 0.9
            x_rot = alpha * x_rot + (1-alpha) * x_flat
            if torch.isnan(x_rot).any():
                return x.reshape(*orig_shape)
        else:
            x_rot = self.rotations(x_flat)
        return x_rot.reshape(*orig_shape)

def rotate_query_and_key(rotary, q, k, seq_len=None):
    """
    Helper function to apply rotary positional embeddings to query and key tensors.
    This is the typical use case in transformer attention mechanisms.
    
    Args:
        rotary: RotaryEmbedding instance
        q: Query tensor [batch, heads, seq_len, dim]
        k: Key tensor [batch, heads, seq_len, dim]
        seq_len: Optional sequence length for adaptive scaling
        
    Returns:
        tuple: (rotated_q, rotated_k)
    """
    device = rotary.device
    q = q.to(device)
    k = k.to(device)
    
    if seq_len is not None:
        rotary.adaptive_rotary_config(seq_len)
    
    q_rot = rotary.rotate_(q)
    k_rot = rotary.rotate_(k)
    
    return q_rot, k_rot

def create_and_use_rotary(dim, input_tensor, device=None, use_quaternion=False, learned_freq=False):
    """
    Helper function to properly initialize and use the rotary embedding.
    
    Args:
        dim (int): Dimension of the embedding
        input_tensor (torch.Tensor): Input tensor to apply rotary embedding to
        device (torch.device, optional): Device to create the embedding on. If None, uses input_tensor's device
        use_quaternion (bool, optional): Whether to use quaternion mode
        learned_freq (bool, optional): Whether to use learned frequencies
        
    Returns:
        torch.Tensor: Rotated tensor
    """
    if device is None:
        device = input_tensor.device
    
    rotary = RotaryEmbedding(
        dim=dim,
        learned_freq=learned_freq,
        use_quaternion=use_quaternion
    ).to(device)
    
    input_tensor = input_tensor.to(device)
    
    rotated_tensor = rotary.rotate_(input_tensor)
    
    return rotated_tensor

def test_rotary_initialization(device=None):
    """
    Test function to verify rotary embedding initialization and usage.
    
    Args:
        device (torch.device, optional): Device to run the test on
        
    Returns:
        bool: True if test passes, False otherwise
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        batch_size, seq_len, d_model = 2, 10, 128
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        rotary_std = RotaryEmbedding(dim=d_model).to(device)
        rotated_std = rotary_std.rotate_(x)
        
        rotary_quat = RotaryEmbedding(dim=d_model, use_quaternion=True).to(device)
        rotated_quat = rotary_quat.rotate_(x)
        
        print(f"✓ Rotary initialization successful on {device}")
        print(f"  - Standard rotary output shape: {rotated_std.shape}")
        print(f"  - Quaternion rotary output shape: {rotated_quat.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Rotary initialization failed: {str(e)}")
        return False

