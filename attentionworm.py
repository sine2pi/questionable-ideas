

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict

from torch.nn.functional import scaled_dot_product_attention
device = torch.device(device="cuda:0")
dtype = torch.float32

class attentionworm(nn.Module):
    def __init__(self, dims: int, head: int, max_dist: int = 512):
        super().__init__()
        assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_dist = max_dist
        self.scale = self.head_dim ** -0.5

    def calculate_attention(q, k, v, mask=None, temperature=1.0, use_sdpa=True, is_causal=True):
        batch_size = q.shape[0]
        ctx = q.shape[2]
        attn_mask = None
        if mask is not None:
            if mask.dim() <= 3:
                attn_mask = create_attention_mask(
                    batch_size=batch_size, 
                    ctx=ctx, 
                    is_causal=is_causal, 
                    padding_mask=mask if mask.dim() > 1 else None,
                    device=q.device)
            else:
                attn_mask = mask
        scaled_q = q
        if temperature != 1.0 and temperature > 0:
            scaled_q = q * (1.0 / temperature)**.5
        a = scaled_dot_product_attention(
            scaled_q, k, v, 
            attn_mask=attn_mask, 
            is_causal=is_causal if attn_mask is None else False)
        out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
        return out, None

    class ProjectionModule(nn.Module):
        def __init__(self, dims: int, head: int, proj_type: str = "query", use_bias: bool = True):
            super().__init__()
            assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
            self.dims = dims
            self.head = head
            self.head_dim = dims // head
            self.proj_type = proj_type
            self.scale = self.head_dim ** -0.25 if proj_type != "value" else 1.0
            self.proj = Linear(in_features=dims, out_features=dims, bias=use_bias)
            self.init_weights()
            
        def init_weights(self):
            nn.init.normal_(tensor=self.proj.weight, std=0.02)
            if hasattr(self.proj, 'bias') and self.proj.bias is not None:
                nn.init.zeros_(tensor=self.proj.bias)
        
        def forward(self, x: Tensor) -> Tensor:
            batch, ctx = x.shape[:2]
            proj = self.proj(x)
            
            proj = proj.view(batch, ctx, self.head, self.head_dim).permute(0, 2, 1, 3)
            if self.proj_type in ["query", "key"]:
                proj = proj * self.scale
            return proj

    def calculate_attention2(q, k, v, mask=None, temperature=1.0, use_sdpa=True, is_causal=True):

        if use_sdpa:
            try:
                if mask is not None:
                    if mask.dtype == torch.bool:
                        float_mask = torch.zeros_like(mask, dtype=torch.float)
                        float_mask = float_mask.masked_fill(mask, float('-inf'))
                        attn_output = scaled_dot_product_attention(
                            q, k, v, attn_mask=float_mask)
                    else:
                        attn_output = scaled_dot_product_attention(
                            q, k, v, attn_mask=mask, is_causal=is_causal)
                else:
                    attn_output = scaled_dot_product_attention(
                        q, k, v, attn_mask=None, is_causal=is_causal)
                return attn_output, None
            except RuntimeError:
                pass
        scale = 1.0 / temperature if temperature > 0 else 1.0
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        if mask is not None:
            if mask.dim() == 4:
                q_len, k_len = q.size(2), k.size(2)
                mask_q_len = min(mask.size(2), q_len)
                mask_k_len = min(mask.size(3), k_len)
                
                if mask.dtype == torch.bool:
                    mask_part = mask[:, :, :mask_q_len, :mask_k_len]
                    attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len].masked_fill(mask_part, float("-inf"))
                else:
                    attn[:, :, :mask_q_len, :mask_k_len] = attn[:, :, :mask_q_len, :mask_k_len] + mask[:, :, :mask_q_len, :mask_k_len]
        attn = F.softmax(attn, dim=-1)
        
        if mask is not None and mask.dtype == torch.bool:
            binary_mask = (~mask).float()
            masked_attn = attn * binary_mask
            attn_sum = masked_attn.sum(dim=-1, keepdim=True)
            attn = masked_attn / (attn_sum + 1e-6)
        attn_output = torch.matmul(attn, v)
        return attn_output, attn

    class BaseAttention(nn.Module):
        """Base class for attention mechanisms with common functionality."""
        use_sdpa = True
        
        def __init__(self, dims: int, head: int, max_dist: int = 512):
            super().__init__()
            assert dims % head == 0, f"dims ({dims}) must be divisible by head ({head})"
            self.dims = dims
            self.head = head
            self.head_dim = dims // head
            self.max_dist = max_dist
            self.scale = self.head_dim ** -0.25
            
        def _shape(self, tensor: torch.Tensor, ctx: int, batch: int):
            return tensor.view(batch, ctx, self.head, self.head_dim).transpose(1, 2).contiguous()
            
        def _reshape_to_output(self, attn_output, batch, ctx):
            return attn_output.permute(0, 2, 1, 3).reshape(batch, ctx, self.dims)

    class AttentionCombiner(BaseAttention):
        def __init__(self, dims: int, head: int):
            super().__init__(dims, head)
            self.out = Linear(in_features=dims, out_features=dims)
            nn.init.normal_(tensor=self.out.weight, std=0.02)
            nn.init.zeros_(tensor=self.out.bias)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, is_causal=True) -> Tensor:
        if q.dim() == 3:
            batch, ctx, dims = q.shape
            self.scale = (dims // self.head) ** -0.5
            q = self._shape(q, ctx, batch)
            k = self._shape(k, k.size(1), batch)
            v = self._shape(v, v.size(1), batch)
        else:
            batch = q.size(0)
            ctx = q.size(2)
        attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa, is_causal=is_causal)
        output = self._reshape_to_output(attn_output, batch, ctx)
        return self.out(output)

    class AdaptiveUpdateAttention(BaseAttention):
        """Attention implementation with content-dependent update frequencies."""
        def __init__(self, dims: int, head: int, max_dist=512):
            super().__init__(dims, head, max_dist)
            
            self.query_module = ProjectionModule(dims, head, "query")
            self.key_module = ProjectionModule(dims, head, "key")
            self.value_module = ProjectionModule(dims, head, "value")
            self.combiner = AttentionCombiner(dims, head)
            self.key_update_predictor = nn.Sequential(
                Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())
            self.value_update_predictor = nn.Sequential(
                Linear(dims, dims // 4), nn.ReLU(), Linear(dims // 4, 1), nn.Sigmoid())

            self.update_threshold = 0.5
            self.stored_key_cache = None
            self.stored_value_cache = None

        def should_update_key(self, x: torch.Tensor) -> torch.Tensor:
            """Predict whether the key should be updated based on content."""
            avg_rep = x.mean(dim=1)
            return self.key_update_predictor(avg_rep) > self.update_threshold

        def should_update_value(self, x: torch.Tensor) -> torch.Tensor:
            """Predict whether the value should be updated based on content."""
            avg_rep = x.mean(dim=1)
            return self.value_update_predictor(avg_rep) > self.update_threshold

        def forward(self, x, xa=None, mask=None, kv_cache=None, is_causal=True):
            """Process inputs with adaptive update mechanism."""
            batch, ctx, _ = x.shape
            q = self.query_module(x)
            kv_input = xa if xa is not None else x
            device = kv_input.device  # noqa: F841

            if kv_cache is None:
                k = self.key_module(kv_input)
                v = self.value_module(kv_input)
                
                self.stored_key_cache = k
                self.stored_value_cache = v
            else:
                update_k = self.should_update_key(kv_input)
                update_v = self.should_update_value(kv_input)
                if update_k.any():
                    new_k = self.key_module(kv_input)
                    if self.stored_key_cache is not None:
                        update_mask = update_k.view(-1, 1, 1, 1).expand_as(self.stored_key_cache)
                        k = torch.where(update_mask, new_k, self.stored_key_cache)
                    else:
                        k = new_k
                else:
                    k = self.stored_key_cache if self.stored_key_cache is not None else self.key_module(kv_input)
                if update_v.any():
                    new_v = self.value_module(kv_input)
                    if self.stored_value_cache is not None:
                        update_mask = update_v.view(-1, 1, 1, 1).expand_as(self.stored_value_cache)
                        v = torch.where(update_mask, new_v, self.stored_value_cache)
                    else:
                        v = new_v
                else:
                    v = self.stored_value_cache if self.stored_value_cache is not None else self.value_module(kv_input)
                self.stored_key_cache = k
                self.stored_value_cache = v
            output = self.combiner(q, k, v, mask=mask, is_causal=is_causal)
            return output

    class Refiner:
        """Q-learning based refiner for adaptive attention span."""
        def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
            self.states = states
            self.actions = actions
            self.R = {}
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.default_value = 0.0

        def get_value(self, state, action):
            return self.R.get((state, action), self.default_value)

        def set_value(self, state, action, value):
            self.R[(state, action)] = value

        def choose_action(self, state):
            if np.random.random() < self.epsilon:
                return np.random.randint(self.actions)
            else:
                action_values = [self.get_value(state, a) for a in range(self.actions)]
                return np.argmax(action_values)

        def update(self, state, action, reward, next_state):
            next_values = [self.get_value(next_state, a) for a in range(self.actions)]
            best_next_value = max(next_values)

            old_value = self.get_value(state, action)
            td_target = reward + self.gamma * best_next_value
            td_error = td_target - old_value
            new_value = old_value + self.alpha * td_error
            self.set_value(state, action, new_value)

    class Predictor(nn.Module):
        """Neural predictor for span scale estimation."""
        def __init__(self, dims):
            super().__init__()
            self.linear = Linear(in_features=dims, out_features=1)
            nn.init.xavier_normal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        def forward(self, global_out):
            if global_out.dim() > 2:
                global_out = global_out.mean(dim=1)
            scale = torch.sigmoid(self.linear(global_out))
            return scale

    class AdaptiveSpan(BaseAttention):
        """Attention with adaptive span size."""
        def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
            super().__init__(dims, head, max_dist)
            self.sharpen = sharpen
            self.temp_scale = temp_scale
            self.span_scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, query, key, value, max_dist=None, max_span=None, span_scale=None, is_causal=True):
            if max_dist is None:
                max_dist = self.max_dist
            if max_span is None:
                max_span = query.shape[1]
            if span_scale is None:
                span_scale = self.span_scale
                
            span_mean = span_scale.mean().item()
            span_len = min(int(max_span * span_mean), query.shape[1], key.shape[1], value.shape[1])
            eff_span = min(span_len, max_dist)
            
            if eff_span == 0:
                batch = query.shape[0]
                return (torch.zeros(batch, eff_span, self.dims, device=query.device), None)
                
            q_span = query[:, :eff_span, :]
            k_span = key[:, :eff_span, :]
            v_span = value[:, :eff_span, :]

            batch = q_span.shape[0]

            q = self._shape(q_span, q_span.size(1), batch)
            k = self._shape(k_span, k_span.size(1), batch)
            v = self._shape(v_span, v_span.size(1), batch)

            temperature = (1.0 + self.temp_scale * (1.0 - span_mean)
                if self.sharpen
                else 0.5 + self.temp_scale * span_mean)
            
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                attn_output, weights = calculate_attention(
                    q, k, v, None, temperature, BaseAttention.use_sdpa, is_causal=is_causal)
                out = self._reshape_to_output(attn_output, batch, eff_span)
            return out, weights

    class MyelinatedLayer(BaseAttention):
        def __init__(self, dims, head, layerA=3, sparsity_threshold=0.1, max_dist=512):
            super().__init__(dims, head, max_dist)
            self.layers = nn.ModuleList()
            self.layerA = layerA
            self.sparsity_threshold = sparsity_threshold
            self.max_dist = max_dist
            
            self.node_predictors = nn.ModuleList([
                nn.Sequential(LayerNorm(dims),
                            Linear(dims, 1),
                            nn.Sigmoid()) for _ in range(layerA)])
            
            for i in range(layerA):
                self.layers.append(nn.ModuleDict({
                    'ln': LayerNorm(dims),
                    'gate': nn.Sequential(Linear(dims, 1), nn.Sigmoid()),
                    'adapter': Linear(dims, dims) if i % 2 == 0 else None
                }))
            self.policy_net = nn.Sequential(Linear(dims, 128), nn.ReLU(), Linear(128, 3))
            self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
            
            mlp = dims * 4
            self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.mlp = nn.Sequential(Linear(dims, mlp), nn.GELU(), Linear(mlp, dims))
            self.mlp_ln = LayerNorm(dims)
            
            self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
            self.memory_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
            self.last_memory_gate_values = None

        def compute_attention(self, norm_x, mask=None, kv_cache=None, is_causal=True):
            """Compute attention with adaptive span and content-dependent updates."""
            batch, ctx = norm_x.shape[:2]
            
            q = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
            k = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
            v = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)

            attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa, is_causal=is_causal)
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)
            return attn_output

        def predict_node_importance(self, x, layer_idx):
            """Dynamically determine if processing should occur at this node."""
            importance = self.node_predictors[layer_idx](x)
            return (importance > self.sparsity_threshold).float()

        def decide_jump(self, policy, jump_weights, i, layerA, x, original_x, working_memory):
            """Decide whether to jump layers based on the policy network."""
            jump_prob = policy[:, 1] if i < layerA - 1 else torch.zeros_like(policy[:, 1])
            should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
            if should_jump:
                jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
                i_next = min(i + jump_length, layerA - 1)
                skip_weight = jump_weights[min(jump_length - 1, 2)]
                x = x + skip_weight * original_x + (1 - skip_weight) * working_memory
                return x, i_next
            return x, i + 1

        def forward(self, x, xa=None, mask=None, kv_cache=None, is_causal=True):
            batch, ctx = x.shape[:2]
            working_memory = self.working_memory.expand(batch, -1, -1)
            original_x = x
            pooled_representation = x.mean(dim=1, keepdim=False)
            policy_logits = self.policy_net(pooled_representation)
            policy = F.softmax(policy_logits, dim=-1)
            jump_history = []
            memory_gate = torch.zeros(batch, 1, 1, device=x.device)
            
            i = 0
            while i < self.layerA:
                layer = self.layers[i]
                node_importance = self.predict_node_importance(x, i)
                print(f"Node importance (Layer {i}): {node_importance}")

                if node_importance.mean() < 0.2 and i > 0:
                    i += 1
                    jump_history.append(i)
                    continue
                norm_x = layer['ln'](x)
                attn_mask = mask * node_importance.squeeze(-1).unsqueeze(1) if mask is not None else node_importance.squeeze(-1).unsqueeze(1)
                
                if node_importance.mean() > 0.3:
                    attn_output = self.compute_attention(norm_x, mask=attn_mask, kv_cache=kv_cache)
                    print(f"Attention output (Layer {i}): {attn_output}")
                    
                    if layer['adapter'] is not None:
                        attn_output = layer['adapter'](attn_output)
                    gate_value = layer['gate'](norm_x)
                    x = x + gate_value * attn_output
                    print(f"Updated representation (Layer {i}): {x}")
                    
                    memory_gate = self.memory_gate(x.mean(dim=1, keepdim=True))
                    mean_x = x.mean(dim=1, keepdim=True)
                    working_memory = memory_gate * working_memory + (1 - memory_gate) * mean_x
                    print(f"Memory gate value: {memory_gate}")
                
                x, i = self.decide_jump(policy, self.jump_weights, i, self.layerA, x, original_x, working_memory)
                jump_history.append(i)

            self.last_memory_gate_values = memory_gate.detach().clone()
            print(f"Jump history: {jump_history}")
            mlp_importance = self.mlp_gate(x)
            mlp_output = self.mlp(self.mlp_ln(x))
            x = x + mlp_importance * mlp_output
            print(f"Final output: {x}")
            return x

    class IntegratedAttention(nn.Module):
        def __init__(self, dims, head, max_dist=512, win_size=256, max_span=384, temp_scale=0.01):
            super().__init__()
            self.head = head
            self.max_dist = max_dist

            self.dims = dims
            self.max_span = max_span
            self.sliding_window = win_size
            self.temp_scale = temp_scale
            self.sharpen = True
            self.head_dim = dims // head

            self.refiner = Refiner(states=10000, actions=10, alpha=0.1, gamma=0.9, epsilon=0.1)
            self.span_pred = Predictor(dims=dims)
            
            self.attn_local = AdaptiveSpan(
                dims=dims, head=head, max_dist=max_dist, sharpen=True, temp_scale=temp_scale)
            
            self.attn_global = MyelinatedLayer(dims=dims, head=head)
            self.cross_attn = MyelinatedLayer(dims=dims, head=head)

            self.self_projection = Linear(in_features=2 * dims, out_features=dims)
            self.cross_projection = Linear(in_features=dims, out_features=dims)
            
            self.ln_a = LayerNorm(normalized_shape=dims)
            self.ln_b = LayerNorm(normalized_shape=dims)
            self.ln_cross = LayerNorm(normalized_shape=dims)

            mask = torch.empty(max_span, max_span).fill_(float("-inf")).triu_(diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)
            self.register_buffer("window_mask", None, persistent=False)
            self.register_buffer("threshold", torch.tensor(1e-4), persistent=False)
            self.register_buffer("s_factor", torch.tensor(0.1), persistent=False)

        def forward(self, x, xa=None, mask=None, kv_cache=None, is_causal=True):
            batch, ctx = x.shape[:2]
            
            if xa is not None:
                x_norm = self.ln_cross(x)
                
                cross_out = self.cross_attn(
                    q=x_norm, k=xa, v=xa, mask=mask)
                return self.cross_projection(cross_out)
            
            local = self.ln_a(x)
            globe = self.ln_b(x)

            globe_out = self.attn_global(globe, xa=None, mask=mask, kv_cache=kv_cache, is_causal=is_causal)
            globe_out = self.cross_projection(globe_out)
            
            freq_scale = self.span_pred(globe_out)
            state = self.extract(local)
            action = self.refiner.choose_action(state=state)
            refine = self.action_scale(action=action)
            span_scale = torch.clamp(freq_scale * refine, min=0.0, max=1.0)
            span_mean = span_scale.mean().item()

            with torch.no_grad():
                current_win_size = max(1, int(self.sliding_window * span_mean))
                current_span_len = max(1, int(self.max_span * span_mean))
                effective_max = min(self.max_dist, local.size(1))
                local_max = min(self.max_dist, current_span_len, current_win_size)
                globe_max = effective_max

            self.attn_local.max_dist = local_max
            self.attn_global.max_dist = globe_max

            local_out = self.slide_win(
                x=local,
                win_size=current_win_size,
                span_len=current_span_len,
                span_scale=span_scale,
                mask=mask,
            )
            
            with torch.no_grad():
                quality = self.quality(output=local_out)
                next_state = self.extract(local_out)
                self.refiner.update(
                    state=state, action=action, reward=quality, next_state=next_state)
            
            combined = torch.cat([local_out, globe_out], dim=-1)
            return self.self_projection(combined)

        def quality(self, output):
            """Calculate quality metric for reinforcement learning."""
            with torch.no_grad():
                safe_output = torch.clamp(output, min=1e-10)
                entropy = -(safe_output * torch.log(safe_output)).sum(-1).mean()
                coverage = (output > 0.01).float().mean()
                return float(coverage - 0.1 * entropy)

        def extract(self, x):
            """Extract state features for RL agent."""
            with torch.no_grad():
                pooled = x.reshape(-1, self.dims)
                meadims = pooled.mean(dim=0)
                var_state = pooled.var(dim=0, unbiased=False)
                state = torch.cat([meadims, var_state])
                state_id = self.discretize(state.cpu().numpy())
            return state_id

        def discretize(self, state):
            """Convert continuous state to discrete state ID."""
            bins = np.linspace(-1, 1, num=10)
            state_discrete = np.digitize(state, bins)
            state_hash = sum(val * (10**i) for i, val in enumerate(state_discrete[:20]))
            state_id = int(state_hash % (self.refiner.states - 1))
            return state_id

        def action_scale(self, action):
            """Convert discrete action to continuous scale factor."""
            span_value = action / (self.refiner.actions - 1)
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            span_scale = torch.tensor([span_value], device=device, dtype=dtype)
            return span_scale

        def _focus(self, query, key, value, span_scale, mask=None):
            """Iterative attention refinement with zero-padding for invalid tokens."""
            max_iterations = 10
            iteration = 0
            prev_attn = torch.zeros_like(input=query)
            attn_out = torch.zeros_like(input=query)
            attn_weights = None

            threshold = self.threshold.item()
            s_factor = self.s_factor.item()

            while iteration < max_iterations:
                span_len = int(self.max_span * span_scale.mean().item())
                span_len = min(span_len, query.size(1), key.size(1), value.size(1))
                eff_span = min(span_len, self.max_dist)

                if eff_span == 0:
                    break

                q_span = query[:, :eff_span, :]
                k_span = key[:, :eff_span, :]
                v_span = value[:, :eff_span, :]

                batch, ctx, dims = q_span.size()
                
                q = q_span.view(batch, ctx, self.head, -1).transpose(1, 2)
                k = k_span.view(batch, ctx, self.head, -1).transpose(1, 2)
                v = v_span.view(batch, ctx, self.head, -1).transpose(1, 2)

                if self.sharpen:
                    temperature = 1.0 + self.temp_scale * (1.0 - span_scale.mean().item())
                else:
                    temperature = 0.5 + self.temp_scale * span_scale.mean().item()  # noqa: F841
                
                scale = (dims // self.head) ** -0.5
                attn = torch.matmul(q, k.transpose(-1, -2)) * scale
                
                if mask is not None:
                    if mask.dim() == 4:
                        q_len, k_len = q.size(2), k.size(2)
                        mask_q_len = min(mask.size(2), q_len)
                        mask_k_len = min(mask.size(3), k_len)
                        mask_part = mask[:, :, :mask_q_len, :mask_k_len]
                        if mask_part.dtype == torch.bool:
                            attn_part = attn[:, :, :mask_q_len, :mask_k_len]
                            masked_attn_part = attn_part.masked_fill(mask_part, float("-inf"))
                            new_attn = attn.clone()
                            new_attn[:, :, :mask_q_len, :mask_k_len] = masked_attn_part
                            attn = new_attn
                        else:
                            attn_part = attn[:, :, :mask_q_len, :mask_k_len]
                            masked_attn_part = attn_part + mask_part
                            new_attn = attn.clone()
                            new_attn[:, :, :mask_q_len, :mask_k_len] = masked_attn_part
                            attn = new_attn
                
                attn = F.softmax(attn, dim=-1)
                
                if mask is not None and mask.dtype == torch.bool:
                    q_len, k_len = q.size(2), k.size(2)
                    mask_q_len = min(mask.size(2), q_len)
                    mask_k_len = min(mask.size(3), k_len)
                    binary_mask = (~mask[:, :, :mask_q_len, :mask_k_len]).float()
                    attn_part = attn[:, :, :mask_q_len, :mask_k_len]
                    masked_attn_part = attn_part * binary_mask
                    attn_sum = masked_attn_part.sum(dim=-1, keepdim=True)
                    normalized_attn_part = masked_attn_part / (attn_sum + 1e-6)
                    new_attn = attn.clone()
                    new_attn[:, :, :mask_q_len, :mask_k_len] = normalized_attn_part
                    attn = new_attn
                    
                attn_output = torch.matmul(attn, v)
                attn_out = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)
                diff = torch.abs(attn_out - prev_attn).mean()
                dynamic_threshold = threshold + s_factor * diff
                if diff < dynamic_threshold:
                    break

                prev_attn = attn_out.clone()
                query = query + attn_out
                iteration += 1
            return attn_out, attn_weights


        def slide_win(self, x, win_size, span_len, span_scale, mask=None):
            """Process input with sliding window attention."""
            batch, ctx, dims = x.size()
            num_windows = (ctx + win_size - 1) // win_size
            output = torch.zeros_like(x)

            for i in range(num_windows):
                start_idx = i * win_size
                end_idx = min((i + 1) * win_size, ctx)
                window_size = end_idx - start_idx  # noqa: F841

                key_start = max(0, start_idx - span_len + win_size)
                key_end = min(start_idx + span_len, ctx)

                query = x[:, start_idx:end_idx, :]
                key = x[:, key_start:key_end, :]
                value = key

                window_mask = None
                if mask is not None:
                    if mask.dim() == 4:
                        window_mask = mask[:, :, start_idx:end_idx, key_start:key_end]
                        
                        if window_mask.size(1) == 1:
                            window_mask = window_mask.expand(-1, self.head, -1, -1)

                attn_out, _ = self._focus(
                    query=query,
                    key=key,
                    value=value,
                    span_scale=span_scale,
                    mask=window_mask)
                output[:, start_idx:end_idx, :] = attn_out
            return output


    class AttentionWrapper(nn.Module):
        """Wrapper to standardize attention layer interfaces"""
        
        def __init__(self, attention_layer):
            super().__init__()
            self.attention = attention_layer
            
        def forward(
            self,
            x: Tensor,
            xa: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            kv_cache: Optional[dict] = None
        ) -> Tuple[Tensor, Optional[Tensor]]:
            result = self.attention(x, xa, mask, kv_cache)
            
            if isinstance(result, tuple):
                return result
            else:
                return (result, None)
