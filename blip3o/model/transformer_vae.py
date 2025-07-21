import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PretrainedConfig, PreTrainedModel

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward

# --- VAE Config ---
class TransformerVAEConfig(PretrainedConfig):
    model_type = "transformer-vae"

    def __init__(
        self,
        hidden_size: int = 1152,
        latent_dim: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
        rope: bool = True,
        qk_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope = rope
        self.qk_norm = qk_norm

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale

class Qwen2BidirectionalSdpaAttention(nn.Module):
    def __init__(self, config: TransformerVAEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim ** -0.5
        self.qk_norm = getattr(config, "qk_norm", False)
        if self.qk_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        attn_output, _ = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            is_causal=False,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

class TransformerVAEEncoderLayer(nn.Module):
    def __init__(self, config: TransformerVAEConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2BidirectionalSdpaAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class TransformerVAEDecoderLayer(nn.Module):
    def __init__(self, config: TransformerVAEConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2BidirectionalSdpaAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class TransformerVAE(PreTrainedModel):
    config_class = TransformerVAEConfig

    def __init__(self, config: TransformerVAEConfig):
        super().__init__(config)
        self.latent_dim = config.latent_dim
        self.encoder_layers = nn.ModuleList(
            [TransformerVAEEncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [TransformerVAEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Qwen2RotaryEmbedding(config=config) if config.rope else None
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.to_mu = nn.Linear(config.hidden_size, config.latent_dim)
        self.to_logvar = nn.Linear(config.hidden_size, config.latent_dim)
        self.latent_to_hidden = nn.Linear(config.latent_dim, config.hidden_size)
        self.post_init()

    def encode(self, hidden_states):
        bsz, seq_len, _ = hidden_states.size()
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if self.rotary_emb else None

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, position_embeddings)
        hidden_states = self.norm(hidden_states)
        pooled = hidden_states.mean(dim=1)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        bsz = z.size(0)
        hidden_states = self.latent_to_hidden(z).unsqueeze(1).expand(-1, seq_len, -1)
        position_ids = torch.arange(seq_len, device=z.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids) if self.rotary_emb else None

        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, position_embeddings)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def forward(self, inputs, seq_len=None):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        if seq_len is None:
            seq_len = inputs.size(1)
        recon = self.decode(z, seq_len)
        return recon, mu, logvar