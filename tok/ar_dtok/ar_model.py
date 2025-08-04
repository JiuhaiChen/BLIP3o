import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .. import models
from .generate import generate as ar_generate


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, scale_factor=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    scale_factor: the base for the scaling factor, default is 10000
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / scale_factor**omega  # Parameterized scaling factor (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32

    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    model_type: str = 'class_cond' # clip_cond, indice_cond
    cond_dim: int = 1152
    cond_vocab_size: int = 8192

    vocab_size: int = 8192
    cls_token_num: int = 1

    max_batch_size: int = 32
    max_seq_len: int = 2048

    use_fixed_pe: bool = False

    frame_prediction: bool = False


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @torch.autocast(device_type='cuda', enabled=False)
    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                            Drop Path Implementation                           #
#################################################################################

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


#################################################################################
#                                   AR Model                                    #
#################################################################################

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2], f"{input_pos.shape[0]} != {k_val.shape[2]}"
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val.to(k_out.dtype)
        v_out[:, :, input_pos] = v_val.to(v_out.dtype)

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        # replace all negative labels with the last class (unconditional class)
        labels = torch.where(labels < 0, self.num_classes, labels)
        embeddings = self.embedding_table(labels)
        return embeddings


class ARModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.max_seq_length = config.max_seq_len
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.is_sampling = False
        self.frame_prediction = config.frame_prediction

        if self.model_type == 'class_cond':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 'clip_cond':
            self.clip_proj = nn.Linear(config.cond_dim, config.dim)
        elif self.model_type == 'indice_cond':
            self.clip_proj = LabelEmbedder(config.cond_vocab_size + 1, config.dim, 0.0)
        else:
            raise Exception("please check model type")
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.use_fixed_pe:
            self.register_buffer('abs_pe', torch.zeros(1, config.max_seq_len + config.cls_token_num - 1, config.dim))
            abs_pe = get_1d_sincos_pos_embed_from_grid(embed_dim=config.dim, pos=np.arange(config.max_seq_len + config.cls_token_num - 1))
            self.abs_pe.copy_(torch.from_numpy(abs_pe).float().reshape_as(self.abs_pe))
            print(f"Using fixed absolute PE")
        else:
            self.abs_pe = nn.Parameter(torch.randn(1, config.max_seq_len + config.cls_token_num - 1, config.dim) * 0.02)
            print(f"Using learned absolute PE")

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        if hasattr(self.output, 'weight') and isinstance(self.output.weight, nn.Parameter):
            nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def dtype(self):
        return next(self.parameters()).dtype
    

    @contextmanager
    def sampling(self):
        self.is_sampling = True
        try:
            yield
        finally:
            self.is_sampling = False


    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        assert max_seq_length == self.max_seq_length + self.cls_token_num, f'{max_seq_length} != {self.max_seq_length} + {self.cls_token_num=}'

        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)

        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(max_batch_size, 1, 1)


    def reset_caches(self):
        for b in self.layers:
            b.attention.kv_cache = None

    def clip_embedding(self, x):
        if self.model_type == 'clip_cond':
            if self.training and self.config.class_dropout_prob > 0:
                drop_ids = torch.rand(x.shape[0], device=x.device) < self.config.class_dropout_prob
                x[drop_ids] = 0.
            x = self.clip_proj(x.to(self.dtype)) # Linear
        elif self.model_type == 'indice_cond':
            if self.training and self.config.class_dropout_prob > 0:
                drop_ids = torch.rand(x.shape[0], device=x.device) < self.config.class_dropout_prob
                x[drop_ids] = self.config.cond_vocab_size
            x = self.clip_proj(x, train=self.training) # Embedding
        return x

    def forward(
        self, 
        idx: Optional[torch.Tensor], # (b, n)
        cond_idx: Optional[torch.Tensor],  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None: # training or naive inference
            if self.model_type == 'class_cond':
                cond_embeddings = self.cls_embedding(cond_idx, train=self.training).unsqueeze(1)[:,:self.cls_token_num]
            elif self.model_type in ['clip_cond', 'indice_cond']:
                cond_embeddings = self.clip_embedding(cond_idx)
            token_embeddings = self.tok_embeddings(idx) # (b, n, d)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)  # (b, cls_token_num + n, d)
            h = self.tok_dropout(token_embeddings)
        else:
            if cond_idx is not None: # prefill in inference
                if self.model_type == 'class_cond':
                    token_embeddings = self.cls_embedding(cond_idx, train=self.training).unsqueeze(1)[:,:self.cls_token_num]
                elif self.model_type in ['clip_cond', 'indice_cond']:
                    token_embeddings = self.clip_embedding(cond_idx)
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
        
        if self.is_sampling:
            h = h + self.abs_pe[:, input_pos]
        else:
            h = h + self.abs_pe[:, :h.shape[1]]
        
        # transformer blocks
        for layer in self.layers:
            h = layer(h, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h)
        # if self.training or self.is_sampling:
        if cond_idx is not None:
        # if self.training:
            # logits = logits[:, self.cls_token_num - 1:].contiguous()
            logits = logits[:, cond_idx.size(1) - 1:].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    
    @torch.inference_mode()
    def sample(
        self, 
        c,
        cfg_scale=2.0,
        cfg_interval=-1,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        seq_length=None,
    ):
        seq_length = self.max_seq_length if seq_length is None else seq_length     
        with self.sampling():
            sampled_seqs = ar_generate(
                self, c, seq_length,
                cfg_scale=cfg_scale, cfg_interval=cfg_interval,
                temperature=temperature, top_k=top_k,
                top_p=top_p, sample_logits=True, 
            )   
        return sampled_seqs
    

    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"checkpoint {ckpt} does not exist"
            ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        else:
            assert isinstance(
                ckpt, dict
            ), f"checkpoint must be a dict or a path to a checkpoint"
        model = models.make(ckpt["model"], load_sd=load_state_dict)
        return model


#################################################################################
#                             LLAMA-ABS Configs                                 #
#################################################################################

def LLAMA_ABS_XXXL(**kwargs):
    return ARModel(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def LLAMA_ABS_XXL(**kwargs):
    return ARModel(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def LLAMA_ABS_XL(**kwargs):
    return ARModel(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def LLAMA_ABS_LP(**kwargs):
    return ARModel(ModelArgs(n_layer=30, n_head=20, dim=1280, **kwargs)) # 632M

def LLAMA_ABS_L(**kwargs):
    return ARModel(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def LLAMA_ABS_B(**kwargs):
    return ARModel(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M

def LLAMA_ABS_S(**kwargs):
    return ARModel(ModelArgs(n_layer=12, n_head=6, dim=384, **kwargs)) # 21.7M

ar_models = {
    'llama-abs-S': LLAMA_ABS_S,
    'llama-abs-B': LLAMA_ABS_B,
    'llama-abs-L': LLAMA_ABS_L,
    'llama-abs-LP': LLAMA_ABS_LP,
    'llama-abs-XL': LLAMA_ABS_XL,
    'llama-abs-XXL': LLAMA_ABS_XXL,
    'llama-abs-XXXL': LLAMA_ABS_XXXL,
}

models.models.update(ar_models)
