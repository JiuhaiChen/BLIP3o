import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .. import models
from ..models import register


@register("bottleneck")
class Bottleneck(nn.Module):
    def __init__(
        self,
        bottleneck_dim: int,
        input_dim: int,
        output_dim: int,
        token_nums: int,
        regularizer=None,
        **kwargs
    ):  
        super().__init__()
        self.token_nums = token_nums
        self.input_dim = input_dim
        self.output_dim = output_dim
        if bottleneck_dim > 0:
            self.bottleneck_dim = bottleneck_dim
        else:
            assert self.input_dim == self.output_dim, "input_dim and output_dim must be the same when bottleneck_dim is not specified"
            self.bottleneck_dim = self.input_dim
        
        self.project_dim = self.bottleneck_dim

        if self.bottleneck_dim > 0:
            self.in_linear = nn.Linear(self.input_dim, self.project_dim)
            self.out_linear = nn.Linear(self.bottleneck_dim, self.output_dim)
        else:
            self.in_linear = self.out_linear = lambda x: x
        
        regularizer['args']['dim'] = self.bottleneck_dim
        regularizer['args']['token_nums'] = self.token_nums
        self.regularizer = models.make(regularizer)

    def project_in(self, x):
        assert len(x.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        z = self.in_linear(x)
        return z

    def project_out(self, z_cat):
        z = self.out_linear(z_cat)
        return z

    def decode(self, bottleneck_rep):
        regularized_z = self.regularizer.decode(bottleneck_rep)
        return self.project_out(regularized_z)

    def forward(self, x):  
        z = self.project_in(x)
        projected_z = z
        regularized_output = self.regularizer(z)
        x_hat = self.project_out(regularized_output['regularized_z'])
        bottleneck_rep = regularized_output.pop('bottleneck_rep')
        return {
            'output': x_hat,
            'bottleneck_rep': bottleneck_rep,
            'projected_z': projected_z,
            **regularized_output,
        }


@register("simvq")
class SimVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        l2_normalized=False,
        same_index_shape=True,
        stochastic=False,
        stochastic_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        assert isinstance(l2_normalized, bool)
        self.l2_normalized = l2_normalized
        self.stochastic = stochastic
        self.eval_deterministic = False
        self.default_stochastic_temperature = stochastic_temperature
        
        if self.stochastic:
            if stochastic_temperature > 0: # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else: # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        # for clear inference code, we remove the codebook init from LLM's embedding
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        self.embedding_proj = nn.Linear(self.dim, self.dim)

        self.same_index_shape = same_index_shape

    def set_eval_deterministic(self, deterministic=True):
        self.eval_deterministic = deterministic

    def set_stochastic_temperature(self, temperature):
        self.stochastic_temperature_inv = 1 / temperature

    @torch.autocast(device_type='cuda', enabled=False)
    def get_emb(self):
        emb = self.embedding_proj(self.embedding.weight)
        if self.l2_normalized:
            emb = F.normalize(emb, p=2, dim=-1)
        # assert emb.dtype == torch.float32, f"Embedding weight dtype is {emb.dtype}, expected float32"
        return emb

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, z):
        emb = self.get_emb()
        z = z.to(emb)
        # z = z.float()
        assert len(z.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        if self.l2_normalized:
            z = F.normalize(z, p=2, dim=-1)

        z_flattened = rearrange(z, 'b n d -> (b n) d')

        if self.stochastic:
            # sample the softmaxed cosine similarity
            assert self.l2_normalized, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, emb)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                q_indices = torch.argmax(probs, dim=-1)
            else:
                q_indices = torch.multinomial(probs, 1).squeeze(-1)
        else:
            d = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(emb**2, dim=1)
                - 2
                * torch.einsum(
                    "bd,dn->bn", z_flattened, rearrange(emb, "n d -> d n")
                )
            )
            q_indices = torch.argmin(d, dim=1)

        quantized = F.embedding(q_indices, emb, self.embedding.padding_idx, self.embedding.max_norm,
            self.embedding.norm_type, self.embedding.scale_grad_by_freq, self.embedding.sparse).view(z.shape)  # (b, n, d)
        
        # preserve gradients
        quantized = z + (quantized - z).detach()

        if self.same_index_shape:
            q_indices = q_indices.reshape(quantized.shape[0], quantized.shape[1])

        return_dict = {
            'unregularized_z': z, # but l2 normalized if l2_normalized=True
            'emb': emb, # but l2 normalized if l2_normalized=True
            'regularized_z': quantized,
            'bottleneck_rep': q_indices
        }
        return return_dict
    
    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        indices_shape = indices.shape
        indices_flatten = rearrange(indices, '... -> (...)')

        # get quantized latent vectors
        emb = self.get_emb()
        z_q = F.embedding(indices_flatten, emb)
        # z_q = self.embedding(indices_flatten)
        if self.l2_normalized:
            z_q = F.normalize(z_q, p=2, dim=-1)

        if shape is not None:
            z_q = z_q.reshape(shape)
        else:
            z_q = z_q.reshape([*indices_shape, self.dim])
        return z_q

    def decode(self, indices):
        return self.get_codebook_entry(indices)