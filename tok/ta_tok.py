import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import Resize
from transformers import AutoConfig, AutoModel, Siglip2VisionConfig, Siglip2VisionModel

from . import models
from .utils import ScalingLayer


class TextAlignedTokenizer(nn.Module):
    def __init__(
        self, 
        bottleneck,
        bottleneck_token_num=256,
        input_size=384,
        teacher='google/siglip2-so400m-patch14-384',
        input_type='quant', # choose from ['quant', 'rec', 'indices']
        pool_scale=1, # choose from [1, 2, 3]
        decoder_depth=3,
        select_layer_id=-2,
        *args,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.bottleneck_token_num = bottleneck_token_num
        self.teacher = teacher
        self.input_type = input_type
        self.pool_scale = pool_scale
        self.decoder_depth = decoder_depth
        self.select_layer_id = select_layer_id
       
        self.bottleneck_dim = bottleneck['args']['bottleneck_dim']

        self.encoder_config = AutoConfig.from_pretrained(teacher)
        self.encoder = AutoModel.from_config(self.encoder_config).vision_model         
        
        self.encoder_hidden_dim = self.encoder.config.hidden_size

        self.decoder_config = Siglip2VisionConfig()
        self.decoder_config.update({
            'patch_size': 1,
            'num_hidden_layers': self.decoder_depth,
            'num_channels': self.bottleneck_dim,
            'hidden_size': self.encoder_hidden_dim,
        })
        self.decoder = Siglip2VisionModel(self.decoder_config)

        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
            nn.Tanh())
        self.decode_task_layer = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim))

        bottleneck_args = {
            'token_nums': self.bottleneck_token_num, 
            'input_dim': self.encoder_hidden_dim, 
            'output_dim': self.bottleneck_dim}
        self.bottleneck = models.make(bottleneck, args=bottleneck_args)

        self.scale_layer = ScalingLayer(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   
        self.image_resize = Resize((self.input_size, self.input_size))
       
    def set_vq_eval_deterministic(self, deterministic=True):
        self.bottleneck.regularizer.set_eval_deterministic(deterministic)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
    @classmethod
    def from_checkpoint(cls, ckpt, load_teacher=True, **kwargs):
        ckpt = torch.load(ckpt, map_location='cpu')
        ckpt_kwargs = ckpt["model"]["args"]
        model = cls(**kwargs, **ckpt_kwargs)
        sd = ckpt["model"]["sd"]
        if not load_teacher:
            sd = {k: v for k, v in sd.items() if not k.startswith('teacher')}
        model.load_state_dict(sd, strict=True)
        return model

    def encode(self, x, **kwargs):
        if x.ndim == 5:
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.scale_layer(x)
        if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = self.image_resize(x)
        vq_feats = self.encoder(x, output_hidden_states=True).hidden_states[self.select_layer_id]

        pool_scale = self.pool_scale
        pool_scale = kwargs.get("pool_scale", pool_scale)
        if pool_scale != 1:
            vq_feats = self.avg_pool(vq_feats, pool_scale)
        vq_feats = self.encode_task_layer(vq_feats.to(x))
        
        bottleneck_out = self.bottleneck(vq_feats)
        z = bottleneck_out.pop('output')

        return {'encoded': z, 'pool_scale': pool_scale, 'vq_feats': vq_feats, **bottleneck_out}

    def avg_pool(self, z, pool_scale=1):
        if z.ndim == 3:
            b, n, c = z.shape
            p = int(n ** 0.5)
            z = rearrange(z, 'b (p1 p2) c -> b c p1 p2', p1=p, p2=p)
        else:
            b, c, p, _ = z.shape
        p_s = int(p // pool_scale)
        z = F.avg_pool2d(
            z,
            kernel_size=(pool_scale, pool_scale),
            stride=(pool_scale, pool_scale)
        ).contiguous()
        z = rearrange(z, 'b c p1 p2 -> b (p1 p2) c')
        return z

    def decode(self, z):
        if z.ndim == 4:
            z = rearrange(z, 'b c p1 p2 -> b (p1 p2) c')
        attention_mask = torch.ones(z.shape[:2], dtype=torch.int, device=z.device)
        p = int(z.shape[1]**0.5)
        spatial_shape = torch.tensor([[p, p]]*z.shape[0], device=self.device)
        z = self.decoder(z, attention_mask, spatial_shape, output_hidden_states=True).last_hidden_state
        z = self.decode_task_layer(z)
        return z

    def decode_from_bottleneck(self, bottleneck_rep):
        z = self.bottleneck.decode(bottleneck_rep) # (b, n, c)
        p = int(z.shape[1]**0.5)
        z = rearrange(z, 'b (p1 p2) c -> b c p1 p2', p1=p, p2=p)
        return self.decode(z)

    def forward(self, data, **kwargs):
        # data: video in shape (b, c, t, h, w)
        encode_output = self.encode(data, **kwargs)
        vq_feats = encode_output['encoded']
        p = int(vq_feats.shape[1] ** 0.5)
        vq_feats = rearrange(vq_feats, 'b (h w) c -> b c h w', h=p, w=p)
        pred_feats = self.decode(vq_feats)

        if self.input_type == 'quant':
            z = encode_output["regularized_z"] # [b, n, c]
        elif self.input_type == 'indices':
            z = encode_output["bottleneck_rep"] # [b, n]
        elif self.input_type == 'rec':
            z = pred_feats # [b, n, c]
        encode_output['encoded'] = z
        return encode_output