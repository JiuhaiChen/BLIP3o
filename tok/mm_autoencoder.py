import torch
import torch.nn as nn

from tok.ar_dtok.ar_model import ARModel
from tok.ar_dtok.vqvae import VQVAE
from tok.ta_tok import TextAlignedTokenizer


class MMAutoEncoder(nn.Module):
    def __init__(self,
        ar_path,
        encoder_path, decoder_path, 
        encoder_args={}, decoder_args={}):
        super().__init__()
        self.ar_model = ARModel.from_checkpoint(ar_path)

        self.encoder = TextAlignedTokenizer.from_checkpoint(encoder_path, load_teacher=False, **encoder_args)
        self.decoder = VQVAE.from_checkpoint(decoder_path, **decoder_args)

    def ar_sample(self, x, args):
        x = self.ar_model.sample(
            x,
            cfg_scale=args.get('cfg_scale', 1.0),
            cfg_interval=args.get('cfg_interval', -1),
            temperature=args.get('temperature', 1.0),
            top_k=args.get('top_k', 0),
            top_p=args.get('top_p', 1.0)
        )
        return x

    def post_process(self, x):
        x = x.cpu().float().clamp(0., 1.) * 255.
        x = x.permute(0, 2, 3, 1) # [b, h, w, c]
        x = x.to(torch.uint8)
        return x
    
    def encode(self, x):
        return self.encoder(x.to(self.encoder.dtype))['encoded']
    
    def get_encoder_indices(self, x):
        # img -> encoder -> indices
        return self.encoder(x.to(self.encoder.dtype))['bottleneck_rep']
    
    @torch.inference_mode()
    def decode_from_encoder_indices(self, indices, args={}):
        # indices -> encoder feats -> ar -> decoder
        encoder_x = self.encoder.decode_from_bottleneck(indices)
        ar_indices = self.ar_sample(encoder_x, args)
        decoder_x = self.decoder.decode_from_bottleneck(ar_indices)
        x = self.post_process(decoder_x)
        return x
    
    def decode_from_vqvae_indices(self, indices):
        decoder_x = self.decoder.decode_from_bottleneck(indices)
        x = self.post_process(decoder_x)
        return x
    
    @torch.inference_mode()
    def forward(self, x, args={}):
        encoder_x = self.encoder(x.to(self.encoder.dtype))['encoded']
        ar_indices = self.ar_sample(encoder_x, args)
        decoder_x = self.decoder.decode_from_bottleneck(ar_indices)
        x = self.post_process(decoder_x)
        return x