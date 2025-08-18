import torch
import torch.nn  as nn
from torch import Tensor
from PIL import Image
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def prepare_image_ids(img_size: int, patch_size: int, batch_size: int) -> Tensor:
    img_ids = torch.zeros(img_size // patch_size, img_size // patch_size, 3) # (14, 14, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(img_size // patch_size)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(img_size // patch_size)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b = batch_size)
    
    return img_ids

def vit_rope(pos: Tensor, dim: int, theta: int):
    assert dim % 2 == 0
    
    # pos shape: [1, 196] 	 dim: 8 	 theta: 10000
    # pos shape: [1, 196] 	 dim: 44 	 theta: 10000
    # pos shape: [1, 196] 	 dim: 44 	 theta: 10000
    
    scale = torch.arange(0, dim, 2) / dim
    # scale: torch.Size([4])
    # scale: torch.Size([22])
    # scale: torch.Size([22])
    
    freqs = 1.0 / (theta ** scale)
    # freqs: torch.Size([4])
    # freqs: torch.Size([22])
    # freqs: torch.Size([22])
    
    freqs_out = torch.einsum("...n, d -> ...nd", pos, freqs)
    # freqs_out at enisum: torch.Size([1, 196, 4])
    # freqs_out at enisum: torch.Size([1, 196, 22])
    # freqs_out at enisum: torch.Size([1, 196, 22])
    
    freqs_out = torch.stack([torch.cos(freqs_out), -torch.sin(freqs_out), torch.sin(freqs_out), torch.cos(freqs_out)], dim = -1)
    # freqs_out at stack sin cosin: torch.Size([1, 196, 4, 4])
    # freqs_out at stack sin cosin: torch.Size([1, 196, 22, 4])
    # freqs_out at stack sin cosin: torch.Size([1, 196, 22, 4])
    
    freqs_out = rearrange(freqs_out, "b n d (i j) -> b n d i j", i=2, j=2)
    # freqs_out at rearrange: torch.Size([1, 196, 4, 2, 2])
    # freqs_out at rearrange: torch.Size([1, 196, 22, 2, 2])
    # freqs_out at rearrange: torch.Size([1, 196, 22, 2, 2])
    
    return freqs_out

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        
    def forward(self, ids: Tensor):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [vit_rope(ids[...,1], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim = -3
        )
        
        return emb.unsqueeze(1)
    
def apply_vit_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> Tensor:
    # xqk shape: torch.Size([1, 8, 196, 96]) 	torch.Size([1, 8, 196, 96]) 	torch.Size([1, 1, 196, 48, 2, 2])
    xq_cls, xq = xq[:, :, :1, :], xq[:, : , 1:, :]
    xk_cls, xk = xk[:, :, :1, :], xk[:, : , 1:, :]
    
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    
    freqs_cis = freqs_cis.to(device)
    # xqk reshape: torch.Size([1, 8, 196, 48, 1, 2]) 	torch.Size([1, 8, 196, 48, 1, 2])
    xq_out = freqs_cis[...,0] * xq_[...,0] + freqs_cis[..., 1] * xq_[...,1]
    xk_out = freqs_cis[...,0] * xk_[...,0] + freqs_cis[..., 1] * xk_[...,1]
    
    # out xqk: torch.Size([1, 8, 196, 48, 2]) 	torch.Size([1, 8, 196, 48, 2])
    xq_out = xq_out.reshape(*xq.shape)
    xk_out = xk_out.reshape(*xk.shape)
    
    xq_out = torch.cat([xq_cls, xq_out], dim=2)
    xk_out = torch.cat([xk_cls, xk_out], dim=2)
    
    # out xqk reshape: torch.Size([1, 8, 196, 96]) 	torch.Size([1, 8, 196, 96])
    return xq_out, xk_out

class patchEmbedding(nn.Module):
    def __init__(self, in_channel: int = 3, emb_size: int = 768, patch_size: int = 16, img_size: int = 224, batch_size : int = 8, num_heads : int = 8, pos_type: str = 'linear'):
        super().__init__()
        
        self.projection = nn.Sequential(
            # input shape [1, 3, 224, 224]
            nn.Conv2d(in_channels=in_channel, out_channels=emb_size, kernel_size=patch_size, stride=patch_size, padding=0),
            # now shape is [1, 768, 14, 14] 
            # image height / patch size = 224/16 =14
            Rearrange('b e (h) (w) -> b (h w) e')
            # rearrange to new shape [1, 196, 768]
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        if pos_type == 'linear':
            self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, emb_size))
        
        elif pos_type == 'sinusoidal':
            max_seq_length = (img_size // patch_size)**2 + 1
            positions = torch.zeros(max_seq_length, emb_size) 
            
            
            for pos in range(max_seq_length):
                for i in range(emb_size):
                    if i % 2 == 0:
                        positions[pos][i] = np.sin(pos/(10000 ** (i/emb_size)))
                    else:
                        positions[pos][i] = np.cos(pos/(10000 ** ((i-1)/emb_size)))
            
            # self.register_buffer('positions', positions.unsqueeze(0)) # for excludes it from training parameters
            self.positions = nn.Parameter(positions.unsqueeze(0))
            
        elif pos_type == 'rope':
            self.axes_dim = [8, 44, 44]
            self.theta = 10000
            self.img_ids = prepare_image_ids(img_size=img_size, patch_size=patch_size, batch_size=batch_size)
            self.pe_dim = emb_size // num_heads
            self.pe_embedds = EmbedND(self.pe_dim, self.theta, self.axes_dim)
            self.pe = self.pe_embedds(self.img_ids)
            
            
        
    def forward(self, x:Tensor) -> Tensor:
        x = x.to(device)
        
        b, _, _, _ = x.shape
        
        # input shape [1, 3, 224, 224]
        x = self.projection(x)
        # now shape [1, 196, 768]
        
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        # cls shape [1, 1, 768]
        
        x = torch.cat([cls_token, x], dim = 1)
        # now x shape [1, 197, 768]
        
        # x += self.positions
        # # shape [1, 197, 768]
        if hasattr(self, 'positions'):
            x += self.positions
            return x, None
        
        elif hasattr(self, 'pe'):
            return x, self.pe
        
        # return x
        


class multiHeadAttention(nn.Module):
    def __init__(self, emb_size:int = 768, num_heads:int = 8, dropouts: float = 0.1):
        super().__init__()
        
        self.emb_size = emb_size
        self.num_head = num_heads
        
        self.query = nn.Linear(self.emb_size, self.emb_size)
        self.key = nn.Linear(self.emb_size, self.emb_size)
        self.value = nn.Linear(self.emb_size, self.emb_size)
        
        self.att_dropout = nn.Dropout(dropouts)
        
        self.projection = nn.Linear(self.emb_size, self.emb_size)
        
    def forward(self, x:Tensor, pe: Tensor = None) -> Tensor:
        # input [1, 197, 768]
        queries = rearrange(self.query(x), "b n (h d) -> b h n d", h = self.num_head) # output [1, 8, 197, 96]
        keys = rearrange(self.key(x), "b n (h d) -> b h n d", h = self.num_head) # output [1, 8, 197, 96]
        values = rearrange(self.value(x), "b n (h d) -> b h n d", h = self.num_head) # output [1, 8, 197, 96]
        
        if pe is not None:
            
            queries, keys = apply_vit_rope(queries, keys, pe)
            
        
        
        # now q dot k
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # output shape [1, 8, 197, 197]
        scaling = self.emb_size ** (1/2)
        # softmax(qk/sqrt(key emb dimension))
        att = F.softmax(energy/scaling, dim=-1)
        att = self.att_dropout(att)
        
        # score dot v
        out = torch.einsum('bhal, bhlv -> bhav', att, values) # output [1, 8, 197, 96]
        out = rearrange(out, 'b h n d -> b n (h d)') # output [1, 197, 768]
        out = self.projection(out) # output [1, 197, 768]
        
        return out
        

        
class Resudial(nn.Module):
    def __init__(self, fn):
        super().__init__()
        
        self.fn = fn        
        
    def forward(self, x, **kwargs):
        
        res = x
        x = self.fn(x, **kwargs)
        x += res
        
        return x
        
        
class feedForward(nn.Module):
    def __init__(self, emb_size:int = 768, expansion: int = 4, drop_p:float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, emb_size*expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(emb_size*expansion, emb_size)
        )

    def forward(self, x):
        return self.net(x)
        
        
class TrabsformerEncoderBlock(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads : int = 8, drop_out: float = 0.1, forward_expansion: int = 4, froward_drop_p: float = 0, **kwargs):
        # super().__init__(
        #     Resudial(nn.Sequential(
        #         nn.LayerNorm(emb_size),
        #         multiHeadAttention(emb_size, **kwargs),
        #         nn.Dropout(drop_out)
        #     )),
        #     Resudial(nn.Sequential(
        #         nn.LayerNorm(emb_size),
        #         feedForward(emb_size, expansion=forward_expansion, drop_p=froward_drop_p),
        #         nn.Dropout(drop_out)
        #     ))
        # )
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = multiHeadAttention(emb_size=emb_size, num_heads=num_heads)
        self.dropout1 = nn.Dropout(drop_out)
        
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = feedForward(emb_size=emb_size, expansion=forward_expansion, drop_p=froward_drop_p)
        self.dropout2 = nn.Dropout(drop_out)
        
        
    def forward(self, x, pe=None):
        # attntion + residual
        attn_out = self.attn(self.norm1(x), pe = pe)
        x = x + self.dropout1(attn_out)
        
        # feedforward + residual
        ff = self.ff(self.norm2(x))
        x = x + ff
        
        return x
        
 
        
class transformerEncoder(nn.Module):
    def __init__(self, depth:int = 12, **kwargs):
        super().__init__()
        self.layer = nn.ModuleList([TrabsformerEncoderBlock(**kwargs) for _ in range(depth)])
        
    def forward(self, x, pe=None):
        for layer in self.layer:
            x = layer(x, pe=pe)
            
        return x
        
        
               
# class classication(nn.Module):
#     def __init__(self, emb_size: int = 768, num_class: int = 1000):
#         super().__init__(
#             Reduce('b n e -> b e', reduction = 'mean'),
#             nn.LayerNorm(emb_size),
#             nn.Linear(emb_size, num_class)
#         )

class classication(nn.Module):
    def __init__(self, emb_size: int = 768, num_class: int = 1000):
        super().__init__()
        self.classifier = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_class)
        )

    def forward(self, x):
        return self.classifier(x)
        
        
        
class vit(nn.Module):
    def __init__(self, in_channel: int = 3, img_size: int = 224, patch_size: int = 16, 
                 num_heads: int = 8, embedding_size: int = 768, batch_size: int = 8, 
                 depth: int = 12, num_class: int = 1000, pos_type: str = "linear", **kwargs):
        super().__init__()
        self.patch_embeeding = patchEmbedding(in_channel=in_channel, emb_size=embedding_size, 
                                              patch_size=patch_size, img_size=img_size, 
                                              batch_size=batch_size, num_heads=num_heads, 
                                              pos_type=pos_type)
        self.encoder = transformerEncoder(depth=depth, emb_size=embedding_size, num_heads = num_heads, **kwargs)
        self.classifier = classication(emb_size=embedding_size, num_class=num_class)
        
        
    def forward(self, x):
        x, pe = self.patch_embeeding(x)
        x = self.encoder(x, pe=pe)
        x = self.classifier(x)
        
        return x
        