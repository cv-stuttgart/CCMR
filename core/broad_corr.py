import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler
try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AltCorrFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, radius):
        ctx.radius = radius
        ctx.save_for_backward(fmap1, fmap2, coords)

        corr, = alt_cuda_corr.forward(fmap1, fmap2, coords, radius)

        return corr

    @staticmethod
    def backward(ctx, d_corr):
        radius = ctx.radius
        fmap1, fmap2, coords = ctx.saved_tensors
        d_corr = d_corr.contiguous()
        d_fmap1, d_fmap2, d_coords = alt_cuda_corr.backward(fmap1, fmap2, coords, d_corr, radius)

        return d_fmap1, d_fmap2, d_coords, None


def alt_corr(fmap1, fmap2, coords, radius):
    return AltCorrFunction.apply(fmap1, fmap2, coords, radius)

class CudaBroadCorrBlock:
    def __init__(self, fmap1, fmap2, correlation_depth):
        self.b, self.feature_dim, self.h, self.w = fmap1.shape
        self.corr_depth = correlation_depth

        self.radius = 4
        self.fmap1 = fmap1
        self.fmap2 = [fmap2]

        fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
        self.fmap2.append(fmap2)

    def __call__(self, coords1):
        r = self.radius

        fmap1_i = self.fmap1.permute(0, 2, 3, 1).contiguous()       # b, h1, w1, c
        coords1 = coords1.permute(0, 2, 3, 1)                       # b, h, w, 2

        corr_list = []
        for i, fmap2_i in enumerate(self.fmap2):
            fmap2_i = fmap2_i.permute(0, 2, 3, 1).contiguous()      # b, h2, w2, c

            coords_i = (coords1 / 2**i).reshape(self.b, 1, self.h, self.w, 2).contiguous()
            corr = alt_corr(fmap1_i, fmap2_i, coords_i, r)          # b, 1, (2r+1)^2, h, w
            corr_list.append(corr.squeeze(1))                       # b, (2r+1)^2, h, w

        corrs = torch.stack(corr_list, dim=1)
        corrs = corrs.reshape(self.b, self.corr_depth, self.h, self.w)
        corrs = corrs / torch.sqrt(torch.tensor(self.feature_dim).float())

        return corrs