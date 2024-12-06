
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock
from extractor import BasicEncoder_resconv, Basic_Context_Encoder_resconv
from broad_corr import CudaBroadCorrBlock, CorrBlock
from utils.utils import coords_grid, upflow2
from update import BasicUpdateBlock
from core.xcit import XCiT
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def downflow(flow, mode='bilinear', factor=0.125):
    old_size = (flow.shape[2], flow.shape[3])
    new_size = (int(factor * flow.shape[2]), int(factor * flow.shape[3]))
    u_scale = new_size[1]/old_size[1]
    v_scale = new_size[0]/old_size[0]
    resized_flow = F.interpolate(flow, size=new_size, mode=mode, align_corners=True) #b 2 h w
    resized_flow_split = torch.split(resized_flow, 1,  dim=1)
    rescaled_flow = torch.cat([u_scale*resized_flow_split[0], v_scale*resized_flow_split[1]], dim=1)
    
    return rescaled_flow

class CCMR(nn.Module):
    def __init__(self, args):
        super(CCMR, self).__init__()
        self.args = args
        
        self.correlation_depth = 162
        
        self.hidden_dim = 128
        self.context_dim = 128

        if self.args["model_type"].casefold() == "ccmr":
            self.args["num_scales"] = 3
        elif self.args["model_type"].casefold() == "ccmr+":
            self.args["num_scales"] = 4
        else:
            raise ValueError(f'Model type "{self.args["model_type"]}" Unknown.')

        #Initiating the improved multi-scale feature encoders:
        self.fnet = BasicEncoder_resconv(output_dim=256,  norm_fn= self.args["fnet_norm"], model_type=self.args["model_type"])   
        self.cnet = Basic_Context_Encoder_resconv(output_dim=256, norm_fn=self.args["cnet_norm"], model_type=self.args["model_type"])
        self.update_block = BasicUpdateBlock(self.args, self.correlation_depth, hidden_dim=128, scale=2, num_heads=8, depth=1 , mlp_ratio=1, num_scales=self.args["num_scales"])
    
        #Initiate global context grouping modules
        self.xcit = nn.ModuleList([XCiT(embed_dim=128, depth=1, num_heads=8, mlp_ratio=1, separate=False)])
        for i in range(self.args["num_scales"] - 1):
            self.xcit.extend([XCiT(embed_dim=128, depth=1, num_heads=8, mlp_ratio=1, separate=False)])
            
        print(f"Model {self.args['model_type']} initiated! :)")

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def initialize_flow16(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//16, W//16).to(img.device)
        coords1 = coords_grid(N, H//16, W//16).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def get_grid(self, img, scale):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//scale, W//scale).to(img.device)
        return coords0

    def upsample_flow(self, flow, mask, scale=8):
        """ Upsample flow field [H/scale, W/scale, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale*H, scale*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        with autocast(enabled=self.args["mixed_precision"]):
            
            fnet_pyramid = self.fnet([image1, image2])
            # run the context network
            cnet_pyramid = self.cnet(image1)
        
        coords0, coords1 = self.initialize_flow16(image1)
        
        flow_predictions = []
        
        if flow_init is not None:
                coords1 = coords1 + flow_init

        assert len(fnet_pyramid) == len(cnet_pyramid), 'fnet and cnet pyramid should have the same length.'
        assert len(fnet_pyramid) == len(iters), 'pyramid levels and the length of GRU iteration lists should be the same.'
        upsampling_offset = self.args["num_scales"] - 1 if self.args["num_scales"]==4 else self.args["num_scales"]
        for index, (fmap1, fmap2) in enumerate(fnet_pyramid):
            fmap1 = fmap1.float()
            fmap2 = fmap2.float()
            if self.args['cuda_corr']:
                '''
                This must be used for training the CCMR+ (4-scale) model.
                If your GPUs don't have much VRAM (< 40 GBs), also use this for the CCMR (3-scale) model.
                For validating CCMR+ on large images, use this option to save memory. 
                This option can also be used for computing the flow by the (3-scale) CCMR model to save memory (for smaller GPUs).
                '''
                corr_fn = CudaBroadCorrBlock(fmap1, fmap2, self.correlation_depth) 
            else:
                corr_fn = CorrBlock(fmap1, fmap2)  
            net, inp = torch.split(cnet_pyramid[index], [128, 128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            with autocast(enabled=self.args["mixed_precision"]):
                global_context = self.xcit[index](inp)
        

            for itr in range(iters[index]):
                coords1 = coords1.detach()
                if index >= 1 and itr == 0:
                    flow = self.upsample_flow(coords1 - coords0, up_mask, scale=2) # bug of MS-RAFT's flow upsampling fixed here.
                    
                    coords0 = self.get_grid(image1, scale=16/(2**index))
                    
                    coords1 = coords0 + flow

                corr= corr_fn(coords1)
                flow = coords1 - coords0
                with autocast(enabled=self.args["mixed_precision"]):
                    net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, global_context, level_index=index)
                       
                # F(t+1) = F(t) + \Delta(t)
                coords1 = coords1 + delta_flow
                # upsample predictions
                flow_up = self.upsample_flow(coords1 - coords0, up_mask, scale=2)
                for i in range(upsampling_offset - index):
                    flow_up = upflow2(flow_up)
               
                flow_predictions.append(flow_up)
                
        if test_mode:
            flow_low = downflow(flow_up, factor=0.0625)
            return flow_low, flow_up
        
        
        return flow_predictions
