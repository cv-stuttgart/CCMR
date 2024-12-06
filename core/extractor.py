import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        if x.shape[1] != y.shape[1]: # for the uplayers.
            return y
       
        return self.relu(x+y)


class BasicEncoder_resconv(nn.Module):
    def __init__(self, output_dim=128, norm_fn='group', model_type="CCMR+"):
        super(BasicEncoder_resconv, self).__init__()
        self.norm_fn = norm_fn
        self.model_type = model_type
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        #Top-down feature extraction

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(160, stride=2)
        self.conv2 = nn.Conv2d(160, 160, kernel_size=1)

        #Bottom-up feature consolidation
        
        self.in_planes = 160 + 128
        self.up_layer2 = self._make_layer(128, stride=1)
        self.after_up_layer2_conv = nn.Conv2d(128, 128, kernel_size=1) 
        self.in_planes = 128 + 96
        self.up_layer1 = self._make_layer(96, stride=1)
        self.after_up_layer1_conv = nn.Conv2d(96, 96, kernel_size=1)

        if self.model_type.casefold() == "ccmr+":
            self.in_planes = 96 + 64
            self.up_layer0 = self._make_layer(64, stride=1)
            self.after_up_layer0_conv = nn.Conv2d(64, 64, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        enc_out1 = x
        x = self.layer2(x)
        enc_out2 = x
        x = self.layer3(x)
        enc_out3 = x

       
        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x        
        
        #uplayer2:
        cur_h, cur_w = list(enc_out3.size())[-2:]
        enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.after_up_layer2_conv(self.up_layer2(up2layer_input))

        #uplayer1:
        cur_h, cur_w = list(enc_out2.size())[-2:]
        up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.after_up_layer1_conv(self.up_layer1(up1layer_input))

        if self.model_type.casefold() == "ccmr+":
            #uplayer0:
            cur_h, cur_w = list(enc_out1.size())[-2:]
            up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
            up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
            up0_out = self.after_up_layer0_conv(self.up_layer0(up0layer_input))
            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            up2_out = torch.split(up2_out, [batch_dim, batch_dim], dim=0)
            up1_out = torch.split(up1_out, [batch_dim, batch_dim], dim=0)
            up0_out = torch.split(up0_out, [batch_dim, batch_dim], dim=0)

            return [enc_out4, up2_out, up1_out, up0_out]    
        elif self.model_type.casefold() == "ccmr":
            enc_out4 = torch.split(enc_out4, [batch_dim, batch_dim], dim=0)
            up2_out = torch.split(up2_out, [batch_dim, batch_dim], dim=0)
            up1_out = torch.split(up1_out, [batch_dim, batch_dim], dim=0)
            
            return [enc_out4, up2_out, up1_out]    

        
class Basic_Context_Encoder_resconv(nn.Module):
    def __init__(self, output_dim=128, norm_fn='group', model_type="CCMR+"):
        super(Basic_Context_Encoder_resconv, self).__init__()
        self.norm_fn = norm_fn
        self.model_type = model_type
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        #Top-down feature extraction

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(160, stride=2)
        self.conv2 = nn.Conv2d(160, output_dim, kernel_size=1)
        
        #Bottom-up feature consolidation
        self.in_planes = 256 + 128
        self.up_layer2 = self._make_layer(128, stride=1)
        self.after_up_layer2_conv = nn.Conv2d(128, output_dim, kernel_size=1)
        self.in_planes = 256 + 96
        self.up_layer1 = self._make_layer(96, stride=1)
        self.after_up_layer1_conv = nn.Conv2d(96, output_dim, kernel_size=1) 
        if self.model_type.casefold() == "ccmr+":
            self.in_planes = 256 + 64
            self.up_layer0 = self._make_layer(64, stride=1)
            self.after_up_layer0_conv = nn.Conv2d(64, output_dim, kernel_size=1) 
       

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):
      
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        enc_out1 = x
        x = self.layer2(x)
        enc_out2 = x
        x = self.layer3(x)
        enc_out3 = x

        x = self.layer4(x)
        x = self.conv2(x)
        enc_out4 = x
        
        #uplayer2:
        cur_h, cur_w = list(enc_out3.size())[-2:]
        enc_out4_resized = TF.resize(enc_out4, (cur_h, cur_w))
        up2layer_input = torch.cat((enc_out4_resized, enc_out3), dim=1)
        up2_out = self.after_up_layer2_conv(self.up_layer2(up2layer_input))
        #uplayer1:
        cur_h, cur_w = list(enc_out2.size())[-2:]
        up2_out_resized = TF.resize(up2_out, (cur_h, cur_w))
        up1layer_input = torch.cat((up2_out_resized, enc_out2), dim=1)
        up1_out = self.after_up_layer1_conv(self.up_layer1(up1layer_input))
        if self.model_type.casefold() == "ccmr+":
            #uplayer0:
            cur_h, cur_w = list(enc_out1.size())[-2:]
            up1_out_resized = TF.resize(up1_out, (cur_h, cur_w))
            up0layer_input = torch.cat((up1_out_resized, enc_out1), dim=1)
            up0_out = self.after_up_layer0_conv(self.up_layer0(up0layer_input))
            return [enc_out4, up2_out, up1_out, up0_out]  
        
        elif self.model_type.casefold() == "ccmr":  
            return [enc_out4, up2_out, up1_out] 
