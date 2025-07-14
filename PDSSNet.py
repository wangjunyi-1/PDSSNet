import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
from .SC_VSS2 import C_VSSBlock
from .Patching import PatchEmbed2D, Final_PatchExpand2D
from .P_VSS1 import HierarchicalVSSBlock

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class WS(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WS, self).__init__()
        self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
        self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

    def forward(self, x, res, ade):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x + fuse_weights[2] * ade
        x = self.post_conv(x)
        return x


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()
class Writingnet(nn.Module):
    def __init__(self, input_feature_dim, feature_dim):
        super(Writingnet, self).__init__()
        assert input_feature_dim == feature_dim, "Should match when residual mode is on ({} != {})".format(
            input_feature_dim, feature_dim)
        self.writefeat = nn.Sequential(
            nn.Conv2d(input_feature_dim, feature_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(feature_dim)
        )
        self.relu = nn.ReLU(inplace=True)
        initialize_weights(self)

    def forward(self, x):
        output = x + self.writefeat(x)
        output = self.relu(output)
        return output
class Memory_sup(nn.Module):
    def __init__(self, memory_size, input_feature_dim, feature_dim, momentum, patch_size, dim_scale, gumbel_read=True,
                 m_min=0.1, m_max=0.9,
                 ):
        super(Memory_sup, self).__init__()

        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.input_feature_dim = input_feature_dim
        self.initial_momentum = momentum
        self.m_min = m_min
        self.m_max = m_max
        self.gumbel_read = gumbel_read
        self.patch_size = patch_size
        self.dim_scale = dim_scale
        self.conv1=nn.Conv2d(5*feature_dim,feature_dim//2,kernel_size=1)
        self.conv2=nn.Conv2d(feature_dim, feature_dim//2, kernel_size=1)

        self.output = nn.Sequential(
            nn.Conv2d(feature_dim * 2, input_feature_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(input_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.writenet = Writingnet(input_feature_dim, feature_dim)
        self.mem_cls = torch.tensor([x for x in range(self.memory_size)]).cuda()
        self.clsfier = nn.Linear(in_features=self.feature_dim, out_features=self.memory_size, bias=True)
        self.celoss = nn.CrossEntropyLoss(ignore_index=255)
        self.m_n_raw = nn.Parameter(torch.ones(memory_size, dtype=torch.float) * momentum, requires_grad=True)
        self.m_items = F.normalize(torch.rand((memory_size, feature_dim), dtype=torch.float), dim=1).cuda()
        self.P_vssblock = HierarchicalVSSBlock(hidden_dim=feature_dim)
        self.patchembed = PatchEmbed2D(patch_size=self.patch_size, in_chans=feature_dim, embed_dim=feature_dim)
        self.final_H = Final_PatchExpand2D(dim=feature_dim, dim_scale=dim_scale)
        self.final_W = Final_PatchExpand2D(dim=feature_dim, dim_scale=dim_scale)
        self.upconv_h = nn.Conv2d(in_channels=feature_dim // dim_scale, out_channels=feature_dim, kernel_size=1)
        self.upconv_w = nn.Conv2d(in_channels=feature_dim // dim_scale, out_channels=feature_dim, kernel_size=1)
        self.conv = ConvBN(2 * feature_dim, feature_dim, kernel_size=1)
        initialize_weights(self)
        self.final = Final_PatchExpand2D(dim=feature_dim, dim_scale=self.dim_scale)
        self.upconv_p = nn.Conv2d(in_channels=feature_dim // self.dim_scale, out_channels=feature_dim,
                                  kernel_size=1)
        self.wf = WF(in_channels=feature_dim, decode_channels=feature_dim)
        self.spatial_modulation = nn.Conv2d(feature_dim, memory_size * feature_dim,
                                        kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 16, bias=False),
            nn.ReLU(),
            nn.Linear(feature_dim // 16, feature_dim, bias=False),
            nn.Sigmoid()
        )

    def get_m_n(self):
        m_n = torch.sigmoid(self.m_n_raw)
        m_n = self.m_min + (self.m_max - self.m_min) * m_n
        return m_n
    def write(self, input, M_1, mask, writing_detach=True):
        if mask is None:
            return [torch.tensor(0).cuda(), torch.tensor(0).cuda()]
        tempmask = mask.clone().detach()
        query = input.clone()
        if not writing_detach:
            query = self.writeTF(query)
        query = self.writenet(query)
        query = F.normalize(query, dim=1)
        B, C, H, W = query.size()
        tempmask[tempmask == 6] = self.memory_size
        tempmask = F.one_hot(tempmask, num_classes=self.memory_size + 1)
        tempmask = F.interpolate(tempmask.permute(0, 3, 1, 2).float(), [H, W], mode='bilinear', align_corners=True)
        tempmask_flat = tempmask.permute(0, 2, 3, 1).view(B, -1, self.memory_size + 1)
        M_1_prime = F.normalize(M_1, dim=1)
        M_1_flat = M_1_prime.view(B, C, H * W)
        feature_sum_batch = torch.matmul(M_1_flat, tempmask_flat)
        pixel_count_batch = tempmask_flat.sum(dim=1, keepdim=True)
        total_feature_sum = feature_sum_batch.sum(dim=0)
        total_pixel_count = pixel_count_batch.sum(dim=0) + 1e-8
        per_class_avg_feature = total_feature_sum / total_pixel_count
        new_feature_part = per_class_avg_feature.t()[:self.memory_size, :]
        old_memory = self.m_items.clone().detach()
        old_memory = F.normalize(old_memory, dim=1)
        updated_memory = 0.7 * old_memory + 0.3 * new_feature_part
        if writing_detach:
            self.m_items = updated_memory.detach()
        else:
            self.m_items = updated_memory
        return 0

    def read(self, query, Structure):

        Structure = F.normalize(Structure.clone(), dim=1)
        B, C, H, W = Structure.size()
        M_0 = self.m_items
        assert self.feature_dim == C, "feature_dim must equal C"
        modulation = self.spatial_modulation(Structure)
        M_0 = M_0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        M_0 = M_0.expand(B, self.memory_size, C, H, W).contiguous().view(B, self.memory_size * C, H, W)  #
        M_0 = M_0 * torch.sigmoid(modulation)
        M_0_compressed = self.conv1(M_0)
        Structure_compressed = self.conv2(Structure)
        S_n = torch.cat([M_0_compressed, Structure_compressed], dim=1)  #
        f = self.patchembed(S_n)
        f = self.P_vssblock(f)
        f = self.final(f)
        f = f.permute(0, 3, 1, 2)
        M_1 = self.upconv_p(f)
        updated_query = self.wf(M_1, query)

        return updated_query


    def forward(self,Structure,query,mask=None, memory_writing=True, writing_detach=True):
        updated_query = self.read(query,Structure)
        if memory_writing and mask is not None:
            self.write(query,updated_query, mask, writing_detach)
        return updated_query



def SobelFilter(input_tensor):

    device = input_tensor.device
    input_tensor = input_tensor.float()
    sobel_x = torch.tensor([
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]]
    ], dtype=torch.float, device=device)
    sobel_y = torch.tensor([
        [[[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]]
    ], dtype=torch.float, device=device)
    C = input_tensor.shape[1]
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)
    Gx = F.conv2d(input_tensor, sobel_x, padding=1, groups=C)
    Gy = F.conv2d(input_tensor, sobel_y, padding=1, groups=C)
    edge_magnitude = torch.sqrt(Gx ** 2 + Gy ** 2 + 1e-10)
    return edge_magnitude



class SANet(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 dropout=0.1,
                 backbone_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=True,
                 patch_size=8,
                 num_classes=6,
                 use_aux_loss=True,
                 dim_scale=8
                 ):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.patch_size = patch_size
        self.backbone = timm.create_model(model_name=backbone_name, features_only=True, pretrained=pretrained,
                                           output_stride=32, out_indices=(0, 1, 2,3))
        self.num_classes = num_classes
        self.conv2 = ConvBN(2 * decode_channels, decode_channels, kernel_size=1)
        self.conv3 = ConvBN(4 * decode_channels, decode_channels, kernel_size=1)
        self.conv4 = ConvBN(8* decode_channels, decode_channels, kernel_size=1)
        self.conv7 = ConvBN(96, decode_channels, kernel_size=1)
        self.dim_scale = dim_scale

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.down = Conv(in_channels=3 * decode_channels, out_channels=decode_channels, kernel_size=1)
        self.decoder_channels = decode_channels
        self.patchembed = PatchEmbed2D(patch_size=self.patch_size, in_chans=decode_channels, embed_dim=decode_channels)

        self.final = Final_PatchExpand2D(dim=decode_channels, dim_scale=self.dim_scale)
        self.ws = WS(in_channels=decode_channels, decode_channels=decode_channels)
        self.wf = WF(in_channels=decode_channels, decode_channels=decode_channels)

        self.conv5 = nn.Conv2d(in_channels=3 * decode_channels, out_channels=decode_channels, kernel_size=3, stride=2,
                               padding=1)
        self.conv6 = nn.Conv2d(in_channels=3 * decode_channels + 2, out_channels=decode_channels, kernel_size=3,
                               stride=2,
                               padding=1)
        self.patchembed_L = PatchEmbed2D(patch_size=self.dim_scale, in_chans=2 * decode_channels,
                                         embed_dim=2 * decode_channels)

        self.C_vssblock = C_VSSBlock(hidden_dim=2 * decode_channels)

        self.final_H = Final_PatchExpand2D(dim=decode_channels, dim_scale=self.dim_scale)

        self.final_W = Final_PatchExpand2D(dim=decode_channels, dim_scale=self.dim_scale)

        self.upconv_h = nn.Conv2d(in_channels=decode_channels // self.dim_scale, out_channels=decode_channels,
                                  kernel_size=1)
        self.upconv_w = nn.Conv2d(in_channels=decode_channels // self.dim_scale, out_channels=decode_channels,
                                  kernel_size=1)
        self.upconv_p = nn.Conv2d(in_channels=decode_channels // self.dim_scale, out_channels=decode_channels,
                                  kernel_size=1)
        self.mem_slot = num_classes - 1
        self.memory = Memory_sup(memory_size=self.mem_slot, input_feature_dim=decode_channels,
                                 feature_dim=decode_channels, momentum=0.5, patch_size=patch_size, dim_scale=dim_scale)

        self.memory_initialized = False


    def forward(self, x, mask=None, imagename=None):

        C, H, W = x.size()[-3:]
        res1, res2, res3, res4 = self.backbone(x)
        res1h, res1w = res1.size()[-2:]
        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        sobel = SobelFilter(res1)
        middleres = torch.cat([res2, res3, res4], dim=1)
        middleres = self.conv5(middleres)
        if mask is not None and not self.memory_initialized:
            gt = mask.clone()
            gt = gt.reshape(-1, H, W).cuda()
            gt[gt == 6] = self.mem_slot
            query = middleres.clone()
            query = F.normalize(query, dim=1)
            batch_size, dims, h, w = query.size()
            gt_one_hot = F.one_hot(gt, num_classes=self.mem_slot + 1)
            gt_one_hot = F.interpolate(gt_one_hot.permute(0, 3, 1, 2).type(torch.float32), [h, w],
                                       mode='bilinear', align_corners=True)
            Ck = gt_one_hot[:, :self.mem_slot, :, :].sum(dim=(2, 3)) + 1e-8
            alpha_k = 1 / (torch.log(Ck + 1) + 1e-6)
            query_flat = query.view(batch_size, dims, -1)
            gt_flat = gt_one_hot.view(batch_size, self.mem_slot + 1, -1)
            nominator = torch.matmul(query_flat, gt_flat[:, :self.mem_slot, :].transpose(1,
                                                                                         2))
            weighted_nominator = nominator * alpha_k.unsqueeze(1)
            basket = weighted_nominator.sum(0)
            weighted_denominator = (alpha_k * Ck).sum(0)
            weighted_denominator[weighted_denominator == 0] = 1
            init_prototypes = basket.t() / weighted_denominator.unsqueeze(1)
            self.memory.m_items = F.normalize(init_prototypes, dim=1)
            self.memory_initialized = True
        middleres = F.interpolate(middleres, size=(self.decoder_channels, self.decoder_channels), mode='bicubic',
                                  align_corners=False)
        sobel = F.interpolate(sobel, size=(self.decoder_channels, self.decoder_channels), mode='bicubic', align_corners=False)

        update_query = self.memory(sobel,middleres,mask)

        middlechannels_feature_H = update_query.permute(0, 2, 3, 1)
        middlechannels_feature_W = update_query.permute(0, 3, 2, 1)
        C_H_W = torch.cat([middlechannels_feature_H, middlechannels_feature_W], dim=1)
        f_l = self.patchembed_L(C_H_W)
        f_l = self.C_vssblock(f_l)
        f_h, f_w = f_l.chunk(2, dim=-1)
        f_h = self.final_H(f_h)
        f_h = f_h.permute(0, 3, 1, 2)
        f_w = self.final_W(f_w)
        f_w = f_w.permute(0, 3, 2, 1)
        f_h = self.upconv_h(f_h)
        f_w = self.upconv_w(f_w)
        f_c = f_h + f_w



        res = F.interpolate(f_c, size=(res1h, res1w), mode='bicubic', align_corners=False)
        middleres = F.interpolate(middleres, size=(res1h, res1w), mode="bicubic", align_corners=False)
        res = self.ws(res, res1, middleres)



        res = self.segmentation_head(res)

        if self.training:
            x = F.interpolate(res, size=(H, W), mode='bilinear', align_corners=False)
            return x
        else:
            x = F.interpolate(res, size=(H, W), mode='bilinear', align_corners=False)
            return x

