import torch
import math
import numpy as np
from torch import nn
from nets.common import CenternetDeconv
from commons.centernet_deocode import CenterNetDecoder




def switch_backbones(bone_name):
    from nets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
    if bone_name == "resnet18":
        return resnet18()
    elif bone_name == "resnet34":
        return resnet34()
    elif bone_name == "resnet50":
        return resnet50()
    elif bone_name == "resnet101":
        return resnet101()
    elif bone_name == "resnet152":
        return resnet152()
    elif bone_name == "resnext50_32x4d":
        return resnext50_32x4d()
    elif bone_name == "resnext101_32x8d":
        return resnext101_32x8d()
    elif bone_name == "wide_resnet50_2":
        return wide_resnet50_2()
    elif bone_name == "wide_resnet101_2":
        return wide_resnet101_2()
    else:
        raise NotImplementedError(bone_name)



class SingleHead(nn.Module):

    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x



class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    :return
    cls: shape=[bs,num_cls,h,w]
    wh: shape=[bs,2,h,w]  2对应原图输入尺度的bbox宽高预测
    reg: shape=[bs,2,h,w] 2==>(dx,dy)对应量化误差的预测
    """
    def __init__(self, num_cls, bias_value):
        super(CenternetHead, self).__init__()
        self.cls_head = SingleHead(
            64,
            num_cls,
            bias_fill=True,
            bias_value=bias_value,
        )
        self.wh_head = SingleHead(64, 2)
        self.reg_head = SingleHead(64, 2)

    @torch.no_grad()
    def decode_prediction(self, pred_dict):
        """
        Args:
            pred_dict(dict): a dict contains all information of prediction
            img_info(dict): a dict contains needed information of origin image
        :return
        boxes: shape=[-1, 4]
        scores:  shape=[-1]
        classes:  shape=[-1]
        """
        fmap = pred_dict["cls"]  # NCHW
        reg = pred_dict["reg"]  # N2HW
        wh = pred_dict["wh"]  # N2HW

        device = wh.device

        boxes, scores, classes = CenterNetDecoder.decode(fmap, wh, reg)  # [batch,K,4/1/1] batch=1 here
        scores = scores.reshape(-1)
        classes = classes.reshape(-1).to(torch.int64)
        boxes = CenterNetDecoder.transform_boxes(boxes)

        boxes = torch.from_numpy(boxes).to(device=device, dtype=torch.float32)

        valid_index = scores > 0.05  # box score thresh
        detects = torch.cat([
                             boxes[valid_index],
                             scores[valid_index].unsqueeze(-1),
                             classes[valid_index].unsqueeze(-1)
                             ],dim=-1)

        # detects.shape=[num_box,6]  6==>x1,y1,x2,y2,score,label
        ret_dets=list()
        ret_dets.append(detects)
        return ret_dets


    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        pred = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        if self.training:
            return pred
        else:
            results = self.decode_prediction(pred)
            return results







class CenterNet(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """
    def __init__(self,
                 num_cls=80,
                 PIXEL_MEAN=[0.485, 0.456, 0.406],
                 PIXEL_STD=[0.229, 0.224, 0.225],
                 backbone='resnet50',
                 cfg=None
                 ):
        super(CenterNet,self).__init__()
        self.cfg = cfg
        self.num_cls = num_cls
        self.mean = PIXEL_MEAN
        self.std = PIXEL_STD
        self.backbone = switch_backbones(backbone)
        self.upsample = CenternetDeconv(self.cfg['DECONV_CHANNEL'], self.cfg['DECONV_KERNEL'], self.cfg['MODULATE_DEFORM'])
        self.head = CenternetHead(num_cls, self.cfg['BIAS_VALUE'])

    def forward(self, x):
        '''
        note: 作验证或者推理时,x.shape=[1,C,H,W]
        :param x:
        :return:
        '''

        features = self.backbone(x)
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)

        return pred_dict






if __name__ == '__main__':
    input_tensor = torch.randn(size=(1, 3, 512, 512)).cuda()
    centernet_cfg=dict(
            DECONV_CHANNEL=[2048, 256, 128, 64],
            DECONV_KERNEL=[4, 4, 4],
            NUM_CLASSES=80,
            MODULATE_DEFORM=True,
            BIAS_VALUE=-2.19,
            DOWN_SCALE=4,
            MIN_OVERLAP=0.7,
            TENSOR_DIM=128,
        )
    net = CenterNet(backbone="resnet50",cfg=centernet_cfg).cuda()
    net.train()
    pred_dict = net(input_tensor)
    cls_out,wh_out,reg_out = pred_dict['cls'],pred_dict['wh'],pred_dict['reg']
    print(cls_out.shape,cls_out.dtype,cls_out.device)
    print(wh_out.shape,wh_out.dtype,wh_out.device)
    print(reg_out.shape,reg_out.dtype,reg_out.device)

    # net.eval()
    # pred_dict=net(input_tensor)
    # boxes,scores,classes=pred_dict['pred_boxes'],pred_dict['scores'],pred_dict['pred_classes']
    # print('pred box info: ',boxes.shape,boxes.dtype,boxes.device)
    # print('score info: ',scores.shape,scores.dtype,scores.device)
    # print('pred classes info: ',classes.shape,classes.dtype,classes.device)
    # print(boxes.max(),boxes.min())
    # print(scores.max(),scores.min())
    # print(classes.max(),classes.min())




