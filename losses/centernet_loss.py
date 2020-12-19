import torch
from losses.commons import modified_focal_loss,reg_l1_loss






class CenterNetLoss(object):
    def __init__(self, cls_weight, wh_weight, reg_weight):
        self.cls_weight = cls_weight
        self.wh_weight = wh_weight
        self.reg_weight = reg_weight

    def __call__(self, pred_dict, gt_dict):
        """
        alculate losses of pred and gt

        Args:
            gt_dict(dict): a dict contains all information of gt
            gt_dict = {
               "score_map": gt scoremap     shape=[bs,num_cls,h,w],
               "wh": gt width and height of boxes    shape=(bs,128,2),
               "reg": gt regression of box center point   shape=(bs,128,2),
               "reg_mask": mask of regression   shape=(bs,128),
               "index": gt index   shape=(bs,128),
                }
            pred(dict): a dict contains all information of prediction
            pred = {
                "cls": predicted score map    shape=[bs,num_cls,h,w]
                "reg": predcited regression     shape=[bs,2,h,w] 2==>(dx,dy)对应量化误差的预测
                "wh": predicted width and height of box   shape=[bs,2,h,w]  2对应原图输入尺度的bbox宽高预测
                }
        """
        # scoremap loss
        pred_score = pred_dict['cls']
        if pred_score.dtype == torch.float16:
            pred_score = pred_score.float()
        cur_device = pred_score.device
        for k in gt_dict:
            gt_dict[k] = gt_dict[k].to(cur_device)

        loss_cls = modified_focal_loss(pred_score, gt_dict['score_map'])

        mask = gt_dict['reg_mask']
        index = gt_dict['index']
        index = index.to(torch.long)
        # width and height loss, better version
        loss_wh = reg_l1_loss(pred_dict['wh'], mask, index, gt_dict['wh'])

        # regression loss
        loss_reg = reg_l1_loss(pred_dict['reg'], mask, index, gt_dict['reg'])

        loss_cls *= self.cls_weight
        loss_wh *= self.wh_weight
        loss_reg *= self.reg_weight

        loss = loss_cls+loss_wh+loss_reg

        return loss, torch.stack([loss_cls,loss_wh,loss_reg]).detach()




