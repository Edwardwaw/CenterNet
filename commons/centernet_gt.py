import numpy as np
import torch



class CenterNetGT(object):

    @staticmethod
    def generate(targets, bs, num_classes, down_scale=4, output_size=(128,128), min_overlap=0.7, tensor_dim=128):
        box_scale = 1. / down_scale
        scoremap_list, wh_list, reg_list, reg_mask_list, index_list = [[] for _ in range(5)]


        for bi in range(bs):
            # init gt tensors
            gt_scoremap = torch.zeros(num_classes, *output_size)  # shape=[num_cls,h,w]
            gt_wh = torch.zeros(tensor_dim, 2)  #(128,2)
            gt_reg = torch.zeros_like(gt_wh)  #(128,2)
            reg_mask = torch.zeros(tensor_dim)  #(128)
            gt_index = torch.zeros(tensor_dim)  #(128)

            target = targets[targets[:, 0] == bi, 2:]  # shape=[num_gt,5] 5==>(label_idx,x1,y1,x2,y2)

            num_gt = target.shape[0]
            if num_gt > tensor_dim: 
                target = target[:tensor_dim, :]

            boxes, classes = target[:,1:], target[:,[0]]

            num_boxes = boxes.shape[0]
            boxes = boxes*box_scale
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            centers_int = centers.to(torch.int32)

            gt_index[:num_boxes] = centers_int[..., 1] * output_size[1] + centers_int[..., 0]
            gt_reg[:num_boxes] = centers - centers_int
            reg_mask[:num_boxes] = 1

            wh = torch.zeros_like(centers)
            wh[..., 0] = boxes[..., 2] - boxes[..., 0]
            wh[..., 1] = boxes[..., 3] - boxes[..., 1]

            CenterNetGT.generate_score_map(
                gt_scoremap, classes, wh,
                centers_int, min_overlap,
            )
            gt_wh[:num_boxes] = wh

            scoremap_list.append(gt_scoremap)
            wh_list.append(gt_wh)
            reg_list.append(gt_reg)
            reg_mask_list.append(reg_mask)
            index_list.append(gt_index)

        '''
        score_map: shape=[bs,num_cls,h,w]
        wh: (bs,128,2)
        reg: (bs,128,2)
        reg_mask: (bs,128)
        index: (bs,128)
        '''
        gt_dict = {
            "score_map": torch.stack(scoremap_list, dim=0),
            "wh": torch.stack(wh_list, dim=0),
            "reg": torch.stack(reg_list, dim=0),
            "reg_mask": torch.stack(reg_mask_list, dim=0),
            "index": torch.stack(index_list, dim=0),
        }
        return gt_dict

    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = CenterNetGT.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            channel_index = int(channel_index.item())
            CenterNetGT.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        box_tensor = torch.Tensor(box_size)
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CenterNetGT.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap
        # return fmap
