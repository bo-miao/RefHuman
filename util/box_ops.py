# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch, os
from torchvision.ops.boxes import box_area
import numpy as np
from torchvision.ops import roi_align

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), boxes1
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), boxes2
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)



# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area

def convert_box_to_5_index(boxes):
    # bxox4 -> bxox5
    b, o, _ = boxes.shape
    index = torch.tensor(list(range(b)), device=boxes.device).view(b, 1).repeat(1, o).unsqueeze(-1)
    return torch.cat([index, boxes], dim=-1)

def crop_object_using_bbox(feat, box, size=(4,4)):
    '''
    Args:
        feat: b,c,h,w
        box: b,o,4; v0~1
        size: (h,w)
    Returns:
        res: b*o, c, s, s
    '''
    _, c, init_h, init_w = feat.shape
    b, o, _ = box.shape
    abs_box = box.clone()
    abs_box[:, :, (0, 2)] *= init_w
    abs_box[:, :, (1, 3)] *= init_h
    abs_box = abs_box.round()
    abs_box = convert_box_to_5_index(abs_box).view(b*o, 5)
    res = roi_align(feat, abs_box, size, aligned=True)  # box should also be CUDA!
    return res


def add_bbox_perturbation(bboxes, offset_perturbation=0.1):
    """
    Add perturbation to bounding boxes considering shape, ratio, and offset.
    Parameters:
    - bboxes (numpy.ndarray): Input bounding boxes tensor of shape (b, n, 4).
    """
    device = bboxes.device
    perturbation_offset = torch.from_numpy(np.random.uniform(-offset_perturbation, offset_perturbation, size=bboxes.shape)).to(device)
    # Apply perturbations
    perturbed_bboxes = bboxes.clone()
    perturbed_bboxes[:, :, [0, 2]] += perturbation_offset[:, :, [0, 2]] * (perturbed_bboxes[:, :, 2:3]-perturbed_bboxes[:, :, 0:1])
    perturbed_bboxes[:, :, [1, 3]] += perturbation_offset[:, :, [1, 3]] * (perturbed_bboxes[:, :, 3:4]-perturbed_bboxes[:, :, 1:2])
    perturbed_bboxes = torch.clamp(perturbed_bboxes, 0, 1)
    return perturbed_bboxes

if __name__ == '__main__':
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)