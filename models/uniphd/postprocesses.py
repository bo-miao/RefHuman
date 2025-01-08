import torch
from torch import nn
import pycocotools.mask as mask_util
import torch.nn.functional as F
from einops import repeat, rearrange
import numpy as np
from util import box_ops


class PostProcess(nn.Module):
    def __init__(self, num_select=20, num_body_points=17) -> None:
        super().__init__()
        self.num_select = num_select
        self.num_body_points = num_body_points

    @torch.no_grad()
    def forward(self, outputs, orig_target_sizes, target_sizes, targets):
        assert target_sizes.shape[1] == 2
        origin_h, origin_w = orig_target_sizes.unbind(1)
        out_logits = outputs["pred_logits"]
        num_select = min(self.num_select, out_logits.shape[1])
        out_keypoints = outputs['pred_keypoints']
        out_boxes = outputs['pred_boxes']
        if 'pred_masks' in outputs:
            out_masks = outputs['pred_masks']

        # topk highest-scoring queries
        assert len(out_logits) == len(target_sizes)
        bs, num_queries = out_logits.shape[:2]
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), k=min(num_select, num_queries), dim=1, sorted=True)
        scores = topk_values
        labels = topk_indexes % out_logits.shape[2]

        # bbox
        topk_boxes = topk_indexes // out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_boxes)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        scale_fct = torch.stack([origin_w, origin_h, origin_w, origin_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{"scores": s, "labels": torch.ones_like(l), "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        assert len(scores) == len(labels) == len(boxes)

        # segm
        if 'pred_masks' in outputs:
            topk_msks = topk_indexes // out_logits.shape[2]
            outputs_masks = [out_m[topk_msks[i]].unsqueeze(0) for i, out_m, in enumerate(out_masks)]
            outputs_masks = torch.cat(outputs_masks, dim=0)
            for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, target_sizes, orig_target_sizes)):
                img_h, img_w = t[0], t[1]
                msk = cur_mask[:, :img_h, :img_w].unsqueeze(1).cpu()
                msk = F.interpolate(msk, size=tuple(tt.tolist()), mode="bilinear", align_corners=False) # resize to init resolution
                msk = (msk.sigmoid() > 0.5).cpu()
                results[i]["masks"] = msk.byte()
                results[i]["rle_masks"] = [mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                        for mask in results[i]["masks"].cpu()]

        # keypoints
        topk_keypoints = topk_indexes // out_logits.shape[2]
        keypoints = torch.gather(out_keypoints, 1, topk_keypoints.unsqueeze(-1).repeat(1, 1, self.num_body_points * 3))
        Z_pred = keypoints[:, :, :self.num_body_points * 2]  # bs, nq, 34
        V_pred = keypoints[:, :, self.num_body_points * 2:]  # bq, nq, 17
        Z_pred = Z_pred * torch.stack([origin_w, origin_h], dim=1).repeat(1, self.num_body_points)[:, None, :]
        keypoints_res = torch.zeros_like(keypoints)
        keypoints_res[..., 0::3] = Z_pred[..., 0::2]
        keypoints_res[..., 1::3] = Z_pred[..., 1::2]
        keypoints_res[..., 2::3] = V_pred[..., 0::1]
        assert len(keypoints_res) == len(results), 'mismatch length.'
        for i, kp in enumerate(keypoints_res):
            results[i]["keypoints"] = kp

        return results, keypoints_res[:,0]  # return highest-scoring one for PCKH@0.5 & oIoU metrics


class PostProcessPose(nn.Module):
    def __init__(self, num_select=100, num_body_points=17) -> None:
        super().__init__()
        self.num_select = num_select
        self.num_body_points = num_body_points

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, target_sizes):
        return results



class PostProcessSegm(nn.Module):
    def __init__(self, num_select=100, num_body_points=17) -> None:
        super().__init__()
        self.num_select = num_select
        self.num_body_points = num_body_points

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, target_sizes):
        return results
