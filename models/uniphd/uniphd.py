# ------------------------------------------------------------------------
# Modified from Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------




import copy

import math
import torch
import torch.nn.functional as F
from torch import nn

from util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)

from .backbones import build_backbone
from .transformer import build_transformer
from .utils import MLP
from .postprocesses import PostProcess, PostProcessPose, PostProcessSegm
from .criterion import SetCriterion
from ..registry import MODULE_BUILD_FUNCS
import random
from einops import rearrange, repeat

from .text_encoder.text_encoder import TextEncoder, FeatureResizer
from .position_encoding import PositionEmbeddingSine1D
from .segmentation import VisionLanguageFusionModule
from .decoder import MSO
from torch.cuda.amp import autocast

from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, crop_object_using_bbox, add_bbox_perturbation

class UniPHD(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, args, matcher, backbone, transformer, num_classes, num_queries,
                    aux_loss=False,
                    num_feature_levels=1,
                    nheads=8,
                    two_stage_type='no',
                    dec_pred_class_embed_share=False,
                    dec_pred_pose_embed_share=False,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    cls_no_bias = False,
                    num_body_points = 17
                    ):
        super().__init__()
        self.args = args
        self.training = not args.eval
        self.matcher = matcher
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.num_body_points = num_body_points

        # visual encoder
        bbn_last_channel = backbone.num_channels[0]
        backbone.num_channels = backbone.num_channels[1:]
        self.backbone = backbone

        ###### Prompt Encoding ######
        # Build Text Encoder
        if 'text' in self.args.train_trigger:
            print("********** Enabling Text Prompt ***************\n")
            self.text_encoder = TextEncoder(args)
            self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)
            self.text_proj = FeatureResizer(
                input_feat_size=self.text_encoder.feat_dim,
                output_feat_size=hidden_dim,
                dropout=0.1,
            )
            self.sentence_proj = FeatureResizer(
                input_feat_size=self.text_encoder.feat_dim,
                output_feat_size=hidden_dim,
                dropout=0.1,
            )
        # Prompt Encode
        if 'scribble' in self.args.train_trigger or 'point' in self.args.train_trigger or 'bbox' in self.args.train_trigger:
            print("********** Enabling Positional Prompt ***************\n")
            self.prompt_proj = FeatureResizer(
                input_feat_size=backbone.num_channels[-3:][-2],
                output_feat_size=hidden_dim,
                dropout=0.1,
            )
            self.glob_prompt_proj = FeatureResizer(
                input_feat_size=hidden_dim,
                output_feat_size=hidden_dim,
                dropout=0.1,
            )

        # multimodal fusion
        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8, modalities=self.args.train_trigger)

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.aux_loss = aux_loss

        # mask
        self.mask_pred = not self.args.no_mask
        if self.mask_pred:
            self.mask_dim = hidden_dim
            self.controller_layers = 2
            self.dynamic_mask_channels = 16
            self.mask_refine = MSO(mask_dim=self.dynamic_mask_channels, img_dim=[bbn_last_channel, backbone.num_channels[0]], out_dim=self.dynamic_mask_channels)
            self.build_controller()

        # class
        _class_embed = nn.Linear(hidden_dim, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        # kps
        _point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_point_embed.layers[-1].bias.data, 0)

        # box
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        _pose_visible_embed = nn.Linear(hidden_dim, 1, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _pose_visible_embed.bias.data = torch.ones(1) * bias_value

        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_pose_embed_share:
            pose_embed_layerlist = [_point_embed for i in range(transformer.num_decoder_layers)]
            pose_visi_embed_layerlist = [_pose_visible_embed for i in range(transformer.num_decoder_layers)]
        else:
            pose_embed_layerlist = [copy.deepcopy(_point_embed) for i in range(transformer.num_decoder_layers)]
            pose_visi_embed_layerlist = [copy.deepcopy(_pose_visible_embed) for i in range(transformer.num_decoder_layers)]

        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        if self.args.kps_visi_trigger:
            self.pose_visi_embed = nn.ModuleList(pose_visi_embed_layerlist)
        else:
            self.pose_visi_embed = [0 for i in range(len(pose_visi_embed_layerlist))]

        self.transformer.decoder.pose_embed = self.pose_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.num_body_points = num_body_points

        _keypoint_embed = MLP(hidden_dim, 2*hidden_dim, 2*num_body_points, 4)
        nn.init.constant_(_keypoint_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_keypoint_embed.layers[-1].bias.data, 0)

        _enc_pose_visible_embed = nn.Linear(hidden_dim, num_body_points, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _enc_pose_visible_embed.bias.data = torch.ones(num_body_points) * bias_value

        if two_stage_bbox_embed_share:
            self.transformer.enc_pose_embed = _keypoint_embed
        else:
            self.transformer.enc_pose_embed = copy.deepcopy(_keypoint_embed)

        if two_stage_class_embed_share:
            self.transformer.enc_out_class_embed = _class_embed
        else:
            self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def build_controller(self):
        self.controller_layers = self.controller_layers
        self.in_channels = self.mask_dim
        self.dynamic_mask_channels = self.dynamic_mask_channels
        self.mask_out_stride = 4
        self.mask_feat_stride = 8
        self.rel_coord = True
        # compute parameter number
        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = MLP(self.hidden_dim, self.hidden_dim, self.num_gen_params, 3)
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)

    def add_noise_to_points(self, points, query_num=5, max_deviation=0.1):
        batch_size = points.size(0)

        noise_shape = (batch_size, query_num-1, 2)
        noise = (torch.rand(noise_shape, device=points.device) * 2 - 1.0) * max_deviation * 0.4

        points_repeated = points.expand(batch_size, query_num, 2).clone()
        points_repeated[:,1:] += noise
        points_repeated = torch.clamp(points_repeated, min=0., max=1.)
        return points_repeated.detach()

    def forward(self, samples: NestedTensor, targets=None):
        if 'caption' in targets[0]:
            captions = [t['caption'] for t in targets]
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        init_h, init_w = samples.tensors.shape[-2:]
        b = samples.tensors.shape[0]

        # visual encode
        features, poss = self.backbone(samples)
        features_4x, poss_4x = features[0], poss[0]
        features, poss = features[1:], poss[1:]

        # prompt encode
        seed_ = random.random()
        if (seed_ < 1 / 2 and self.training and 'text' in self.args.train_trigger) or \
                (self.training and self.args.train_trigger == 'text') or (not self.training and self.args.eval_trigger == 'text'):
            prompt_type = "text"
            text_features, text_sentence_features = self.forward_text(captions, device=poss[0].device)
            text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
            text_word_features, text_word_masks = text_features.decompose()
            text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]
        else:
            prompt_type = "visual"
            scribbles, points = None, None
            seed_ = random.random()
            if (seed_ < 1 / 2 and self.training and 'scribble' in self.args.train_trigger) or \
                    (self.training and self.args.train_trigger == 'scribble') or (
                    not self.training and self.args.eval_trigger == 'scribble'):
                scribbles = torch.stack([x['scribble'] for x in targets], dim=0)  # b l 2
                text_features, text_sentence_features, text_pos, scribbles_norm = \
                    self.forward_positional_prompt(scribbles, None, features, poss, init_size=(init_w, init_h))
            elif (seed_ >= 1 / 2 and self.training and 'point' in self.args.train_trigger) or \
                    (self.training and self.args.train_trigger == 'point') or (
                    not self.training and self.args.eval_trigger == 'point'):
                # TODO: expand point to cover 3x3 points
                idx_ = 5 if not self.training else random.randint(4, 7)
                points = torch.stack([x['scribble'] for x in targets], dim=0)[:, idx_].unsqueeze(1).float()  # b l 2
                text_features, text_sentence_features, text_pos, points_norm = \
                    self.forward_positional_prompt(points, None, features, poss, init_size=(init_w, init_h))
            else:
                raise NotImplementedError

            text_word_features, text_word_masks = text_features.decompose()
            text_word_features, text_pos = text_word_features.transpose(0, 1), text_pos.transpose(0, 1)

        # multimodal fusion
        srcs = []
        masks = []
        poses = []
        for l, (feat, pos_l) in enumerate(zip(features, poss)):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)
            b, c, h, w = src_proj_l.shape

            src_proj_l = rearrange(src_proj_l, 'b c h w -> h w b c')
            src_proj_l = self.fusion_module(visual=src_proj_l,
                                            text=text_word_features,
                                            text_key_padding_mask=text_word_masks,
                                            text_pos=text_pos,
                                            visual_pos=None,
                                            prompt_type=prompt_type
                                            )
            src_proj_l = rearrange(src_proj_l, '(h w) b c -> b c h w', h=h, w=w)
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                src = rearrange(src, 'b c h w -> h w b c')
                src = self.fusion_module(visual=src,
                                         text=text_word_features,
                                         text_key_padding_mask=text_word_masks,
                                         text_pos=text_pos,
                                         visual_pos=None,
                                         prompt_type=prompt_type
                                         )
                src = rearrange(src, '(h w) b c -> b c h w', h=h, w=w)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        # multimodal encoding & pose-centric hierarchical decoder
        with autocast(enabled=False):
            text_embed = repeat(text_sentence_features, 'b c -> b q c', q=self.num_queries).unsqueeze(-2)
            hs_pose, refpoint_pose, mix_refpoint, mix_embedding, memory = self.transformer(srcs, masks, poses, text_embed)

        # heads
        outputs_class=[]
        outputs_box=[]
        outputs_keypoints_list = []
        outputs_keypoints_visi_list = []
        for dec_lid, (hs_pose_i, refpoint_pose_i_, layer_pose_embed, layer_pose_visi_embed, layer_cls_embed, layer_bbox_embed) \
            in enumerate(zip(hs_pose, refpoint_pose, self.pose_embed, self.pose_visi_embed, self.class_embed, self.bbox_embed)):
            # pose
            bs, nq, np = refpoint_pose_i_.shape
            refpoint_pose_i = refpoint_pose_i_.reshape(bs, nq, np // 2, 2)
            delta_pose_unsig = layer_pose_embed(hs_pose_i[:, :, 1:])
            layer_outputs_pose_unsig = inverse_sigmoid(refpoint_pose_i[:, :, 1:]) + delta_pose_unsig
            vis_flag = torch.ones_like(layer_outputs_pose_unsig[..., -1:], device=layer_outputs_pose_unsig.device)
            layer_outputs_pose_unsig = torch.cat([layer_outputs_pose_unsig, vis_flag], dim=-1).flatten(-2)
            layer_outputs_pose_unsig = layer_outputs_pose_unsig.sigmoid()
            outputs_keypoints_list.append(keypoint_xyzxyz_to_xyxyzz(layer_outputs_pose_unsig))
            if self.args.kps_visi_trigger:
                outputs_keypoints_visi_list.append(layer_pose_visi_embed(hs_pose_i[:, :, 1:]).squeeze(-1))  # b q k
            else:
                outputs_keypoints_visi_list.append(0)
            # cls
            layer_cls = layer_cls_embed(hs_pose_i[:, :, 0])
            outputs_class.append(layer_cls)
            # box
            layer_box = layer_bbox_embed(hs_pose_i[:, :, 0])
            layer_box[..., :2] += inverse_sigmoid(refpoint_pose_i[:, :, 0])
            outputs_box.append(layer_box.sigmoid())

        out = {'pred_logits': outputs_class[-1], 'pred_keypoints': outputs_keypoints_list[-1],
               'pred_keypoints_visi': outputs_keypoints_visi_list[-1], 'pred_boxes': outputs_box[-1]}

        if self.mask_pred:
            tar_h, tar_w = memory[0].shape[-2:]
            mask_features = sum([F.interpolate(x, size=(tar_h, tar_w), mode="bicubic", align_corners=False) for x in memory])  # b c h w
            bs, nq, np = refpoint_pose[-2].shape
            refpoint_pose_i = refpoint_pose[-2].reshape(bs, nq, np // 2, 2)
            hs_pose_i = hs_pose[-1]
            pred_masks = []
            for query_idx in range(self.num_queries):
                refpoint_pose_i_ = refpoint_pose_i[:, query_idx].unsqueeze(1)
                hs_pose_i_ = hs_pose_i[:, query_idx].unsqueeze(1)
                dynamic_mask_head_params = self.controller(hs_pose_i_[:, :, 0])
                lvl_references = refpoint_pose_i_[:, :, 0][..., :2]
                outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references, targets)
                outputs_seg_mask_high = self.mask_refine(outputs_seg_mask.squeeze(1), [features_4x, features[0]])
                outputs_seg_mask_high = rearrange(F.pixel_shuffle(outputs_seg_mask_high, 4).squeeze(1), '(b q) h w -> b q h w', b=b, q=1)
                pred_masks.append(outputs_seg_mask_high)
            pred_masks = torch.cat(pred_masks, dim=1)
        else:
            pred_masks = torch.ones((b, self.num_queries, init_h, init_w), device=out['pred_logits'].device).float()
        out['pred_masks'] = pred_masks
        out['padded_hw'] = (init_h, init_w)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_keypoints, outputs_keypoints_visi, outputs_box):
        return [{'pred_logits': a, 'pred_keypoints': b, 'pred_keypoints_visi': c, "pred_boxes": d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_keypoints[:-1], outputs_keypoints_visi[:-1], outputs_box[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            text_features, text_sentence_features, text_pad_mask = self.text_encoder(captions, device)
            text_features = self.text_proj(text_features)
            text_sentence_features = self.sentence_proj(text_sentence_features)
            text_features = NestedTensor(text_features, text_pad_mask)
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features, text_sentence_features

    def forward_positional_prompt(self, scribbles, bboxes, feats, feats_pos, init_size):
        if scribbles is not None:
            # norm
            init_w, init_h = init_size
            scribbles = scribbles.float()
            scribbles[..., 0] /= float(init_w)
            scribbles[..., 1] /= float(init_h)
            scribbles = torch.clamp(scribbles, min=0, max=1)

            if scribbles.shape[1] == 1:
                with torch.no_grad():
                    scribbles_norm = self.add_noise_to_points(scribbles, self.num_queries)
            else:
                scribbles_norm = scribbles

            scribble_features, scribble_pos = [], []
            feats, feats_pos = [feats[-2]], [feats_pos[-2]]
            for feat, feat_pos in zip(feats, feats_pos):
                feat_, feat_pos_ = feat.tensors, feat_pos
                scribbles_ = scribbles.clone().unsqueeze(1) * 2 - 1
                samp_feat = torch.nn.functional.grid_sample(feat_, scribbles_, align_corners=True).squeeze(-2).transpose(-1,-2)
                samp_pos = torch.nn.functional.grid_sample(feat_pos_, scribbles_, align_corners=True).squeeze(-2).transpose(-1,-2)

                scribble_features.append(samp_feat)
                scribble_pos.append(samp_pos)

            scribble_features = torch.cat(scribble_features, dim=-1)
            scribble_pos = torch.cat(scribble_pos, dim=-1)

            scribble_features = self.prompt_proj(scribble_features)
            scribble_pad_mask = torch.zeros(scribble_features.shape[:2]).bool().to(scribble_pos.device)
            scribble_features = NestedTensor(scribble_features, scribble_pad_mask)

            scribble_sentence_features = torch.mean(scribble_features.tensors, dim=1)
            scribble_sentence_features = self.glob_prompt_proj(scribble_sentence_features)
        else:
            raise NotImplementedError


        return scribble_features, scribble_sentence_features, scribble_pos, scribbles_norm

    def forward_positional_prompt_bbox(self, bboxes, feats, feats_pos, init_size):
        bbox_features = crop_object_using_bbox(feats[-2].tensors, bboxes).flatten(-2).transpose(-1, -2)  # b l c
        bbox_pos = crop_object_using_bbox(feats_pos[-2], bboxes).flatten(-2).transpose(-1, -2)  # b l c
        bbox_features = self.prompt_proj(bbox_features)
        bbox_pad_mask = torch.zeros(bbox_features.shape[:2]).bool().to(bbox_pos.device)
        bbox_features = NestedTensor(bbox_features, bbox_pad_mask)  # b l c

        bbox_sentence_features = torch.mean(bbox_features.tensors, dim=1)
        bbox_sentence_features = self.glob_prompt_proj(bbox_sentence_features)
        return bbox_features, bbox_sentence_features, bbox_pos, None

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        mask_features = mask_features.unsqueeze(1)
        b, t, c, h, w = mask_features.shape
        _, num_queries = reference_points.shape[:2]
        q = num_queries // t

        # prepare reference points in image size (the size is input size to the model)
        # use xyxy rather than xy in xywh
        new_reference_points = []
        for i in range(b):
            img_h, img_w = targets[i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0)
            tmp_reference_points = reference_points[i] * scale_f[None, :]
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0)
        reference_points = new_reference_points

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q)
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride)
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - locations.reshape(1, 1, 1, h, w, 2)
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3, 4)

            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q)
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q)
        mask_features = mask_features.reshape(1, -1, h, w)

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic conditional segmentation
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0])
        mask_logits = rearrange(mask_logits, 'n (b t q c) h w -> (n b) (t q) c h w', t=t, q=q, c=16)
        return mask_logits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
        bias_splits[l] = bias_splits[l].reshape(num_insts * channels)

    return weight_splits, bias_splits

def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

@MODULE_BUILD_FUNCS.registe_with_name(module_name='uniphd')
def build_uniphd(args):
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    matcher = None

    model = UniPHD(
        args,
        matcher,
        backbone,
        transformer,
        aux_loss=args.aux_loss,
        num_classes=num_classes,
        nheads=args.nheads,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        dec_pred_class_embed_share=args.dec_pred_class_embed_share,
        dec_pred_pose_embed_share=args.dec_pred_pose_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        cls_no_bias=args.cls_no_bias,
        num_body_points=args.num_body_points
    )

    # prepare weight dict
    weight_dict = {
        'loss_ce': args.cls_loss_coef, 
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        "loss_keypoints":args.keypoints_loss_coef,
        "loss_keypoints_visi": args.keypoints_visi_loss_coef,
        "loss_oks":args.oks_loss_coef,
        'loss_mask': args.mask_loss_coef,
        'loss_dice': args.dice_loss_coef,
        'loss_mask_low': args.mask_loss_coef,
        'loss_dice_low': args.dice_loss_coef
    }
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            for k, v in clean_weight_dict.items():
                aux_weight_dict.update({k + f'_{i}': v})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        no_interm_loss = args.no_interm_loss
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_loss else 0.0,
            'loss_keypoints': 1.0 if not no_interm_loss else 0.0,
            'loss_keypoints_visi': 1.0 if not no_interm_loss else 0.0,
            'loss_oks': 1.0 if not no_interm_loss else 0.0,
            'loss_mask': 1.0 if not no_interm_loss else 0.0,
            'loss_dice': 1.0 if not no_interm_loss else 0.0,
            'loss_mask_low': 1.0 if not no_interm_loss else 0.0,
            'loss_dice_low': 1.0 if not no_interm_loss else 0.0,
        }
        interm_weight_dict.update({k + f'_interm': v * args.interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict.items()})
        weight_dict.update(interm_weight_dict)
    
    losses = ['labels', 'boxes', "keypoints", "matching"]
    if not args.no_mask:
        losses += ["masks"]
    criterion = SetCriterion(args, num_classes, matcher=matcher, weight_dict=weight_dict, focal_alpha=args.focal_alpha, losses=losses, num_body_points=args.num_body_points)
    criterion.to(device)
    postprocessors = {
        'bbox': PostProcess(num_body_points=args.num_body_points),  # only use this to process results.
        'segm': PostProcessSegm(),
        'keypoints': PostProcessPose(),
    }

    return model, criterion, postprocessors