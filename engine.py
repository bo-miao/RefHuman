import math
import sys
from typing import Iterable
from util.utils import to_device
import torch
import util.misc as utils
import os
from datasets.coco_eval import CocoEvaluator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.utils import calculate_precision_at_k_and_iou_metrics, calculate_bbox_precision_at_k_and_iou_metrics
import cv2
import numpy as np
from PIL import Image
from einops import rearrange, repeat

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k not in ["caption", "dataset_name", "original_id", 'img_name']} for t in targets]
        for i in range(len(targets)):  # add caption
            targets[i]['caption'] = captions[i]

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if args.amp:
            # amp backward function (not used)
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def calculate_pckh(predictions, ground_truths, threshold=0.5, kps_num=17):
    B, _, _ = predictions.shape
    correct_keypoints = 0
    total_keypoints = 0

    for i in range(B):
        pred = predictions[i]
        gt = ground_truths[i]

        if gt[3, 2] < 1 or gt[4, 2] < 1:
            continue

        head_length = np.linalg.norm(gt[3, :2] - gt[4, :2])
        for j in range(kps_num):
            if gt[j, 2] > 0:
                total_keypoints += 1
                distance = np.linalg.norm(pred[j, :2] - gt[j, :2])
                if distance <= threshold * head_length:
                    correct_keypoints += 1

    return correct_keypoints, total_keypoints

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    try:
        useCats = args.useCats
    except:
        useCats = True

    iou_types = tuple(postprocessors.keys())
    evaluator_list = []
    if args.dataset_file == "refhuman":
        from datasets.coco_eval import CocoEvaluator
        evaluator_list.append(CocoEvaluator(base_ds, iou_types, useCats=useCats, coco_path=args.coco_path))

    predictions = []
    PCKH_predictions = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        img_names = [t['img_name'] for t in targets]
        captions = [t["caption"] for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k not in ["caption", "dataset_name", "original_id", 'img_name', 'img_obj_num']} for t in targets]
        for i in range(len(targets)):
            targets[i]['caption'] = captions[i]

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, targets)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            if 'bbox' in postprocessors.keys():  # this one processes all predictions
                results, best_kps_pred = postprocessors['bbox'](outputs, orig_target_sizes, target_sizes, targets)
            if 'segm' in postprocessors.keys():
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            if 'keypoints' in postprocessors.keys():
                results = postprocessors['keypoints'](results, outputs, orig_target_sizes, target_sizes)

        # ********* PCKh@0.5 using top-1 query *******
        best_kps_pred = best_kps_pred.cpu().numpy()
        gt_kps = torch.cat([t['origin_keypoints'] for t in targets], dim=0).flatten(-2).cpu().numpy()
        gt_kps[:, 2::3] = (gt_kps[:, 2::3] > 0)
        best_kps_pred, gt_kps = rearrange(best_kps_pred, 'b (k c) -> b k c', c=3), rearrange(gt_kps, 'b (k c) -> b k c', c=3)
        batch_pckh_correct, batch_pckh_total = calculate_pckh(best_kps_pred, gt_kps)
        PCKH_predictions.append({"PCKH_CORRECT": batch_pckh_correct, "PCKH_TOTAL": batch_pckh_total})

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for evaluator in evaluator_list:
            evaluator.update(res)
        for p, target in zip(results, targets):
            for s, b, m in zip(p['scores'], p['boxes'], p['rle_masks']):
                predictions.append({'image_id': target['image_id'].item(),
                                    'category_id': 1,
                                    'bbox': b.tolist(),
                                    'segmentation': m,
                                    'score': s.item()})

    # accumulate predictions
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()
    for evaluator in evaluator_list:
        evaluator.accumulate()
        evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()
            if "keypoints" in postprocessors.keys():
                stats['coco_eval_keypoints'] = evaluator.coco_eval['keypoints'].stats.tolist()

    # Compute final PCKh@0.5 score
    PCKH_predictions = utils.all_gather(PCKH_predictions)
    PCKH_predictions = [p for p_list in PCKH_predictions for p in p_list]
    PCKH_CORRECT, PCKH_TOTAL = sum([x['PCKH_CORRECT'] for x in PCKH_predictions]), sum(x['PCKH_TOTAL'] for x in PCKH_predictions)
    print("\n\n ************** Pose PCKh@0.5 score is {}. **************\n\n".format(PCKH_CORRECT/PCKH_TOTAL))

    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]
    predictions = sorted(predictions, key=lambda x: x['image_id'])
    eval_metrics = {}
    if utils.is_main_process():
        coco_gt = COCO(os.path.join(args.coco_path, 'RefHuman_val.json'))
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        print(f'\n\n ********* Segm overall/mean IoU is {overall_iou}/{mean_iou}. ********* \n\n')
        stats.update(eval_metrics)

    return stats
