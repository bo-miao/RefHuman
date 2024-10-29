from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import random

import datasets.transforms as T

__all__ = ['build']

def project_kps_on_image(kps, img, radius=4):
    pose_palette = [[255, 128, 0], [255, 153, 51], [255, 178, 102],
                             [230, 230, 0], [255, 153, 255], [153, 204, 255],
                             [255, 102, 255], [255, 51, 255], [102, 178, 255],
                             [51, 153, 255], [255, 153, 153], [255, 102, 102],
                             [255, 51, 51], [153, 255, 153], [102, 255, 102],
                             [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                             [255, 255, 255], [255, 128, 0], [255, 153, 51], [255, 178, 102],
                             [230, 230, 0], [255, 153, 255], [153, 204, 255]]
    num = len(kps)
    for i in range(num-1):
        x_coord, y_coord = kps[i]
        cv2.circle(img, (int(x_coord), int(y_coord)), radius, pose_palette[i], -1)
    return img

class RefHuman(torch.utils.data.Dataset):
    def __init__(self, root_path, image_set, transforms, return_masks):
        super(RefHuman, self).__init__()
        return_masks = True
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, image_set)
        if image_set == "train":
            self.mode = 'train'
            self.img_folder = root_path / "images"
            self.coco = COCO(root_path / "RefHuman_train.json")
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            for image_id in imgIds:
                if self.coco.getAnnIds(imgIds=image_id) == []:
                    continue
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                target = self.coco.loadAnns(ann_ids)
                num_keypoints = [obj["num_keypoints"] for obj in target]
                if sum(num_keypoints) == 0:
                    continue
                self.all_imgIds.append(image_id)
            print("****** Total train img number is {}. ******".format(len(self.all_imgIds)))
        else:
            self.mode = 'val'
            self.img_folder = root_path / "images"
            eval_folder = root_path / "RefHuman_val.json"
            self.coco = COCO(eval_folder)
            imgIds = sorted(self.coco.getImgIds())
            self.all_imgIds = []
            for image_id in imgIds:
                self.all_imgIds.append(image_id)
            print("****** Total eval img number is {}. ******".format(len(self.all_imgIds)))

    def __len__(self):
        return len(self.all_imgIds)

    def __getitem__(self, idx):
        flag = False
        while not flag:
            image_id = self.all_imgIds[idx]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            target = self.coco.loadAnns(ann_ids)
            coco_img = self.coco.loadImgs(image_id)[0]
            coco_img_name = coco_img["file_name"]
            caption = coco_img["caption"] if 'caption' in coco_img else 'ssssss'

            target = {'image_id': image_id, 'annotations': target, "caption": caption, "img_name": coco_img_name}
            img = Image.open(self.img_folder / self.coco.loadImgs(image_id)[0]['file_name'])
            img, target = self.prepare(img, target)
            target_ = target.copy()
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            if self.mode == 'val':
                target['img_name'] = coco_img_name
                target['origin_mask'] = target_['masks']
                target['origin_keypoints'] = target_['keypoints']
                target['origin_boxes'] = target_['boxes']
                target['origin_area'] = target_['area']
                target['origin_scribble'] = target_['scribble']
                target['img_obj_num'] = self.obj_num_counter[coco_img_name]

            if self.mode == 'train' and len(target['boxes']) == 0:
                idx = random.randint(0, self.__len__() - 1)
            else:
                flag = True

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, mode="train"):
        self.return_masks = return_masks
        self.mode = mode

    def __call__(self, image, target):
        w, h = image.size

        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        anno = [obj for obj in anno if obj['num_keypoints'] != 0]
        keypoints = [obj["keypoints"] for obj in anno]
        boxes = [obj["bbox"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, 17, 3)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)  # now is xmin, ymin, xmax, ymax
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        keypoints = keypoints[keep]
        if self.return_masks:
            masks = masks[keep]

        caption = target["caption"] if 'caption' in target else None
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # load caption
        if caption is not None:
            target["caption"] = caption

        # load scribble
        if 'scribble' in anno[0]:
            idx = random.randint(0, 4) if self.mode == 'train' else 0
            target['scribble'] = torch.stack([torch.from_numpy(np.array(obj["scribble"]).reshape((5, 12, 2)))[idx] for obj in anno], dim=0)
            for scrib, msk in zip(target['scribble'], target['masks']):
                assert T.are_points_within_mask(msk.int().numpy(), scrib), 'error scribble {}.'.format(image_id)
        return image, target


def make_coco_transforms(image_set, args=None):
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    normalize_eval = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], eval_mode=True)])

    # config the params for data aug
    scales = [360, 392, 416, 480, 512, 544, 576, 608, 640]
    max_size = 640
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])
    elif image_set in ['val', 'test']:
        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize_eval,
        ])
    else:
        raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    dataset = RefHuman(root, image_set, transforms=make_coco_transforms(image_set), return_masks=True)
    return dataset

