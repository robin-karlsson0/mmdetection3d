# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet3d.core import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.builder import HEADS
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmdet3d.models.utils import clip_sigmoid


@HEADS.register_module()
class CenterHeadWeakSupRP(CenterHead):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CenterHeadWeakSupRP,
              self).__init__(in_channels, tasks, train_cfg, test_cfg,
                             bbox_coder, common_heads, loss_cls, loss_bbox,
                             separate_head, share_conv_channel,
                             num_heatmap_convs, conv_cfg, norm_cfg, bias,
                             norm_bbox, init_cfg)

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Weakly supervised labels == Point labels

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        # Generate 'bboxes' by stacking center location (x, y, z) and
        # (w, l, h, rot)
        #     gravity_center --> torch.Tensor dim (N, 3)
        #     gt_bboxes_3d --> torch.Tensor dim (N, 7)
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)

        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(
            self.train_cfg['grid_size'])  # (1408, 1600, 40)
        pc_range = torch.tensor(
            self.train_cfg['point_cloud_range'])  # (0, -40, -3, 70, 40, 1)
        voxel_size = torch.tensor(
            self.train_cfg['voxel_size'])  # (0.1, 0.1, 0.2)
        gt_annotation_num = len(self.train_cfg['code_weights'])  # 8

        # Generate 'feature_map' dimensions from 'grid_size' scaled by a factor
        # - Why not just decrease grid size?
        # self.train_cfg['out_size_factor'] --> 8 (i.e. 8 voxels / feature)
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []  # List of bbox_3d torch.Tensor dim (7)
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        # One iteration for each head (2)
        for idx, task_head in enumerate(self.task_heads):
            # Initialize empty array (1, H, W)
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))
            # Empty row-vector matrix to add bbox_3d annotations
            anno_box = gt_bboxes_3d.new_zeros((max_objs, gt_annotation_num),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            # Process each object 'k' one-by-one
            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                # Convert [m] --> [feats] (i.e. elements in feature map)
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]  # gravity_center
                    # Convert 'lidar coords' --> 'feature map coords'
                    # (x = -40 --> 0)
                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']
                    # NOTE Object center in feature map coordinates (i, j)
                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    # NOTE HEATMAP: Draw Gaussians for objects in
                    # class-specific map
                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    # Matrix element order idx
                    ind[new_idx] = y * feature_map_size[
                        0] + x  # [k] --> Object location (i, j) in feature map
                    mask[new_idx] = 1  # Record that 'object w. idx' exists

                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    # Normalize dimension of bbox
                    if self.norm_bbox:
                        box_dim = box_dim.log()

                    # NOTE BBOX: List of torch.Tensor [
                    #     Refinement (dx, dy) within feature map element (i,j),
                    #     z coord,
                    #     (w,l,h),
                    #     sin(rot),
                    #     cos(rot)]
                    # NOTE Feature map position (i,j) for object 'k' stored as
                    # matrix element index?
                    anno_elems = [
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0)
                    ]
                    # Assumes datasets with bbox annotations with 9
                    # values have two additional velocity components (vx, vy)
                    # in addition to the standard KITTI-like 7 values
                    # (H, W, L, x, y, z, rot).
                    # NOTE: Rotation is split into two (sin, cos) components,
                    # hence incrementing the annotation number by one.
                    if gt_annotation_num == 10:
                        vx, vy = task_boxes[idx][k][7:10]
                        anno_elems += [vx.unsqueeze(0), vy.unsqueeze(0)]

                    # Add row-vector to annotation matrix
                    anno_box[new_idx] = torch.cat(anno_elems)  # dim (8)

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for weakly supervised CenterHead region proposal.

        Applies loss for
        1. Object center prediction heatmaps
        2. Object center refinements (dx, dy) within feature map element (i, j)

        Structure of 'preds_dicts':
            List of dicts with index corresponding to 'task'
                pred_dicts[0] --> car_task
                pred_dicts[1] --> pedestrian_task
                ...
            Each task has a dict inside a single element list
                car_task[0] --> car_dict
            Each task dict consists of torch.Tensors w. dim (B, feat, H, W)
                car_dict['reg'] --> torch.Tensor
                car_dict['heatmap']

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        # NOTE 'anno_boxes' are matrices w. row-vectors as bbox_3d annotations
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()

            # Original heatmap loss
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))

            # 'Soft focal loss' heatmap loss
            # preds_dict[0]['heatmap'] --> torch.Tensor (2, 1, H, W)
            # heatmaps[task_id] --> torch.Tensor (2, 1, H, W)
            # y = preds_dict[0]['heatmap']
            # gt = heatmaps[task_id]
            # y_hat = y * gt + (1 - y) * (1 - gt)
            # ALPHA = 0.25
            # GAMMA = 2.
            # loss_heatmap = ALPHA * torch.pow(1. - y_hat,
            #                                  GAMMA) * torch.log(y_hat)
            # loss_heatmap = torch.mean(loss_heatmap, dim=(1, 2, 3))
            # loss_heatmap = torch.mean(loss_heatmap, dim=0)

            target_box = anno_boxes[task_id]
            # Reconstruct the anno_box from multiple reg heads
            # Default keys assumed to exist for annotations with standard
            # KITTI-like 7 values
            anno_box = [preds_dict[0]['reg']]
            preds_dict[0]['anno_box'] = torch.cat(anno_box, dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)

            # TODO Temporary WS annotation hack before modifying actual targets
            target_box = target_box[:, :, :2]
            bbox_weights = bbox_weights[:, :, :2]

            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
        return loss_dict

    def get_rps(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate region proposals from head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []  # One 'rets' entry per class
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']

            N = batch_heatmap.shape[0]  # TODO find meaning
            H, W = batch_heatmap.shape[-2:]
            RP_SIZE = 4.  # [m]
            dummy_rots = torch.zeros(N, 1, H, W)
            dummy_rotc = torch.zeros(N, 1, H, W)
            dummy_hei = 0.5 * RP_SIZE * torch.ones(N, 1, H, W)
            dummy_dim = RP_SIZE * torch.ones(N, 3, H, W)
            dummy_vel = None

            # temp: Single element list with [dict]
            # Keys:
            #   'bboxes' --> torch.Tensor dim (#obj N, 7)
            #   'scores' --> torch.Tensor dim (#obj N)
            #   'labels' --> torch.Tensor dim (#obj N)
            #
            # Function decode() requires all inputs
            # ==> Replace with dummy tensors
            temp = self.bbox_coder.decode(
                batch_heatmap,  # (1,1,200,176)
                dummy_rots,  # (1,1,200,176)
                dummy_rotc,  # (1,1,200,176)
                dummy_hei,  # (1,1,200,176)
                dummy_dim,  # (1,3,200,176)
                dummy_vel,  # (1,2,200,176)
                reg=batch_reg,  # (1,2,200,176)
                task_id=task_id)

            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])  # TODO find meaning

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    # Concatenate bbox row-vectors of all classes (N, 7)
                    bboxes = torch.cat([ret[i]['bboxes'] for ret in rets])
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])

        return ret_list
