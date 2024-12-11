import torch
from torch import nn
from mmengine.model import BaseModel
from datasets.metrics import estimateOverlap, estimateAccuracy
import numpy as np
from datasets import points_utils
from nuscenes.utils import geometry_utils
from mmengine.registry import MODELS
from models.backbone.pointnet import MiniPointNet, SegPointNet, FeaturePointNet
from models.head.Seq2SeqFormer import Seq2SeqFormer
from torchmetrics import Accuracy
from datasets.misc_utils import get_tensor_corners_batch
from datasets.misc_utils import create_corner_timestamps
from datasets.misc_utils import get_history_frame_ids_and_masks,get_last_n_bounding_boxes
from datasets.misc_utils import generate_timestamp_prev_list
import open3d as o3d

@MODELS.register_module()
class seqtrack3d(BaseModel):

    def __init__(self,
                 backbone=None,
                 fuser=None,
                 head=None,
                 cfg=None):
        super().__init__()
        self.config = cfg
        self.backbone = MODELS.build(backbone)
        self.backbone_frame = MODELS.build(backbone)
        self.fuse = MODELS.build(fuser)
        # self.head = MODELS.build(head)
        self.hist_num = getattr(cfg, 'hist_num', 1)
        self.seg_acc = Accuracy(task='multiclass', num_classes=2, average='none')

        self.box_aware = getattr(cfg, 'box_aware', False)
        # self.use_motion_cls = getattr(cfg, 'use_motion_cls', True)
        # self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
        #                                 per_point_mlp1=[64, 64, 64, 128, 1024],
        #                                 per_point_mlp2=[512, 256, 128, 128],
        #                                 output_size=2 + (9 if self.box_aware else 0))
        # self.mini_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
        #                                   per_point_mlp=[64, 128, 256, 512],
        #                                   hidden_mlp=[512, 256],
        #                                   output_size=-1)
        # if self.use_motion_cls:
        #     self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
        #                                           nn.BatchNorm1d(128),
        #                                           nn.ReLU(),
        #                                           nn.Linear(128, 128),
        #                                           nn.BatchNorm1d(128),
        #                                           nn.ReLU(),
        #                                           nn.Linear(128, 2))
        #     self.motion_acc = Accuracy(task='multiclass', num_classes=2, average='none')

        self.motion_mlp = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 7))

        # self.feature_pointnet = FeaturePointNet(
        #     input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
        #     per_point_mlp1=[64, 64, 64, 128, 1024],
        #     per_point_mlp2=[512, 256, 128, 128],
        #     output_size=128)
        self.pos_emd = PositionEmbeddingLearned(2, 128)
        self.Transformer = MODELS.build(head)

    def forward(self,
                inputs,
                data_samples=None,
                mode: str = 'predict',
                **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def get_feats(self, input_dict):
        output_dict = {}
        stack_points = [torch.cat((points, bc), dim=-1) for points, bc in zip(input_dict["stack_points"], input_dict["candidate_bc"])]
        stack_feats = self.backbone(stack_points, len(stack_points))
        point_feature = self.fuse(stack_feats)
        B = point_feature.size(0)
        HL = input_dict["valid_mask"][0].shape[0]  # Number of historical frames, default 3
        L = HL + 1  # Total length of the point cloud sequence, 1 represents the current frame
        # chunk_size = N // L
        num = torch.stack(input_dict['points_num'])
        solo_points = []
        for i in range(B):
            a = list(torch.split(stack_points[i], num[i].tolist(), dim=0))
            solo_points.extend(a)
        # solo_points = torch.stack(solo_points)
        # seg_out = self.seg_pointnet(x)
        # seg_logits = seg_out[:, :2, :]  # B,2,N
        # pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        # mask_points = x[:, :4, :] * pred_cls
        #
        # if self.box_aware:
        #     pred_bc = seg_out[:, 2:, :]
        #     mask_pred_bc = pred_bc * pred_cls
        #     mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
        #     output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        # Coarse initial motion prediction
        # point_feature = self.mini_pointnet(mask_points)  # N*256
        motion_pred = self.motion_mlp(point_feature)  # B,4

        motion_pred_masked = motion_pred

        prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)

        # Get corners of the current and historical boxes
        bbox_size = torch.stack(input_dict["bbox_size"])
        bbox_size_repeated = bbox_size.repeat_interleave(L, dim=0)

        ref_boxs = torch.stack(input_dict["ref_boxs"])
        box_seq = torch.cat((ref_boxs, aux_box.unsqueeze(1)), dim=1)
        box_seq = box_seq.reshape(B * L, 4)
        box_seq_corner = get_tensor_corners_batch(box_seq[:, :3], bbox_size_repeated, box_seq[:, -1])
        box_seq_corners = box_seq_corner.reshape(B, L * 8,
                                                 -1)  # B*(L*8)*3 represents a total of L*8 points, each with 3 features

        # Appending timestamp features to the box corners
        corner_stamps = create_corner_timestamps(B, HL, 8).to('cuda')
        box_seq_corners = torch.cat((box_seq_corners, corner_stamps),
                                    dim=-1)  # B*(L*8)*4 where 4 represents features for x, y, z, and timestamp

        # solo_x = x.reshape(B * L, -1, chunk_size)  # Reshape into separate point clouds
        feature = self.backbone_frame(solo_points, len(solo_points))  # (B*num) * C * N Note: N is the number of points per frame

        x_grid, y_grid = torch.meshgrid(torch.arange(16).cuda(), torch.arange(
            16).cuda())  # (B*num) * C * N Note: N is the number of points per frame
        indices = torch.stack((x_grid, y_grid), dim=1).transpose(0, 1).unsqueeze(0).expand(feature.shape[0], 2, 16, 16)
        pos = self.pos_emd(indices)
        feature = feature + pos

        NEW_N = feature.shape[2]*feature.shape[3]
        # feature = feature

        points_feature = feature.reshape(B,-1,NEW_N*L).transpose(1, 2)

        delta_motion = self.Transformer(box_seq_corners, points_feature, input_dict["valid_mask"])  # B*4*4

        updated_ref_boxs = delta_motion[:, :HL, :]
        updated_aux_box = delta_motion[:, -1, :]

        output_dict["estimation_boxes"] = torch.cat((aux_box, motion_pred[:, 4:]),dim=-1)
        output_dict.update({"motion_pred": motion_pred,
                            'aux_estimation_boxes': updated_aux_box,
                            'ref_boxs': input_dict['ref_boxs'],
                            'valid_mask': input_dict["valid_mask"],
                            'updated_ref_boxs': updated_ref_boxs,
                            })

        return output_dict

    def inference(self, inputs, ref_box):
        results = self.get_feats(inputs)
        estimation_box = results['aux_estimation_boxes']
        estimation_box_cpu = estimation_box.squeeze(0).detach().cpu().numpy()

        valid_mask = results['valid_mask'][0].detach().cpu().numpy()
        estimation_box_cpu[4] = 0
        if len(estimation_box.shape) == 3:
            best_box_idx = estimation_box_cpu[:, 4].argmax()
            estimation_box_cpu = estimation_box_cpu[best_box_idx, 0:4]

        candidate_box = points_utils.getOffsetBB(ref_box, estimation_box_cpu, degrees=False,
                                                 use_z=True, limit_box=False)

        return candidate_box, valid_mask

    def loss(self, inputs, data_samples):
        results = self.get_feats(inputs)
        losses = dict()
        losses.update(self.Transformer.loss(data_samples,results))

        return losses

    def predict(self, inputs):
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(inputs)):  # tracklet
            this_bb = inputs[frame_id]["3d_bbox"]

            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
                last_coors = np.array([0., 0.])
            else:
                data_dict, ref_bb = self.build_input_dict(inputs, frame_id, results_bbs)
                if torch.sum(data_dict['stack_points'][:,:3]) == 0:
                    print("Empty pointcloud!")


                data_dict['stack_points'] = [data_dict['stack_points']]
                candidate_box,*_ = self.inference(data_dict, ref_box=ref_bb)
                results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            ious.append(this_overlap)
            distances.append(this_accuracy)

        return ious, distances

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(frame_id, self.hist_num)
        prev_frames = [sequence[id] for id in prev_frame_ids]
        this_frame = sequence[frame_id]
        this_pc = this_frame['pc']
        bbox_size = this_frame['3d_bbox'].wlh
        prev_pcs = [frame['pc'] for frame in prev_frames]
        ref_boxs = get_last_n_bounding_boxes(results_bbs, valid_mask)
        num_hist = len(valid_mask)

        prev_frame_pcs = []
        # pcd = o3d.geometry.PointCloud()

        for i, prev_pc in enumerate(prev_pcs):
            prev_frame_pc = points_utils.crop_pc_in_range(prev_pc, ref_boxs[0], self.config.point_cloud_range)
            # prev_frame_pc = points_utils.generate_subwindow_with_aroundboxs(prev_pc, ref_boxs[i], ref_boxs[0],
            #                                                                 scale=self.config.bb_scale,
            #                                                                 offset=self.config.bb_offset)
            # pcd.points = o3d.utility.Vector3dVector(prev_frame_pc.points.T[:,:3])
            prev_frame_pcs.append(prev_frame_pc)

        this_frame_pc = points_utils.crop_pc_in_range(this_pc, ref_boxs[0], self.config.point_cloud_range)
        # this_frame_pc = points_utils.generate_subwindow_with_aroundboxs(this_pc, ref_boxs[0], ref_boxs[0],
        #                                                                 scale=self.config.bb_scale,
        #                                                                 offset=self.config.bb_offset)

        # canonical_box = points_utils.transform_box(ref_boxs[0], ref_boxs[0])
        ref_boxs = [
            points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs
        ]

        prev_points_list = [
            prev_frame_pc.points.T if prev_frame_pc.points.shape[1] >= 1 else np.zeros((1, 4), dtype='float32') for
            prev_frame_pc in prev_frame_pcs]
        this_points = this_frame_pc.points.T
        if this_points.shape[0] < 1:
            this_points = np.zeros((1, 4), dtype='float32')

        timestamp_prev_list = generate_timestamp_prev_list(valid_mask, [t.shape[0] for t in prev_points_list])
        timestamp_this = np.full((this_points.shape[0], 1), fill_value=0.1)

        prev_points_list = [
            np.concatenate([prev_points, timestamp_prev],
                           axis=-1)
            for prev_points, timestamp_prev in zip(
                prev_points_list, timestamp_prev_list)
        ]

        this_points = np.concatenate([this_points, timestamp_this], axis=-1)

        stack_points_list = prev_points_list + [this_points]
        stack_points = np.concatenate(stack_points_list, axis=0)

        ref_box_thetas = [
            ref_box.orientation.degrees * ref_box.orientation.axis[-1]
            if self.config.degrees else ref_box.orientation.radians *
            ref_box.orientation.axis[-1] for ref_box in ref_boxs
        ]
        ref_box_list = [
            np.append(ref_box.center,
                      theta).astype('float32') for ref_box, theta in zip(
                ref_boxs, ref_box_thetas)
        ]
        ref_boxs_np = np.stack(ref_box_list, axis=0)

        data_dict = {"stack_points": torch.tensor(stack_points, dtype=torch.float32).cuda(),
                     "ref_boxs": [torch.tensor(ref_boxs_np, dtype=torch.float32).cuda()],
                     "valid_mask": [torch.tensor(valid_mask, dtype=torch.float32).cuda()],
                     "bbox_size": [torch.tensor(bbox_size, dtype=torch.float32).cuda()],
                     'points_num': [torch.tensor([x.shape[0] for x in stack_points_list]).cuda()],
                     }

        if getattr(self.config, 'box_aware', False):
            hist_points_list = stack_points_list[:num_hist]
            candidate_bc_prev_list = [
                points_utils.get_point_to_box_distance(hist_points[:, :3], ref_box)
                for hist_points, ref_box in zip(hist_points_list, ref_boxs)
            ]
            candidate_bc_this = np.zeros([this_points.shape[0],9])
            candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
            candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)

            data_dict.update({'candidate_bc': [torch.as_tensor(candidate_bc.astype('float32'), dtype=torch.float32).cuda()]})
        return data_dict, results_bbs[-1]

class Open3D_visualizer():

    def __init__(self, points, gt_bboxes, pred_bboxes) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.points = self.points2o3d(points)
        self.gt_boxes = self.box2o3d(gt_bboxes, 'red') if gt_bboxes is not None else None
        self.pred_boxes = self.box2o3d(pred_bboxes, 'green') if pred_bboxes is not None else None

    def points2o3d(self, points):
        """
        points: np.array, shape(N, 3)
        """
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points.T[:, :3])
        # pointcloud.colors = o3d.utility.Vector3dVector(
        #     [[255, 255, 255] for _ in range(len(points))])
        return pointcloud

    def box2o3d(self, bbox, color):
        """
        bboxes: np.array, shape(N, 7)
        color: 'red' or 'green'
        """

        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                      [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        if color == 'red':
            colors = [[1, 0, 0] for _ in range(len(bbox_lines))]  # red
        elif color == 'green':
            colors = [[0, 1, 0] for _ in range(len(bbox_lines))]  # green
        else:
            print("请输入 green 或者 red。green 表示预测框，red 表示真值框。")

        all_bboxes = o3d.geometry.LineSet()
        ang = bbox.orientation.radians * bbox.orientation.axis[-1]
        corners_3d = get_tensor_corners_batch(torch.tensor(bbox.center).unsqueeze(0),
                                 torch.tensor(bbox.wlh).unsqueeze(0),
                                 torch.tensor(ang).unsqueeze(0))
        corners_3d = np.array(corners_3d.squeeze(0).data.cpu())
        o3d_bbox = o3d.geometry.LineSet()
        o3d_bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
        o3d_bbox.colors = o3d.utility.Vector3dVector(colors)
        o3d_bbox.points = o3d.utility.Vector3dVector(corners_3d)
        all_bboxes += o3d_bbox

        return all_bboxes

    def show(self):
        # 创建窗口
        self.vis.create_window(window_name="Open3D_visualizer")
        opt = self.vis.get_render_option()
        opt.point_size = 1
        opt.background_color = np.asarray([0, 0, 0])
        # 添加点云、真值框、预测框
        self.vis.add_geometry(self.points)
        if self.gt_boxes is not None:
            self.vis.add_geometry(self.gt_boxes)
        if self.pred_boxes is not None:
            self.vis.add_geometry(self.pred_boxes)

        self.vis.get_view_control().rotate(180.0, 0.0)
        self.vis.run()


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1)
            )

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz.float())
        return position_embedding