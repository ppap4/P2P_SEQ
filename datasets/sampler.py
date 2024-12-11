import numpy as np
import torch
from nuscenes.utils import geometry_utils
from torch.utils.data import Dataset
from . import points_utils
from mmengine.registry import DATASETS
from datasets.misc_utils import get_history_frame_ids_and_masks, \
    create_history_frame_dict, \
    generate_timestamp_prev_list, \
    generate_virtual_points
import open3d as o3d


class KalmanFiltering:
    def __init__(self, bnd=[1, 1, 10]):
        self.bnd = bnd
        self.reset()

    def sample(self, n=10):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def addData(self, data, score):
        score = score.clip(min=1e-5)  # prevent sum=0 in case of bad scores
        self.data = np.concatenate((self.data, data))
        self.score = np.concatenate((self.score, score))
        self.mean = np.average(self.data, weights=self.score, axis=0)
        self.cov = np.cov(self.data.T, ddof=0, aweights=self.score)

    def reset(self):
        self.mean = np.zeros(len(self.bnd))
        self.cov = np.diag(self.bnd)
        if len(self.bnd) == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.array([])


@DATASETS.register_module()
class TrainSampler(torch.utils.data.Dataset):

    def __init__(self, dataset=None, cfg=None):
        super().__init__()
        self.config = cfg
        self.dataset = DATASETS.build(dataset)
        self.num_candidates = cfg.num_candidates
        num_frames_total = 0
        self.tracklet_start_ids = [num_frames_total]
        for i in range(self.dataset.get_num_tracklets()):
            num_frames_total += self.dataset.get_num_frames_tracklet(i)
            self.tracklet_start_ids.append(num_frames_total)

    @staticmethod
    def processing(data, config):
        prev_frame = data['prev_frame']
        this_frame = data['this_frame']
        candidate_id = data['candidate_id']
        prev_pc, prev_box = prev_frame['pc'], prev_frame['3d_bbox']
        this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']

        if config.target_thr is not None:
            num_points_in_prev_box = geometry_utils.points_in_box(prev_box, prev_pc.points).sum()
            assert num_points_in_prev_box > config.target_thr, 'not enough target points'

        if candidate_id == 0:
            bbox_offset = np.zeros(4)
        else:
            bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=4)
            bbox_offset[3] = bbox_offset[3] * 5.0

        pcd = o3d.geometry.PointCloud()
        o3d_bbox = o3d.geometry.LineSet()
        o3d_bbox.lines = o3d.utility.Vector2iVector(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        corners = np.array([[2.0000, 0.5000, 0.5000], [2.0000, -0.5000, 0.5000], [2.0000, -0.5000, -0.5000],
                            [2.0000, 0.5000, -0.5000], [-2.0000, 0.5000, 0.5000], [-2.0000, -0.5000, 0.5000],
                            [-2.0000, -0.5000, -0.5000], [-2.0000, 0.5000, -0.5000]])
        ref_box = points_utils.getOffsetBB(
            prev_box, bbox_offset, limit_box=False, use_z=True, degrees=True)
        prev_frame_pc = points_utils.crop_pc_in_range(prev_pc, ref_box, config.point_cloud_range)

        if candidate_id == 0:
            bbox_offset = np.zeros(4)
        else:
            gaussian = KalmanFiltering(bnd=[1, 1, 1, 0])
            bbox_offset = gaussian.sample(1)[0]
            bbox_offset[0] *= 0.3
            bbox_offset[1] *= 0.1
            bbox_offset[2] *= 0.1

        base_box = points_utils.getOffsetBB(ref_box, bbox_offset, limit_box=False, use_z=True, degrees=True)
        this_frame_pc = points_utils.crop_pc_in_range(this_pc, base_box, config.point_cloud_range)
        if config.search_thr is not None:
            assert this_frame_pc.nbr_points() > config.search_thr, 'not enough search points'

        this_box = points_utils.transform_box(this_box, base_box)
        prev_box = points_utils.transform_box(prev_box, ref_box)
        ref_box = points_utils.transform_box(ref_box, ref_box)

        prev_points = prev_frame_pc.points.T
        this_points = this_frame_pc.points.T
        if config.regular_pc:
            prev_points, _ = points_utils.regularize_pc(prev_points, 1024)
            this_points, _ = points_utils.regularize_pc(this_points, 1024)
        else:
            if prev_points.shape[0] < 1:
                prev_points = np.zeros((1, 4), dtype='float32')
            if this_points.shape[0] < 1:
                this_points = np.zeros((1, 4), dtype='float32')

        if config.flip:
            prev_points, prev_box, this_points, this_box = \
                points_utils.flip_augmentation(prev_points, prev_box, this_points, this_box)

        theta = this_box.orientation.degrees * this_box.orientation.axis[-1]

        box_label = this_box.center
        inputs = {'prev_points': torch.as_tensor(prev_points, dtype=torch.float32),
                  'this_points': torch.as_tensor(this_points, dtype=torch.float32),
                  'wlh': torch.as_tensor(ref_box.wlh, dtype=torch.float32)}
        data_samples = {
            'box_label': torch.as_tensor(box_label, dtype=torch.float32),
            'theta': torch.as_tensor([0.2 * theta], dtype=torch.float32),
        }
        data_dict = {
            'inputs': inputs,
            'data_samples': data_samples,
        }

        return data_dict

    def get_anno_index(self, index):
        return index // self.num_candidates

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        return self.dataset.get_num_frames_total() * self.num_candidates

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:
            for i in range(0, self.dataset.get_num_tracklets()):
                if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                    tracklet_id = i
                    this_frame_id = anno_id - self.tracklet_start_ids[i]
                    prev_frame_id = max(this_frame_id - 1, 0)
                    frame_ids = (prev_frame_id, this_frame_id)
            prev_frame, this_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            data = {
                "prev_frame": prev_frame,
                "this_frame": this_frame,
                "candidate_id": candidate_id}
            return self.processing(data, self.config)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]


@DATASETS.register_module()
class TestSampler(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = DATASETS.build(dataset)

    def __len__(self):
        return self.dataset.get_num_tracklets()

    def __getitem__(self, index):
        tracklet_annos = self.dataset.tracklet_anno_list[index]
        frame_ids = list(range(len(tracklet_annos)))
        return self.dataset.get_frames(index, frame_ids)

@DATASETS.register_module()
class MotionTrackingSamplerMF(torch.utils.data.Dataset):

    def __init__(self, dataset=None, cfg=None):
        super().__init__()
        self.config = cfg
        self.dataset = DATASETS.build(dataset)
        self.num_candidates = cfg.num_candidates
        num_frames_total = 0
        self.tracklet_start_ids = [num_frames_total]
        for i in range(self.dataset.get_num_tracklets()):
            num_frames_total += self.dataset.get_num_frames_tracklet(i)
            self.tracklet_start_ids.append(num_frames_total)

    @staticmethod
    def processing(data, config):
        """

        :param data:
        :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
        :return:
        point_sample_size
        bb_scale
        bb_offset
        """
        prev_frames = data['prev_frames']
        this_frame = data['this_frame']
        candidate_id = data['candidate_id']
        valid_mask = data['valid_mask']
        num_hist = len(valid_mask)
        empty_counter = 0

        prev_pcs = [prev_frames[key]['pc'] for key in
                    sorted(prev_frames, key=lambda k: abs(int(k)))]  # Ordered point clouds, -1, -2, -3
        prev_boxs = [prev_frames[key]['3d_bbox'] for key in
                     sorted(prev_frames, key=lambda k: abs(int(k)))]  # Ordered point clouds, -1, -2, -3
        this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']

        # Check the number of empty boxes
        for prev_box, prev_pc in zip(prev_boxs, prev_pcs):
            num_points_in_prev_box = geometry_utils.points_in_box(prev_box, prev_pc.points[0:3, :]).sum()
            if num_points_in_prev_box < config.limit_num_points_in_prev_box:
                empty_counter += 1
        assert empty_counter < config.empty_box_limit, 'not enough valid box'

        ref_boxs = []
        for i, prev_box in enumerate(prev_boxs):  # Apply a random offset to each box, not uniformly
            if candidate_id == 0:
                sample_offsets = np.zeros(3)
            else:
                sample_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
                sample_offsets[2] = sample_offsets[2] * (5 if config.degrees else np.deg2rad(5))
            ref_box = points_utils.getOffsetBB(prev_box, sample_offsets, limit_box=False, use_z=True, degrees=False)
            ref_boxs.append(ref_box)
        # pcd = o3d.geometry.PointCloud()
        # o3d_bbox = o3d.geometry.LineSet()
        # o3d_bbox.lines = o3d.utility.Vector2iVector(
        #     [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        # corners = np.array([[2.0000, 0.5000, 0.5000], [2.0000, -0.5000, 0.5000], [2.0000, -0.5000, -0.5000],
        #          [2.0000, 0.5000, -0.5000], [-2.0000, 0.5000, 0.5000], [-2.0000, -0.5000, 0.5000],
        #          [-2.0000, -0.5000, -0.5000], [-2.0000, 0.5000, -0.5000]])
        # o3d_bbox.points = o3d.utility.Vector3dVector(corners)
        prev_frame_pcs = []
        for i, prev_pc in enumerate(prev_pcs):
            prev_frame_pc = points_utils.crop_pc_in_range(prev_pc, ref_boxs[i], config.point_cloud_range)
            # prev_frame_pc = points_utils.generate_subwindow_with_aroundboxs(prev_pc, ref_boxs[i], ref_boxs[0],
            #                                                                 scale=config.bb_scale,
            #                                                                 offset=config.bb_offset)
            prev_frame_pcs.append(prev_frame_pc)


        if candidate_id == 0:
            bbox_offset = np.zeros(4)
        else:
            gaussian = KalmanFiltering(bnd=[1, 1, 1, 0])
            bbox_offset = gaussian.sample(1)[0]
            bbox_offset[0] *= 0.3
            bbox_offset[1] *= 0.1
            bbox_offset[2] *= 0.1

        base_box = points_utils.getOffsetBB(ref_boxs[0], bbox_offset, limit_box=False, use_z=True, degrees=False)
        this_frame_pc = points_utils.crop_pc_in_range(this_pc, base_box, config.point_cloud_range)
        if config.search_thr is not None:
            assert this_frame_pc.nbr_points() > config.search_thr, 'not enough search points'
        # this_frame_pc = points_utils.generate_subwindow_with_aroundboxs(this_pc, ref_boxs[0], ref_boxs[0],
        #                                                                 scale=config.bb_scale,
        #                                                                 offset=config.bb_offset)

        # tmp_box = points_utils.transform_box(this_box, ref_boxs[0])
        this_box = points_utils.transform_box(this_box, ref_boxs[0])
        prev_boxs = [points_utils.transform_box(prev_box, ref_boxs[0]) for prev_box in prev_boxs]
        ref_boxs = [points_utils.transform_box(ref_box, ref_boxs[0]) for ref_box in ref_boxs]
        motion_boxs = [points_utils.transform_box(this_box, ref_box) for ref_box in ref_boxs]

        # Resample each frame of the point cloud to a specific number
        prev_points_list = [prev_frame_pc.points.T if prev_frame_pc.points.shape[1] >= 1 else np.zeros((1, 4), dtype='float32') for
                            prev_frame_pc in prev_frame_pcs]
        this_points = this_frame_pc.points.T
        if this_points.shape[0] < 1:
            this_points = np.zeros((1, 4), dtype='float32')

        # seg_label_this = geometry_utils.points_in_box(this_box, this_points.T[:3, :], config.bb_scale).astype(int)
        # seg_label_prev_list = [geometry_utils.points_in_box(prev_box, prev_points.T[:3, :], config.bb_scale).astype(int)
        #                        for prev_box, prev_points in zip(prev_boxs, prev_points_list)]  # 应当只考虑xyz特征
        # seg_mask_prev_list = [geometry_utils.points_in_box(ref_box, prev_points.T[:3, :], config.bb_scale).astype(float)
        #                       for ref_box, prev_points in zip(ref_boxs, prev_points_list)]  # 应当只考虑xyz特征
        # if candidate_id != 0:
        #     for seg_mask_prev in seg_mask_prev_list:
        #         # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        #         # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        #         seg_mask_prev[seg_mask_prev == 0] = 0.2
        #         seg_mask_prev[seg_mask_prev == 1] = 0.8
        # seg_mask_this = np.full(seg_mask_prev_list[0].shape, fill_value=0.5)

        timestamp_prev_list = generate_timestamp_prev_list(valid_mask, [t.shape[0] for t in prev_points_list])
        timestamp_this = np.full((this_points.shape[0], 1), fill_value=0.1)

        prev_points_list = [
            np.concatenate([prev_points, timestamp_prev],
                           axis=-1)
            for prev_points, timestamp_prev in zip(
                prev_points_list, timestamp_prev_list)
        ]
        this_points = np.concatenate(
            [this_points, timestamp_this], axis=-1)

        stack_points_list = prev_points_list + [this_points]
        stack_points = np.concatenate(stack_points_list, axis=0)

        theta_this = this_box.orientation.degrees * this_box.orientation.axis[-1] if config.degrees else \
            this_box.orientation.radians * this_box.orientation.axis[-1]
        box_label = np.append(this_box.center, theta_this).astype('float32')
        theta_prev_list = [
            prev_box.orientation.degrees * prev_box.orientation.axis[-1]
            if config.degrees else prev_box.orientation.radians *
                                   prev_box.orientation.axis[-1] for prev_box in prev_boxs
        ]
        box_label_prev_list = [
            np.append(prev_box.center, theta_prev).astype('float32')
            for prev_box, theta_prev in zip(prev_boxs, theta_prev_list)
        ]

        # Generate a reference box sequence
        theta_ref_list = [
            ref_box.orientation.degrees * ref_box.orientation.axis[-1]
            if config.degrees else ref_box.orientation.radians *
                                   ref_box.orientation.axis[-1] for ref_box in ref_boxs
        ]
        ref_box_list = [
            np.append(ref_box.center, theta_ref).astype('float32')
            for ref_box, theta_ref in zip(ref_boxs, theta_ref_list)
        ]

        theta_motion_list = [
            motion_box.orientation.degrees * motion_box.orientation.axis[-1]
            if config.degrees else motion_box.orientation.radians *
                                   motion_box.orientation.axis[-1] for motion_box in motion_boxs
        ]

        motion_label_list = [
            np.append(motion_box.center, theta_motion).astype('float32')
            for motion_box, theta_motion in zip(motion_boxs, theta_motion_list)
        ]
        motion_state_label_list = [
            np.sqrt(np.sum((this_box.center - prev_box.center) ** 2))
            > config.motion_threshold for prev_box in prev_boxs
        ]

        inputs = {
            'stack_points': torch.as_tensor(stack_points, dtype=torch.float32),  # Historical first, then current
            'ref_boxs': torch.as_tensor(np.stack(ref_box_list, axis=0), dtype=torch.float32),
            'bbox_size': torch.as_tensor(this_box.wlh, dtype=torch.float32),
            'valid_mask': torch.as_tensor(np.array(valid_mask).astype('int'), dtype=int),
            'points_num': torch.tensor([x.shape[0] for x in stack_points_list]),
            # 'points': stack_points_list
        }
        data_samples = {
            'box_label': torch.as_tensor(box_label, dtype=torch.float32),
            'box_label_prev': torch.as_tensor(np.stack(box_label_prev_list, axis=0), dtype=torch.float32),
            'motion_state_label': torch.as_tensor(np.stack(motion_state_label_list, axis=0).astype('int'), dtype=int),
            'motion_label': torch.as_tensor(np.stack(motion_label_list, axis=0), dtype=torch.float32),
        }

        if getattr(config, 'box_aware', False):
            # stack_points_split = np.split(stack_points, num_hist + 1, axis=0)
            hist_points_list = stack_points_list[:num_hist]
            # prev_bc_list = [
            #     points_utils.get_point_to_box_distance(hist_points[:, :3], prev_box)
            #     for hist_points, prev_box in zip(hist_points_list, prev_boxs)
            # ]
            this_points_split = stack_points_list[-1]
            this_bc = points_utils.get_point_to_box_distance(this_points_split[:, :3], this_box)

            candidate_bc_prev_list = [
                points_utils.get_point_to_box_distance(hist_points[:, :3], prev_box)
                for hist_points, prev_box in zip(hist_points_list, ref_boxs)
            ]

            candidate_bc_this = np.zeros_like(this_bc)
            candidate_bc_prev_list = candidate_bc_prev_list + [candidate_bc_this]
            candidate_bc = np.concatenate(candidate_bc_prev_list, axis=0)
            # 'prev_bc': torch.as_tensor(np.concatenate(prev_bc_list, axis=0).astype('float32'), dtype=torch.float32)
            data_samples.update({
                              'this_bc': torch.as_tensor(this_bc.astype('float32'), dtype=torch.float32)})
            inputs.update({'candidate_bc': torch.as_tensor(candidate_bc.astype('float32'), dtype=torch.float32)})

        data_dict = {
            'inputs': inputs,
            'data_samples': data_samples,
        }

        return data_dict

    def get_anno_index(self, index):
        return index // self.num_candidates

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        return self.dataset.get_num_frames_total() * self.num_candidates

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:
            for i in range(0, self.dataset.get_num_tracklets()):
                if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                    tracklet_id = i
                    this_frame_id = anno_id - self.tracklet_start_ids[i]
                    prev_frame_ids, valid_mask = get_history_frame_ids_and_masks(this_frame_id, self.dataset.hist_num)

                    frame_ids = (0, this_frame_id)

            first_frame, this_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            prev_frames_tuple = self.dataset.get_frames(tracklet_id, frame_ids=prev_frame_ids)
            prev_frames_dict = create_history_frame_dict(prev_frames_tuple)
            data = {
                "first_frame": first_frame,
                "prev_frames": prev_frames_dict,
                "this_frame": this_frame,
                "candidate_id": candidate_id,
                "valid_mask": valid_mask, }

            return self.processing(data, self.config)
        except AssertionError:
            # return 1
            return self[torch.randint(0, len(self), size=(1,)).item()]