_base_ = '../../default_runtime.py'
data_dir = '/home/ubuntu/os/nuscenes'
category_name = 'Car'
batch_size = 64
point_cloud_range = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]
box_aware = True
use_rot = False


model = dict(
    type='seqtrack3d',
    backbone=dict(type='VoxelNeet',
                  points_features=14,
                  point_cloud_range=point_cloud_range,
                  voxel_size=[0.075, 0.075, 0.15],
                  grid_size=[21, 128, 128],
                  output_channels=128
                  ),
    fuser=dict(type='SEQFuser'),
    head=dict(
        type='Seq2SeqFormer',
        d_word_vec=64,
        d_model=64,
        d_inner=512,
        n_layers=3,
        n_head=4,
        d_k=64,
        d_v=64,
        n_position=1024 * 4,
        cfg=dict(
            center_weight=2,
            angle_weight=10.0,
            seg_weight=0.1,
            bc_weight=1,
            ref_center_weight=0.2,
            ref_angle_weight=1,
            motion_cls_seg_weight=0.1,
        ),
    ),
    cfg=dict(
        point_cloud_range=point_cloud_range,
        post_processing=False,
        use_rot=use_rot,
        num_candidates=4,
        target_thr=None,
        search_thr=5,
        regular_pc=False,
        flip=True,
        limit_num_points_in_prev_box=1,
        empty_box_limit=3,
        degrees=False,
        data_limit_box=False,
        bb_scale=1,
        bb_offset=2,
        motion_threshold=0.15,
        box_aware=True,
        hist_num=3,
        use_z=True,
        limit_box=False,
        IoU_space=3
    )
)

train_dataset = dict(
    type='MotionTrackingSamplerMF',
    dataset=dict(
        type='NuScenesMFDataset',
        path=data_dir,
        split='train_track',
        category_name=category_name,
        preloading=False,
        preload_offset=10,
        hist_num=3
    ),
    cfg=dict(
        num_candidates=4,
        target_thr=None,
        search_thr=5,
        point_cloud_range=point_cloud_range,
        regular_pc=False,
        flip=True,
        limit_num_points_in_prev_box=1,
        empty_box_limit=3,
        degrees=False,
        data_limit_box=False,
        bb_scale=1,
        bb_offset=2,
        motion_threshold=0.15,
        box_aware=True
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='NuScenesMFDataset',
        path=data_dir,
        split='val',
        category_name=category_name,
        preloading=False
    ),
)

train_dataloader = dict(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)
