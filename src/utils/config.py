from dataclasses import dataclass, field

OUTPUT_HEIGHT = 260
OUTPUT_WIDTH = 346

MANO_CMPS = 15  # 15 for N-HOT3D (hot3d), 6 for the Ev2Hands pretrained model

AVAILABLE_LOSSES = [
    "vertices",
    "joints_3d",
    "joints_3d_relative",
    "joints_2d",
    "hand_pose",
    "global_orient",
    "shape",
    "transl",
    "inter_j3d",
    "inter_shape",
    "inter_global_orient",
    "inter_transl",
    "fingertip_3d",
    "bone_length",
]


@dataclass
class HandTrainConfig:
    batch_size: int = 32
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 0.01
    gamma: float = 0.8
    step_size: int = 1
    alpha: float = 0.3
    event_input_dir: str = ""
    gt_file_dir: str = ""
    rgb_dir: str = ""
    yolo_image_dir: str = ""
    dataset_divide: int = 1,
    object_library_path: str = ""
    checkpoint_path: str = ""
    left_checkpoint_path: str = ""
    right_checkpoint_path: str = ""
    is_get_image: bool = False
    augmentation: bool = False
    save_path: str = ""
    log_interval: int = 100
    save_interval: int = 5
    losses: list = field(default_factory=lambda: AVAILABLE_LOSSES)
    loss_vertices_ratio: float = 1.0
    loss_global_orient_ratio: float = 1.0
    loss_hand_pose_ratio: float = 1.0
    loss_j3d_ratio: float = 1.0
    loss_j3d_relative_ratio: float = 1.0
    loss_j2d_ratio: float = 1.0
    loss_shape_ratio: float = 1.0
    loss_transl_ratio: float = 1.0
    loss_inter_j3d_ratio: float = 1.0
    loss_inter_shape_ratio: float = 1.0
    loss_inter_global_orient_ratio: float = 1.0
    loss_inter_transl_ratio: float = 1.0
    loss_fingertip_3d_ratio: float = 5.0
    loss_bone_length_ratio: float = 1.0
    bbox_expand_ratio: float = 1.0
    use_mask: bool = False
    mask_dilation_kernel_size: int = 5
    mask_dilation_iterations: int = 2

@dataclass
class HandTestConfig:
    batch_size: int = 32
    event_input_dir: str = ""
    gt_file_dir: str = ""
    rgb_dir: str = ""
    yolo_image_dir: str = ""
    object_library_path: str = ""
    checkpoint_path: str = ""
    is_get_image: bool = False
    log_interval: int = 100
    save_path: str = ""
    bbox_expand_ratio: float = 1.0
    use_mask: bool = False
    mask_dilation_kernel_size: int = 5
    mask_dilation_iterations: int = 2


@dataclass
class SegmentationTrainConfig:
    batch_size: int = 8
    lr: float = 0.001
    weight_decay: float = 0.01
    gamma: float = 0.8
    step_size: int = 1
    epochs: int = 50
    save_model: bool = True
    log_interval: int = 100
    checkpoint_interval: int = 5
    save_path: str = ""
    event_input_dir: str = ""
    mask_input_dir: str = ""
    augment: bool = False
    save_interval: int = 5


@dataclass
class SegmentationTestConfig:
    batch_size: int = 8
    checkpoint_path: str = ""
    event_input_dir: str = ""
    mask_input_dir: str = ""
    log_interval: int = 100


@dataclass
class SegmentationConfig:
    seed: int = 42
    devices: list = field(default_factory=list)
    mode: str = "train"
    output_width: int = 346
    output_height: int = 260
    n_channels: int = 3
    n_classes: int = 2
    model: str = "UNet"
    sigmoid_threshold: float = 0.51
    train: SegmentationTrainConfig = SegmentationTrainConfig()
    test: SegmentationTestConfig = SegmentationTestConfig()


@dataclass
class HandEstimationConfig:
    seed: int = 42
    devices: list = field(default_factory=list)
    mode: str = "train"
    dataset_mode: str = "normal"
    model: str = ""
    loss: str = ""
    output_width: int = 346
    output_height: int = 260
    n_channels: int = 3
    n_classes: int = 1
    event_number_threshold: int = 2048
    sigmoid_threshold: float = 0.51
    mano_hand_model_path: str = ""
    mano_cmps: int = 15
    mano_use_pose_pca: bool = True
    output_mano_dim: int = 31 * 2 # 31*2: pose 15, shape 10, global_orient 3, transl 3 or 61*2: pose 45, shape 10, global_orient 3, transl 3
    focal_length: float = 5000.0
    segmentation_checkpoint_path: str = ""
    time_bin: int = 5
    kernel_size: int = 5
    itterations: int = 2
    is_sequence: bool = False
    augmentation: bool = False
    dataset_mode: str = "mask"
    dataset_divide: int = 1
    ignore_files_count: int = 50
    # Attention settings
    use_cross_attention: bool = False
    use_self_attention: bool = False
    attention_num_heads: int = 4
    attention_dropout: float = 0.1
    use_positional_encoding: bool = True
    num_attention_layers: int = 1
    train: HandTrainConfig = HandTrainConfig()
    valid: HandTestConfig = HandTestConfig()
    test: HandTestConfig = HandTestConfig()
