# EEH-R Dataset

## Dataset Structure

```
EEH-R/
├── YOLO/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── train.yaml
├── annotations_120fps (or 30fps)
│   ├── P04_01
│   │   ├── mocap_data
│   │   │   ├── frame_0000000000.jsonl
│   │   │   ├── frame_0000000001.jsonl
│   │   │   └── ...
│   │   └── mocap_data_camera
│   │       ├── frame_0000000000.jsonl
│   │       ├── frame_0000000001.jsonl
│   │       └── ...
|   ├── P04_02
|   ...
|
├── mano_annotations_120fps (or 30fps)
│   ├── P04_01
│   │   ├── mano_params_0000000000.json
│   │   ├── mano_params_0000000001.json
│   │   └── ...
│   ├── P04_02
│   ...
│
├── calibration
│   ├── calibration_20250814.xml
│   ├── calibration_20250815.xml
│   └── calibration_20250817.xml
├── data
│   ├── P04_01
│   │   ├── events
│   │   │   ├── events_0000000000.h5
│   │   │   ├── events_0000000001.h5
│   │   │   ├── ...
|   |   |
│   │   ├── event_frames
│   │   │   ├── event_frame_0000000000.jpg
│   │   │   ├── event_frame_0000000001.jpg
│   │   │   ├── ...
|   |   |
│   │   ├── lnes_frames
│   │   │   ├── frame_0000000000.png
│   │   │   ├── frame_0000000001.png
│   │   │   ├── ...
|   |   |
│   │   └── images
│   │       ├── frame_0000000000.png
│   │       ├── frame_0000000001.jpg
│   │       ├── ...
|   |
│   ├── P04_02
|   ...
|
├── mapping_date_to_id.csv
├── mapping_id_to_scene.csv
├── mapping_sync_event_start_frame.csv
├── train.txt
├── test.txt
├── valid.txt
└── README.md
```


- `YOLO/` (YOLO format files)　**※ v2_EEHR_yolo is a refined version of v1_EEHR_yolo. The results reported in the paper were obtained using models trained on v1_EEHR_yolo**
  -  images/
      - train/*.png
      - val/*.png
      - test/*.png
  -  lables/
      - train/*.txt
      - val/*.txt
      - test/*.txt
  -  train.yaml
- `annotations_120fps/` (or 30fps)
  - `PXX_YY/`
    - `mocap_data/`
      - `frame_0000000000.jsonl`
        ```json
        {
          "id": export_id,
          "left": {
            "joints_3d": left_cam.tolist(),
            "world": {
              "joints_3d": left_pts.tolist(),
              "rotation": left_rot_quat.tolist(), # (x, y, z, w) * number of joints
            },
          },
          "right": {
            "joints_3d": right_cam.tolist(),
            "world": {
              "joints_3d": right_pts.tolist(),
              "rotation": right_rot_quat.tolist(), # (x, y, z, w) * number of joints
            },
          },
          "camera": {
            "position": rigid_pt.tolist(),
            "rotation": rigid_rot_quat.tolist(), # (x, y, z, w)
          },
        }
        ```
      - `frame_0000000001.jsonl`
      - `...`
    - `mocap_data_camera/`
      - `frame_0000000000.jsonl`
        ```json
        {
          "id": export_id,
          "left_joints_3d": left_cam.tolist(),
          "right_joints_3d": right_cam.tolist(),
        }
        ```
      - `frame_0000000001.jsonl`

- mocap data
```
  |   |   |   |
  8   11  5   2
  |   |   |   |
  7   10  4   1   |
  |   |   |   |   14
  6   9   3   0   |
  \   \   |   /   13
    \   \  |  /   /
    \   \ | /  12
      \  -|-  /
          15
```

- `mano_annotations_30fps/` **※ mano_annotations_30fps_v2 is a refined version of mano_annotations_30fps. The numbers reported in the paper were trained on v1.**
  - `PXX_YY/`
    - `mano_params_0000000000.json`
    - `mano_params_0000000001.json`
    - `...`
- `mano_annotations_120fps/`
  - `PXX_YY/`
    - `mano_params_0000000000.json`
    - `mano_params_0000000001.json`
    - `...`
    ```json
    {
      "frame_id": 0,
      "left_hand": {
        "vertices": vertices.tolist(),
        "joints": joints.tolist(),
        "global_orient": global_orient.tolist(),
        "hand_pose": hand_pose.tolist(),
        "betas": betas.tolist(),
        "transl": transl.tolist(),
      },
      "right_hand": {
        "vertices": vertices.tolist(),
        "joints": joints.tolist(),
        "global_orient": global_orient.tolist(),
        "hand_pose": hand_pose.tolist(),
        "betas": betas.tolist(),
        "transl": transl.tolist(),
      },
    }
    ```
- `calibration/`
  - `calibration_YYYYMMDD.xml`
- `data/`
  - `PXX_YY/`
    - `events/` (30fps) event data (t, x, y, p)
      - `events_0000000000.h5`
      - `events_0000000001.h5`
      - `...`
    - `event_frames/` (30fps) event frame image (blue: positive event, red: negative event)
      - `event_frame_0000000000.jpg`
      - `event_frame_0000000001.jpg`
      - `...`
    - `images/` (30fps) intensity image
      - `frame_0000000000.png`
      - `frame_0000000001.png`
      - `...`
- `mapping_date_to_id.csv`
  ```csv
  date,id
  20250814,P03
  20250814,P04
  20250815,P05
  20250815,P06
  20250815,P07
  20250817,P08
  20250817,P09
  20250817,P10
  ...
  ```
- `mapping_id_to_scene.csv`
  ```csv
  id,scene,light_or_dark
  P04_00,kitchen,light
  P04_01,kitchen,light
  P04_02,kitchen,light
  P04_03,kitchen,light
  P04_04,kitchen,dark
  P04_05,kitchen,dark
  P04_06,kitchen,dark
  P04_07,kitchen,dark
  P04_08,desk,dark
  P04_09,desk,dark
  P04_10,desk,dark
  P04_11,desk,light
  P04_12,desk,light
  P04_13,desk,dark
  ...
  ```
- `mapping_sync_event_start_frame.csv`
  - synchronize the event start frame with mocap data
  ```csv
  id,event_start_frame
  P04_01,134
  P04_02,74
  ...
  ```
- `README.md`

## Mapping Example
- event_start_frame (30fps) = 10
- event_frame (30fps) event_frame_0000000010.jpg → id 0
- PXX_YY_mocap_data.json,PXX_YY_mocap_data_camera.json (120fps)
  - event_frame_0000000010.jpg → json_id 0 * 4 = json_id 0 → id 0
  - event_frame_0000000011.jpg → json_id 1 * 4 = json_id 4 → id 1
- PXX_YY_edit.csv (30fps)
  - 10, 20 → unusable frames → json_id 10*4, 20*4 → id 10, 20

## 120fps Training / Evaluation

`events/`, `lnes_frames/` and `event_frames/` are stored at the base event rate
(30fps), while MANO / MoCap annotations also exist at 120fps. Setting
`annotation_fps: 120` makes `EEHRDataset` rebuild the LNES (and therefore the
`event_frame` and the ev2hands point cloud) for every sub-frame directly from
`events/*.h5`, instead of reading the pre-rendered `lnes_frames/*.png`.

### Sub-frame time window

- Annotation frame `f` maps to event frame `f // 4` (unchanged) and to
  sub-frame `f % 4`.
- MANO frame `4n` at 120fps is bit-identical to MANO frame `n` at 30fps, so
  sub-frame 0 is time-aligned with the 30fps frame and sub-frames 1..3 are
  `1/120s`, `2/120s`, `3/120s` later.
- The LNES window keeps its length (`lnes_window_sec`, default `1/30`) and is
  only shifted: `[T1 + k/120 - W, T1 + k/120)`, where `T1` is the boundary of
  event chunk `f // 4`. Raw event chunks follow a repeating 30/30/40ms pattern
  (mean exactly 1/30s), so a window may span the previous and the next chunk;
  both are pulled in when needed.
- Keeping the window length constant means the event density per frame matches
  the 30fps training data, so 30fps-trained checkpoints stay compatible.

### LNES rasterization

The reconstruction reproduces the stored PNGs: per pixel and polarity the most
recent event wins, its normalized timestamp is written as `0-255`
(`G` = positive, `R` = negative), and the 346-wide sensor is center-cropped to
260x260. Verified against `lnes_frames/*.png`: max difference 1 (rounding only).

### Config keys

```yaml
annotation_fps: 120       # 30 or 120
lnes_from_h5: false       # forced on when annotation_fps > 30;
                          # set true to also rebuild the 30fps LNES from h5
lnes_window_sec: 0.03333333333333333  # LNES integration window (1/30s)
```

Notes:

- Dataset size becomes 4x, so epoch time grows accordingly.
- GT masks (`gt_mask_root`) only exist at 30fps, so all four sub-frames of one
  event frame would share the same mask label. `use_gt_mask: true` together with
  `annotation_fps > 30` therefore raises a `ValueError`; set
  `allow_subframe_gt_mask: true` only if that misalignment is intended.
- Set `lnes_from_h5: true` at `annotation_fps: 30` to use the exact same
  reconstruction pipeline for 30fps runs (useful when comparing 30 vs 120fps).
