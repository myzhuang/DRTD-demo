

# YOLO Training & Inference on DRTD

This demo trains, evaluates, and runs inference for YOLOv9 / YOLOv10 / YOLOv11 on the DRTD dataset (Event and RGB modalities). It uses Ultralytics YOLO 8.3.189 and includes automatic dataset folder switching to meet Ultralytics' required structure.

## Requirements
- OS: Linux (recommended for training)
- Python 3.10
- PyTorch 2.1.0 + CUDA 12.1 (NVIDIA GPU recommended)
- Ultralytics 8.3.189

## Dataset Preparation
- Edit `DRTD_cfg.yaml` and set `path` to your DRTD dataset root, e.g.:
  ```yaml
  path: /home/DRTD/aligned
  train: images/train
  val: images/val
  test: images/test
  names:
    0: Car
    1: Motorcycle
    2: Bus
    3: Truck
  ```
- The dataset root must contain the subfolders `RGB/` and `event/` with their own `train/`, `val/`, and `test/` subtrees.
- During training, the scripts temporarily rename the selected modality (`RGB` or `event`) to `images` before running, and restore it afterward. Ensure there is no existing `images` folder at the dataset root.

## Pretrained Weights
- Place pretrained weights under `weights/`:
  - `yolov9t.pt`
  - `yolov10n.pt`
  - `yolo11n.pt`

## Training
- Set `DRTD_root_path` inside each training script to your dataset root (the folder containing `RGB/` and `event/`).
- Run the desired script:
  ```bash
  python train_yolov9_DRTD_event.py
  python train_yolov9_DRTD_rgb.py
  python train_yolov10_DRTD_event.py
  python train_yolov10_DRTD_rgb.py
  python train_yolov11_DRTD_event.py
  python train_yolov11_DRTD_rgb.py
  ```
- Default hyperparameters in the scripts: `epochs=20`, `imgsz=640`, `batch=64`, `optimizer=AdamW`, `lr0=0.001`, `workers=8`, `device='cuda'`, `amp=False`.
- Outputs are saved to `output_data\<name>\` (for example, `output_data\yolov10_event\`).

## Evaluation
- After training, each script automatically evaluates `output_data\<name>\weights\best.pt` on `split=test` and writes results (CSV, plots, confusion matrices) to the same directory.

## Inference
- Copy a few test images to:
  - Event: `input_data\event_images_from_test\`
  - RGB: `input_data\rgb_images_from_test\`
- Run inference with the corresponding script and trained `best.pt`:
  ```bash
  python inference_yolov9_DRTD_event.py
  python inference_yolov9_DRTD_rgb.py
  python inference_yolov10_DRTD_event.py
  python inference_yolov10_DRTD_rgb.py
  python inference_yolov11_DRTD_event.py
  python inference_yolov11_DRTD_rgb.py
  ```
- Predictions and visualizations are written under `output_data\<name>\`.

## Output Directory
- Typical files under `output_data\<name>\`:
  - `weights\best.pt`, `weights\last.pt`
  - `results.csv`, `results.png`
  - `confusion_matrix.png`, `confusion_matrix_normalized.png`
  - `BoxP_curve.png`, `BoxR_curve.png`, `BoxPR_curve.png`, `BoxF1_curve.png`
  - `train_batch*.jpg`, `val_batch*_labels.jpg`, `val_batch*_pred.jpg`
  - `args.yaml`, `*_train_log.log`

## Automatic Folder Switching (Important)
- Functions `rename_yolo_images_before_run()` and `rename_back_after_run()` ensure Ultralytics sees the selected modality as `images/`.
- If an `images` folder already exists at the dataset root, the scripts raise: `images file is exist, please check the path.` Remove or rename the conflicting folder.

## Troubleshooting
- CUDA out of memory: reduce `batch`, lower `imgsz`, or enable `amp=True`.
- Weights not found: download the corresponding `.pt` files into `weights/`.
- Dataset path error: set `DRTD_root_path` in each training script and `path` in `DRTD_cfg.yaml` correctly.


