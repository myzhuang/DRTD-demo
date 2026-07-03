

# YOLO26, ICAFusion and RVT Training & Inference on DRTD

This demo trains, evaluates, and runs inference for YOLO26, ICAFusion and RVT on the DRTD (Event and RGB modalities). 
It uses Ultralytics YOLO 8.4.86 and includes automatic dataset folder switching to meet Ultralytics' required structure. 


## Requirements
- OS: Linux (recommended for training)
- Python 3.10
- PyTorch 2.1.0 + CUDA 12.1 (NVIDIA GPU recommended)
- Ultralytics 8.4.86


## YOLO26 Training for RGB/Event Single Modality
- Dataset Preparation
  - Edit `DRTD_cfg_for_yolo26.yaml` and set `path` to your DRTD dataset root, e.g.:
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
- The code for YOLO26 needs to download from link:https://github.com/ultralytics/ultralytics
- Set `DRTD_root_path` inside each training script to your dataset root (the folder containing `RGB/` and `event/`).
- Default hyperparameters in the scripts: `epochs=30`, `imgsz=640`, `batch=32`, `optimizer=AdamW`, `lr0=0.001`, `workers=8`, `device='cuda'`, `amp=False`.
- Outputs are saved to `output_data\<name>\` (for example, `output_data\yolov10_event\`).
- After training, each script automatically evaluates `output_data\<name>\weights\best.pt` on `split=test` and writes results (CSV, plots, confusion matrices) to the same directory.
- Copy a few test images to:
  - Event: `input_data\event_images_from_test\`
  - RGB: `input_data\rgb_images_from_test\`
- Run inference with the corresponding script and trained `best.pt`:
- Run the desired script:
  ```bash
  python train_yolo26_drtd_event.py
  python train_yolo26_drtd_rgb.py
  ```
- Predictions and visualizations are written under `output_data\<name>\`.
- Typical files under `output_data\<name>\`:
  - `weights\best.pt`, `weights\last.pt`
  - `results.csv`, `results.png`
  - `confusion_matrix.png`, `confusion_matrix_normalized.png`
  - `BoxP_curve.png`, `BoxR_curve.png`, `BoxPR_curve.png`, `BoxF1_curve.png`
  - `train_batch*.jpg`, `val_batch*_labels.jpg`, `val_batch*_pred.jpg`
  - `args.yaml`, `*_train_log.log`
- Automatic Folder Switching (Important)
- Functions `rename_yolo_images_before_run()` and `rename_back_after_run()` ensure Ultralytics sees the selected modality as `images/`.
- If an `images` folder already exists at the dataset root, the scripts raise: `images file is exist, please check the path.` Remove or rename the conflicting folder.


## ICAFusion Training for RGB-Event Dual-modality Fusion
- The code for ICAFusion needs to download from link: https://github.com/chanchanchan97/ICAFusion
- Adding config file
  - copy DRTD_config_for_ICAFusion.yaml to file `ICAFusion\data\multispectral\DRTD_config_for_ICAFusion.yaml`
- Dual modality data preparing


- Run the desired script:
  ```bash
  python train_icafusion_drtd.py
  ```

## RVT Training for Event Steam Modality
- The code for RVT needs to download from link: https://github.com/uzh-rpg/RVT
- Event stream format dataset processing
  ```bash
  python convert_to_rvt.py
  ```

- Run the desired script:
  ```bash
  python train_rvt_drtd.py
  ```


## Troubleshooting
- CUDA out of memory: reduce `batch`, lower `imgsz`, or enable `amp=True`.
- Weights not found: download the corresponding `.pt` files into `weights/`.
- Dataset path error: set `DRTD_root_path` in each training script and `path` in `DRTD_cfg.yaml` correctly.


## Citation
@misc{yolo26,
  title        = {Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
  author       = {Glenn Jocher and Jing Qiu and Mengyu Liu and Shuai Lyu and
                  Fatih Cagatay Akyon and Muhammet Esat Kalfaoglu},
  year         = {2026},
  eprint       = {2606.03748},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  doi          = {10.48550/arXiv.2606.03748},
  url          = {https://arxiv.org/abs/2606.03748}
}

@article{icafusion,
  title={ICAFusion: Iterative cross-attention guided feature fusion for multispectral object detection},
  author={Shen, Jifeng and Chen, Yifei and Liu, Yue and Zuo, Xin and Fan, Heng and Yang, Wankou},
  journal={Pattern Recognition},
  volume={145},
  pages={109913},
  year={2024},
}

@inproceedings{rvt,
  title={Recurrent vision transformers for object detection with event cameras},
  author={Gehrig, Mathias and Scaramuzza, Davide},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13884--13893},
  year={2023}
}



