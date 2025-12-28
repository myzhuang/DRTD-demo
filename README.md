# Dual-modal Roadside Traffic Dataset (DRTD) for Object Detection Task Code Repository (v1.0.0)

This repository corresponds to the manuscript under review at Scientific Data, "A Spatially Aligned RGB-Event Modality Dataset for Roadside Traffic Object Detection". It provides dataset-related processing and experimental demo code intended for illustration, visualization, and reproducibility, rather than a full production-level training framework. Each demo directory includes a detailed `README.md`; please follow the instructions there.
Demo 1, 2, 3, 4, and 6 are minimal, self-contained programs. Their inputs come from the `input_data` folder inside each demo, and outputs are written to the `output_data` folder. They do not require downloading the DRTD.
Demo 5 contains training and inference code, which requires downloading DRTD and configuring the path to the dataset.

## Version

This repository corresponds to version v1.0.0, which is used for generating the results reported in the manuscript.

## Repository Overview
- `demo1_rgb_process`: RGB modality calibration and rectification. Input: JPG files under `input_data`; output: calibrated RGB images to `output_data`.
- `demo2_ev_process`: Event modality calibration and rectification. Input: HDF5 files under `input_data`; output: rectified event images to `output_data`.
- `demo3_rgb_event_modality_fusion_with_annotation`: RGB–Event fusion with annotation visualization. Input: RGB and event images under `input_data`; output: fusion images with annotation visualization to `output_data`.
- `demo4_annotation_generation_with_vslm`: Annotation generation based on visual segmentation large model (VSLM). yolov11 model is use to be a visual prompter and SAM2 is use to be a VSLM. Download `yolo11x.pt` to `yolo11_pretrained_model`, and `sam2.1_b.pt` to `SAM2_pretrained_model`, then run `python generate_annotation.py`. Outputs files (prompt points, segmentation, annotation visualization, DRTD annotation txt file) are saved to `output_data`.
- `demo5_train_yolo_model_on_DRTD`: Training, evaluation, and inference for YOLOv9 / YOLOv10 / YOLOv11 on DRTD. Includes logic to temporarily rename the selected modality to `images`; training outputs and evaluation results are saved under `output_data\<name>\`.
- `demo6_effective_area_DRTD_and_TUMTraf`: Compute and visualize the effective area of DRTD and TUMTraf (valid pixel region). Scripts use the example inputs already in `input_data`; visualizations and pixel counts are written to `output_data`.

## Run Environment

- Operating system: Linux (recommended for training demos)
- Python >= 3.8
- NVIDIA GPU recommended for Demo4 and Demo5

## Dependencies
- Demo1 (RGB calibration processing)
  - Dependencies: `numpy`, `os`, `glob`, `cv2`
- Demo2 (Event calibration processing)
  - Dependencies: `numpy`, `os`, `glob`, `cv2`, `h5py`
- Demo3 (RGB-Event fusion with annotation visualization)
  - Dependencies: `os`, `glob`, `cv2`
- Demo4 (Annotation generation with visual segmentation large model (VSLM))
  - Dependencies: `os`, `cv2`, `numpy`, `torch`, `pathlib`, `ultralytics` (version 8.3.189)
- Demo5 (YOLO training/evaluation/inference)
  - Dependencies: `torch 2.1.0 + CUDA 12.1` (NVIDIA GPU recommended), `ultralytics` (version 8.3.189)
  - Training must be run on Linux. The scripts temporarily rename `RGB` or `event` to `images` at the dataset root and restore after training.
- Demo6 (effective area computation & visualization)
  - Dependencies: `numpy`, `matplotlib`, `h5py`, `cv2`

Install the dependencies and run according to each demo's `README.md`.

## Quick Start
- Demo1 (RGB calibration)
  - `cd demo1_rgb_process`
  - `python rgb_img_process.py`
- Demo2 (Event calibration)
  - `cd demo2_ev_process`
  - `python event_img_process.py`
- Demo3 (RGB-Event fusion with annotation)
  - `cd demo3_rgb_event_modality_fusion_with_annotation`
  - `python fusion_with_annotation_process.py`
- Demo4 (Annotation generation with visual segmentation large model (VSLM))
  - Place `yolo11x.pt` into `yolo11_pretrained_model` and `sam2.1_b.pt` into `SAM2_pretrained_model`
  - `cd demo4_annotation_generation_with_vslm`
  - `python generate_annotation.py`
- Demo5 (YOLO training/evaluation/inference)
  - `cd demo5_train_yolo_model_on_DRTD`
  - Set `path` in `DRTD_cfg.yaml` to the DRTD root
  - Set `DRTD_root_path` in each training script, then run:
    - `python train_yolov9_DRTD_event.py`
    - `python train_yolov9_DRTD_rgb.py`
    - `python train_yolov10_DRTD_event.py`
    - `python train_yolov10_DRTD_rgb.py`
    - `python train_yolov11_DRTD_event.py`
    - `python train_yolov11_DRTD_rgb.py`
  - See the directory `README.md` for inference scripts
- Demo6 (Effective area visualization)
  - `cd demo6_effective_area_DRTD_and_TUMTraf`
  - `python get_DRTD_effective_area.py`
  - `python get_TUMTraf_effective_area.py`

## Input/Output Conventions
- Input example: each demo's `input_data\...`
- Output example: each demo's `output_data\...` (e.g., visualization images, logs, label files, model weights, and evaluation results)

## Notes
- Demo4 and Demo5 involve deep-learning inference/training; a GPU environment is recommended.
- Demo5 requires the dataset structure expected by Ultralytics YOLO (`images/train`, `images/val`, `images/test`). The scripts will temporarily rename the selected modality to `images` and restore it after training; ensure there is no conflicting `images` folder at the dataset root.
- Version and path details are governed by each demo's `README.md`.

## Citation

If you use the DRTD, please cite the dataset paper:
"A Spatially Aligned RGB-Event Modality Dataset for Roadside Traffic Object Detection", Scientific Data (under review).
If you use this demo code, please also cite the above paper.

## License

This repository is released under the MIT License.
See the LICENSE file for details.
