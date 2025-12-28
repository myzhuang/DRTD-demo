
# Annotation generation with visual segmentation large model (VSLM)

This code demonstrates the automatic generation of DRTD label.

Environment:
Python 3.9
Required libraries: os, cv2, numpy, torch, ultralytics
ultralytics (version 8.3.189) is imported to run the code: https://github.com/ultralytics/ultralytics/tree/v8.3.198

Steps:
1. Download the yolo11x.pt model accoeding yolo11_pretrained_model/yolo_download_link.txt file and place it in the yolo11_pretrained_model directory.
2. Download the sam2.1_b.pt model accoeding SAM2_pretrained_model/sam_download_link.txt file and place it in the SAM2_pretrained_model directory.
3. Run the generate_annotation.py script to generate label file.
Commands:
  ```bash
  python generate_annotation.py
  ```
The visualization results are saved in the output_data directory. For example, generated prompt point visualization (the red dots), SAM segmentation result, generated annotation visualization (green boxes),and txt annotation file are all saved in the output_data directory.  


