
# RGB Calibration Processing Code for DRTD

This code is used to demonstrate the calibration processing for RGB modality.  
It shows how to calibrate RGB images using the calibration parameters from the DRTD.

Environment:
Python 3.9
Required libraries: numpy, os, glob, cv2

Steps:
1. Run the rgb_img_process.py script to generate the calibrated RGB images for the DRTD.

Commands:
Generate calibrated RGB images for the DRTD dataset:
  ```bash
  python rgb_img_process.py
  ```
The input data is the JPG file from the input_data directory. 
The output data is the calibrated RGB image, saved under output_data directory. 




