
# Event Calibration Processing Code for DRTD

This code is used to demonstrate the calibration process for event modality.  
It shows how to calibrate event images using the calibration parameters from the DRTD.

Environment:
Python 3.9
Required libraries: numpy, os, glob, cv2, h5py

Steps:
1. Run the event_img_process.py script to generate the calibrated event images for the DRTD.
The input data is the hdf5 file from the input_data directory. 
The output data is the calibrated event image, saved under output_data directory. 

Commands:
Generate calibrated event images for the DRTD:
  ```bash
  python event_img_process.py
  ```





