
# Effective area visualization for DRTD and TUMTraf datasets

This code is companion code for the paper and the DRTD. It demonstrates the computation and visualization of the effective area for the DRTD and TUMTraf datasets.
It is a minimal, self-contained demo; all required input files are in the input_data directory and you do not need to obtain anything from DRTD or TUMTraf.
The visualization results are saved in the output_data directory.

Environment:
Python 3.9
Required libraries: numpy, matplotlib, h5py, cv2

Steps:
1. Run the get_DRTD_effective_area.py script to generate the effective area visualization for the DRTD.
2. Run the get_TUMTraf_effective_area.py script to generate the effective area visualization for the TUMTraf dataset.

Commands:
Generate the effective area visualization for the DRTD:
  ```bash
  python get_DRTD_effective_area.py
  ```
This script computes the effective area for the DRTD. The inputs are the event-stream HDF5 file 103_1736046733_2958794.hdf5 and the JPG file 103_1736046733_2958794.jpg from the input_data/DRTD directory. The visualization output is effective_area_of_DTRD_103_1736046733_2958794.jpg, saved under output_data. It also prints the pixel count of the effective area for DRTD: 548640.

Generate the effective area visualization for the TUMTraf dataset:
  ```bash
  python get_TUMTraf_effective_area.py
  ```   
This script computes the effective area for the TUMTraf dataset. The inputs from the input_data/TUMTraf directory are the fused RGB–event image 20231114-081517.972886.jpg, the event modality label file 20231114-081517.972886_ev_label.txt, and the RGB modality label file 20231114-081517.972886_rgb_label.txt. The visualization output is effective_area_of_TUMTraf_20231114-081517.972886.jpg, saved under output_data. It also prints the pixel count of the effective area for TUMTraf: 228416.
