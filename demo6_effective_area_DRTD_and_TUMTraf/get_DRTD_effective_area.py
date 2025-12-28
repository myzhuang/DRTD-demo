import cv2  
import numpy as np  
import h5py

def events_to_diff_image(events0,events1,events2, sensor_size, strict_coord=True):
    xs = events0
    ys = events1
    ps = events2 * 2 - 1

    mask = (xs < sensor_size[1]) * (ys < sensor_size[0]) * (xs >= 0) * (ys >= 0)
    if strict_coord:
        assert (mask == 1).all()
    coords = np.stack((ys*mask, xs*mask))
    ps *= mask

    try:
        abs_coords = np.ravel_multi_index(coords, sensor_size)
    except ValueError:
        raise ValueError("Issue with input arrays! coords={}, min_x={}, min_y={}, max_x={}, max_y={}, coords.shape={}, sum(coords)={}, sensor_size={}".format(
            coords, min(xs), min(ys), max(xs), max(ys), coords.shape, np.sum(coords), sensor_size))

    img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0]*sensor_size[1])
    img = img.reshape(sensor_size)
    return img

if __name__ == "__main__":
       
    # This raw event hdf5 file is from DRTD dataset (DRTD/data/raw/train.zip/103_1736046733_2958794.hdf5).
    event_hdf5_path = '.\\input_data\\DRTD\\103_1736046733_2958794.hdf5'
    # This raw uncalibrated RGB jpg file is from the DRTD dataset (DRTD/data/raw/train/103_1736046733_2958794.jpg).
    rgb_path = '.\\input_data\\DRTD\\103_1736046733_2958794.jpg'
    img_result_save_path = '.\\output_data\\effective_area_of_DTRD_103_1736046733_2958794.jpg'

    # RGB camera matrix and distortion coefficients. They are from DRTD dataset (DRTD/data/calibration/rgb_camera_intrinsics.yaml).
    mtx_rgb = np.array([[1.77825701e+03, 0.00000000e+00, 6.73148125e+02],
                        [0.00000000e+00, 1.77829255e+03, 3.94380560e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_rgb = np.array([[-7.01676268e-01,  6.43982152e-01, -7.19575196e-04, -8.19403186e-03, -8.35652072e-01]])
    # event camera matrix and distortion coefficients. They are from DRTD dataset (DRTD/data/calibration/event_camera_intrinsics.yaml).
    mtx_ev = np.array([[1.74722570e+03, 0.00000000e+00, 6.42214538e+02],
                       [0.00000000e+00, 1.74604304e+03, 3.08360732e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_ev = np.array([[-8.00216024e-01,  1.48864224e+00,  4.81119812e-03, -3.21381674e-03,  -4.13338009e+00]])

    # Four points defined the polygon of event area, these points are from manual measurements.
    points=[(264,180), (1441,180), (1441,807), (264,807)]
    # height, width of rgb image
    height, width = 720, 1280
    
    # Create a black canvas 'black' larger than the RGB image by padding height and width; this canvas is used to place the image after overlaying event and RGB.
    pad_x_min, pad_x_max, pad_y_min, pad_y_max = 180, 180, 180, 180 
    black = np.zeros((720+pad_y_min+pad_y_max, 1280+pad_x_min+pad_x_max, 3), dtype=np.uint8)

    # Scaling and offset parameters; 
    # these values come from the DRTD dataset documentation（DRTD/data/calibration/）
    w_offset, h_offset, rgb_scale = 31.0, 62.0, 0.919000 # x, y, scale_ratio

    # Read event data from the HDF5 file.
    hfile=h5py.File(event_hdf5_path,'r')
    aa = hfile['events']
    # Parse the event data stream in the HDF5 file.
    xx = aa[0]
    yy = aa[1]
    pp = aa[2]
    tt = aa[3]
    # Accumulate the event stream and visualize it.
    dvs_img_ = events_to_diff_image(xx, yy, pp,  (720,1280), strict_coord=True)
    dvs_img_[dvs_img_>0] = 255
    dvs_img_[dvs_img_<0] = 128
    dvs_img_ = cv2.undistort(dvs_img_, mtx_ev, dist_ev, None, None)
    dvs_img = np.zeros((720+pad_y_min+pad_y_max, 1280+pad_x_min+pad_x_max), dtype=np.uint8)
    dvs_img[pad_y_min:pad_y_min+720, pad_x_min:pad_x_min+1280] = dvs_img_

    # Use a white rectangle to mark the boundary of the event image on the black canvas.
    cv2.rectangle(black, (pad_x_min, pad_y_min), (pad_x_min+width, pad_y_min+height), (255, 255, 255), 2)
    # Load the uncalibrated RGB image.
    rgb_img_orign = cv2.imread(rgb_path)
    # Calibrate the RGB image.
    rgb_img_calib = cv2.undistort(rgb_img_orign, mtx_rgb, dist_rgb, None, None)
    # Resize the calibrated RGB image.
    height_new = int(height*rgb_scale)
    width_new = int(width*rgb_scale)
    rgb_img = cv2.resize(rgb_img_calib, (width_new, height_new), interpolation=cv2.INTER_AREA)
    # Place the resized RGB image at the corresponding position on the black canvas.
    x_min, y_min = int((width)//2 + w_offset - width_new//2)+pad_x_min, int((height)//2 - h_offset - height_new//2)+pad_y_min
    x_max, y_max = x_min + width_new, y_min + height_new
    # Place the resized RGB image at the corresponding position on the black canvas.
    black[y_min:y_max, x_min:x_max, :] = rgb_img
    # Place the event image at the corresponding position on the black canvas as well.
    black[dvs_img>240,:] = [0,0,225]
    black[(dvs_img<135)&(dvs_img>100),:] = [255,0,0]
    
    # code for calculate the effective area of DRTD
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], color=255)
    enclosed_pixels = np.sum(mask == 255)
    print('DRTD')
    print('pixels of effective area:', enclosed_pixels)

    # Draw the effective area boundaries using yellow lines.
    for i in range(4):
        start = points[i]
        end = points[(i+1)%4]
        cv2.line(black, start, end, (0, 255, 255), thickness=2)

    # Save the result image.
    cv2.imwrite(img_result_save_path, black)