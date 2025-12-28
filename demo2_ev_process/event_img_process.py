import cv2
import numpy as np
import h5py
import os
import glob
import h5py

height_cam, width_cam = 720, 1280

# event camera matrix and distortion coefficients. 
# They are from DRTD (DRTD/calibration/event_camera_intrinsics.yaml).
mtx_ev = np.array([[1.74722570e+03, 0.00000000e+00, 6.42214538e+02],
                    [0.00000000e+00, 1.74604304e+03, 3.08360732e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_ev = np.array([[-8.00216024e-01,  1.48864224e+00,  4.81119812e-03, -3.21381674e-03,  -4.13338009e+00]])

# offset and scale parameters for each scenario.
# These parameters are from DRTD (DRTD/calibration/cut_and_scale_data.yaml).
w_offset_s1, h_offset_s1, rgb_scale_s1 = 31.0, 62.0, 0.919000 # x, y, scale_ratio
w_offset_s2, h_offset_s2, rgb_scale_s2 = 26.0, 60.0, 0.888000 # x, y, scale_ratio
w_offset_s3, h_offset_s3, rgb_scale_s3 = 35.0, 63.0, 0.981000 # x, y, scale_ratio
w_offset_s4, h_offset_s4, rgb_scale_s4 = 37.0, 60.0, 1.037000 # x, y, scale_ratio

event_before_calib_path = '.\\input_data\\'
event_after_calib_path = '.\\output_data\\'

scene_map = {
    "scenario1_index": ['101', '102', '103', '104', '105', '106', '111', '112', '113', '114', 
                        '115', '118', '119', '120', '121', '122', '123', '124', '125', '126', 
                        '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', 
                        '137', '138', '139', '140', '141'],
    "scenario2_index": ['232', '233', '234', '235', '236', '237', '238', '239', '240', '241', 
                        '242', '243', '244', '245', '246', '247', '248'],
    "scenario3_index": ['152', '153', '154', '155', '156', '157', '158', '159', '160', '161', 
                        '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', 
                        '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', 
                        '182', '196', '197', '198'],
    "scenario4_index": ['283', '284', '285', '286', '287', '288', '289', '290', '291', '292', 
                        '293']
}

def events_to_diff_image(events0,events1,events2, sensor_size, strict_coord=True):
    """
    Place events into an image using numpy
    """
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


if __name__ == '__main__':

    event_list = glob.glob(event_before_calib_path+'*.hdf5')
    print('event hdf5 num to process:', len(event_list))
    for i,h in enumerate(event_list):
        basename = os.path.basename(h).replace('.hdf5','.jpg')
        index = basename.split('_')[0]
        hfile = h5py.File(h,'r')
        aa = hfile['events']
        x = np.array(aa[0])
        y = np.array(aa[1])
        time_index = [1] 
        dvs_img_ = events_to_diff_image(aa[0], aa[1], aa[2], (height_cam, width_cam), strict_coord=True)
        dvs_img_[dvs_img_>0] = 255
        dvs_img_[dvs_img_<0] = 128
        dvs_img_ = cv2.undistort(dvs_img_, mtx_ev, dist_ev, None, None)

        # cut and scale process according to the scene
        if index in scene_map["scenario1_index"]:
            w_offset, h_offset, rgb_scale = w_offset_s1, h_offset_s1, rgb_scale_s1 # x, y, scale_ratio
            left = int(w_offset + 0.5000 * width_cam * (1.0-rgb_scale)) 
            down = int(0.5000 * height_cam * (1.0+rgb_scale) - h_offset) 
            right = int(0.5000 * width_cam * (1.0+rgb_scale) + w_offset) 
            dvs_img = dvs_img_[0:down,left:right]
        elif index in scene_map["scenario2_index"]:
            w_offset, h_offset, rgb_scale = w_offset_s2, h_offset_s2, rgb_scale_s2 # x, y, scale_ratio
            left = int(w_offset + 0.5000 * width_cam * (1.0-rgb_scale)) 
            down = int(0.5000 * height_cam * (1.0+rgb_scale) - h_offset) 
            right = int(0.5000 * width_cam * (1.0+rgb_scale) + w_offset)    
            dvs_img = dvs_img_[0:down,left:right]
        elif index in scene_map["scenario3_index"]:
            w_offset, h_offset, rgb_scale = w_offset_s3, h_offset_s3, rgb_scale_s3 # x, y, scale_ratio
            left = int(w_offset + 0.5000 * width_cam * (1.0-rgb_scale)) 
            down = int(h_offset + 0.5000 * height_cam * (1.0-rgb_scale)) 
            dvs_img = dvs_img_[0:-down,left:]
        elif index in scene_map["scenario4_index"]:
            w_offset, h_offset, rgb_scale = w_offset_s4, h_offset_s4, rgb_scale_s4 # x, y, scale_ratio
            left = int(w_offset + 0.5000 * width_cam * (1.0-rgb_scale)) 
            down = int(h_offset + 0.5000 * height_cam * (1.0-rgb_scale)) 
            dvs_img = dvs_img_[0:-down,left:]
        else:
            print(h, basename, index)

        black = np.zeros((dvs_img.shape[0],dvs_img.shape[1],3), dtype=np.uint8)
        black[dvs_img>240,:] = [0,0,225]
        black[(dvs_img<135)&(dvs_img>100),:] = [255,0,0]
        black = cv2.resize(black, (width_cam,height_cam), interpolation=cv2.INTER_NEAREST)
        save_name = event_after_calib_path+basename
        cv2.imwrite(save_name, black)


