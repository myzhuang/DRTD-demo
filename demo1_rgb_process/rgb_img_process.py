import cv2
import numpy as np
import os
import glob

height_cam, width_cam = 720, 1280

# RGB camera matrix and distortion coefficients. 
# They are from DRTD (DRTD/calibration/rgb_camera_intrinsics.yaml).
mtx_rgb = np.array([[1.77825701e+03, 0.00000000e+00, 6.73148125e+02],
                    [0.00000000e+00, 1.77829255e+03, 3.94380560e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_rgb = np.array([[-7.01676268e-01,  6.43982152e-01, -7.19575196e-04, -8.19403186e-03, -8.35652072e-01]])

# offset and scale parameters for each scenario.
# These parameters are from DRTD (DRTD/calibration/cut_and_scale_data.yaml).
w_offset_s1, h_offset_s1, rgb_scale_s1 = 31.0, 62.0, 0.919000 # x, y, scale_ratio
w_offset_s2, h_offset_s2, rgb_scale_s2 = 26.0, 60.0, 0.888000 # x, y, scale_ratio
w_offset_s3, h_offset_s3, rgb_scale_s3 = 35.0, 63.0, 0.981000 # x, y, scale_ratio
w_offset_s4, h_offset_s4, rgb_scale_s4 = 37.0, 60.0, 1.037000 # x, y, scale_ratio

rgb_before_calib_path = '.\\input_data\\'
rgb_after_calib_path = '.\\output_data\\'

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

if __name__ == '__main__':

    img_list = glob.glob(rgb_before_calib_path+'*.jpg')
    print('img num to process:', len(img_list))
    for i,h in enumerate(img_list):
        basename = os.path.basename(h)
        index = basename.split('_')[0]
        rgb_ = cv2.imread(rgb_before_calib_path+basename)
        rgb_ = cv2.undistort(rgb_, mtx_rgb, dist_rgb, None, None)
        # cut and scale process according to the scene
        if index in scene_map["scenario1_index"]:
            w_offset, h_offset, rgb_scale = w_offset_s1, h_offset_s1, rgb_scale_s1 # x, y, scale_ratio
            top = int(h_offset - 0.5000 * height_cam * (1.0-rgb_scale)) 
            rgb_ = cv2.resize(rgb_, (int(rgb_scale*width_cam), int(rgb_scale*height_cam)), interpolation=cv2.INTER_AREA)
            rgb_2 = rgb_[top:,:,:]
        elif index in scene_map["scenario2_index"]:
            w_offset, h_offset, rgb_scale = w_offset_s2, h_offset_s2, rgb_scale_s2 # x, y, scale_ratio
            top = int(h_offset - 0.5000 * height_cam * (1.0-rgb_scale))
            rgb_ = cv2.resize(rgb_, (int(rgb_scale*width_cam), int(rgb_scale*height_cam)), interpolation=cv2.INTER_AREA)
            rgb_2 = rgb_[top:,:,:]
        elif index in scene_map["scenario3_index"]:
            w_offset, h_offset, rgb_scale = w_offset_s3, h_offset_s3, rgb_scale_s3 # x, y, scale_ratio
            top = int(h_offset - 0.5000 * height_cam * (1.0-rgb_scale)) 
            right = int(w_offset - 0.5000 * width_cam * (1.0-rgb_scale)) 
            rgb_ = cv2.resize(rgb_, (int(rgb_scale*width_cam), int(rgb_scale*height_cam)), interpolation=cv2.INTER_AREA)
            rgb_2 = rgb_[top:,:-right]
        elif index in scene_map["scenario4_index"]:
            w_offset, h_offset, rgb_scale = w_offset_s4, h_offset_s4, rgb_scale_s4 # x, y, scale_ratio
            top = int(h_offset - 0.5000 * height_cam * (1.0-rgb_scale)) 
            right = int(w_offset - 0.5000 * width_cam * (1.0-rgb_scale)) 
            rgb_ = cv2.resize(rgb_, (int(rgb_scale*width_cam), int(rgb_scale*height_cam)), interpolation=cv2.INTER_AREA)
            rgb_2 = rgb_[top:,:-right]
        else:
            print(h, basename, index)
        rgb_3 = cv2.resize(rgb_2, (width_cam,height_cam), interpolation=cv2.INTER_AREA)
        save_name = rgb_after_calib_path+basename
        # print(a, i, save_name, acc_time_index)
        cv2.imwrite(save_name, rgb_3)


