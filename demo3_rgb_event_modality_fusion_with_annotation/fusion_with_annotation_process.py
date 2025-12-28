import cv2
import os
import glob

height_cam, width_cam = 720, 1280

class_dict = {
  0: 'Car',
  1: 'Motorcycle',
  2: 'Bus',
  3: 'Truck'}

rgb_jpg_path = '.\\input_data\\rgb\\'
event_jpg_path = '.\\input_data\\event\\'
label_path = '.\\input_data\\label\\'
save_path = '.\\output_data\\'

def draw_label(img, label_path):
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            labels = f.readlines()
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        class_name = class_dict[class_id]
        x_center = float(parts[1]) * width_cam
        y_center = float(parts[2]) * height_cam
        width = float(parts[3]) * width_cam
        height = float(parts[4]) * height_cam
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, f"{class_name}", (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
    return img

if __name__ == '__main__':

    rgb_list = glob.glob(rgb_jpg_path+'*.jpg')
    event_list = [i.replace(rgb_jpg_path,event_jpg_path) for i in rgb_list]
    label_list = [i.replace(rgb_jpg_path,label_path).replace('.jpg','.txt') for i in rgb_list]

    print('num to fusion:', len(rgb_list))

    for i, (r, e, l) in enumerate(zip(rgb_list, event_list, label_list)):
        rgb_img = cv2.imread(r)
        event_img = cv2.imread(e)

        rgb_img[event_img[:, :, 2] > 128] = [0,0,255]
        rgb_img[event_img[:, :, 0] > 128] = [255,0,0]

        rgb_label_img = draw_label(rgb_img, l)
        cv2.imwrite(save_path+os.path.basename(r), rgb_label_img)


