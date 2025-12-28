import cv2  
import numpy as np  
import os

class_dict = {0:'Pedestrian', 1:'Bicycle', 2:'Car', 3:'Motorcycle', 4:'Bus', 5:'Truck', 6:'Trailer'}

def draw_label(img, label_path, height, width, bbox_color = (0, 255, 0)):
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            labels = f.readlines()
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        class_name = class_dict[class_id]
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        bbox_width = float(parts[3]) * width
        bbox_height = float(parts[4]) * height
        x_min = int(x_center - bbox_width / 2)
        y_min = int(y_center - bbox_height / 2)
        x_max = int(x_center + bbox_width / 2)
        y_max = int(y_center + bbox_height / 2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), bbox_color, 2)
        # print(label, class_name, x_min, y_min, x_max, y_max)
        cv2.putText(img, f"{class_name}", (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bbox_color, 1)
    return img
  
if __name__ == "__main__":
    # image and label from TUMTraf dataset
    rgb_label_path = '.\\input_data\\TUMTraf\\20231114-081517.972886_rgb_label.txt'
    event_label_path = '.\\input_data\\TUMTraf\\20231114-081517.972886_ev_label.txt'
    RGB_img_path = '.\\input_data\\TUMTraf\\20231114-081517.972886.jpg'
    img_result_save_path = '.\\output_data\\effective_area_of_TUMTraf_20231114-081517.972886.jpg'
    # Four points defined the polygon of event area, these points are from manual measurements.
    points = [(113, 0), (117, 458), (611, 467), (607, 4)]
    img = cv2.imread(RGB_img_path)
    height, width = 480, 640
    
    # code for calculate the effective area of TUMTraf
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], color=255)
    enclosed_pixels = np.sum(mask == 255)
    print('TUMTraf')
    print('pixels of effective area:', enclosed_pixels)

    # draw label txt from rgb_label_path and event_label_path
    draw_label(img, event_label_path, height, width, bbox_color = (0, 255, 0)) # color is green
    draw_label(img, rgb_label_path, height, width, bbox_color = (0, 0, 255)) # color is red
    
    # draw yellow border of effective area
    for i in range(4):
        start = points[i]
        end = points[(i+1)%4]
        cv2.line(img, start, end, (0, 255, 255), thickness=2) # draw yellow border

    cv2.imwrite(img_result_save_path, img)

