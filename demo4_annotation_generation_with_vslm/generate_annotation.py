import os
import cv2
import numpy as np
import torch
from ultralytics import SAM, YOLO

# category mapping
category_dict_yolo = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
category_dict_DRTD = {0:'car', 1:'motorcycle', 2:'bus', 3:'truck'}
prompt_to_drtd_map = {2:0, 3:1, 5:2, 7:3}

# define different colors for different masks
colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], 
        [255, 128, 0], [128, 0, 255], [255, 128, 255], [0, 120, 255], [150, 128, 0], [128, 0, 51], 
        [255, 89, 0],  [110, 255, 0], [0, 130, 255], [255, 128, 75]]

def yolo_detect_and_sam_segment(image_path, yolo_model_path="yolo11x.pt", sam_model_path="sam2.1_b.pt", 
                                device="cuda:0", output_dir="output", category_dict=None, conf=0.5):
    basename = os.path.basename(image_path).split(".")[0]
    prompt_vis_path = os.path.join(output_dir, basename + '_prompt_vis.jpg')
    sam_seg_path = os.path.join(output_dir, basename + '_sam_seg_vis.jpg')
    sam_gen_annotation_txt_path = os.path.join(output_dir, basename + '.txt')
    annotation_txt_vis_path = os.path.join(output_dir, basename + '_annotation_vis.jpg')
    
    # load YOLO model for prompt generation
    vision_prompt_generator = YOLO(yolo_model_path)
    # load SAM model
    sam_model = SAM(sam_model_path)
    # read image
    image_bgr = cv2.imread(image_path)
    # convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # use YOLO for detection
    prompt_results = vision_prompt_generator(image_path, conf=conf, device=device, classes=list(category_dict.keys()))

    # extract detection results
    prpmpt_list = []
    prompt_info = []
    if prompt_results and len(prompt_results) > 0:
        prompt_result = prompt_results[0]
        if hasattr(prompt_result, 'boxes') and prompt_result.boxes is not None:
            boxes = prompt_result.boxes
            print(f"Found {len(boxes)} detections")
            # process each detection box
            for i, box in enumerate(boxes):
                # get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # calculate geometric center
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                # confidence = float(box.conf[0].cpu().numpy())
                class_name = category_dict.get(class_id, f"class_{class_id}")
                # add to point list
                prpmpt_list.append((center_x, center_y))
                # save detection information
                prompt_info.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': (center_x, center_y),
                    'class_id': class_id,
                    'class_name': class_name,
                    'shape' : prompt_result.orig_shape
                })
    # draw prompt points on image
    prompt_vis_image = image_bgr.copy()
    for i, info in enumerate(prompt_info):
        x1, y1, x2, y2 = info['bbox']
        center_x, center_y = info['center']
        class_name = info['class_name']
        # draw prompt point
        cv2.circle(prompt_vis_image, (center_x, center_y), 8, (0, 0, 255), -1)
    # save prompt vis results image
    cv2.imwrite(prompt_vis_path, prompt_vis_image)

    # run SAM segmentation with prompt
    sam_results = sam_model(image_rgb, 
                                    points=np.array(prpmpt_list, dtype=np.float32), 
                                    labels=np.ones(len(prpmpt_list), dtype=np.int32), 
                                    verbose=False, save=False, device=device)
    if sam_results and len(sam_results) > 0:
        sam_result = sam_results[0]
        if hasattr(sam_result, 'masks') and sam_result.masks is not None:
            masks = sam_result.masks.data  # get mask data
            # build combined overlay visualization (reference: v3 combined_overlay_bgr)
            sam_mask_vis_image = image_bgr.copy()
            combined_colored_mask = np.zeros_like(sam_mask_vis_image)
            for i, mask in enumerate(masks):
                color = colors[i % len(colors)]
                combined_colored_mask[mask.cpu().numpy() > 0] = color
            combined_overlay = cv2.addWeighted(sam_mask_vis_image, 0.7, combined_colored_mask, 0.3, 0)
            for j, info in enumerate(prompt_info):
                center_x, center_y = info['center']
                cv2.circle(combined_overlay, (center_x, center_y), 8, (0, 0, 255), -1)
            cv2.imwrite(sam_seg_path, combined_overlay)

            bbox_info = []
            # for each mask, calculate minimum bounding box
            for i, mask in enumerate(masks):
                # convert mask to numpy array
                mask_np = mask.cpu().numpy().astype(np.uint8)
                # find contours
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # merge all contour points
                    all_points = np.vstack(contours)
                    # calculate minimum bounding rectangle
                    rect = cv2.minAreaRect(all_points)
                    box = cv2.boxPoints(rect)
                    box = box.astype(np.int32)
                    # calculate axis-aligned bounding box (AABB)
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    bbox_info.append({
                                    'mask_id': i,
                                    'class_id': prompt_to_drtd_map[prompt_info[i]['class_id']],
                                    'minimum_bounding_box': {
                                        'x_min': x_min,
                                        'y_min': y_min,
                                        'x_max': x_max,
                                        'y_max': y_max,
                                        'width': x_max - x_min,
                                        'height': y_max - y_min
                                                        }
                                    })
            img_h, img_w = image_rgb.shape[0], image_rgb.shape[1]
            annotation_vis_image = image_bgr.copy()
            lines = []
            for bbox in bbox_info:
                x_min = bbox['minimum_bounding_box']['x_min']
                y_min = bbox['minimum_bounding_box']['y_min']
                x_max = bbox['minimum_bounding_box']['x_max']
                y_max = bbox['minimum_bounding_box']['y_max']
                class_id = bbox.get('class_id', -1)
                if class_id >= 0:
                    cx = ((x_min + x_max) / 2) / img_w
                    cy = ((y_min + y_max) / 2) / img_h
                    w = (x_max - x_min) / img_w
                    h = (y_max - y_min) / img_h
                    lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                cv2.rectangle(annotation_vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(annotation_vis_image, f"{category_dict_DRTD[int(class_id)]}", (int(x_min), max(int(y_min)-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(annotation_txt_vis_path, annotation_vis_image)
            with open(sam_gen_annotation_txt_path, 'w') as f:
                f.write("\n".join(lines))


def main():
    device = "cuda:0"
    output_folder = "./output_data/"  
    yolo_model_path = "./yolo11_pretrained_model/yolo11x.pt"  # YOLO model path
    sam_model_path = "./SAM2_pretrained_model/sam2.1_b.pt"  # SAM model path

    image_path = "./input_data/173_1736061813_1981964.jpg"
    # image_path = "./input_data/174_1736061823_4189897.jpg"
    # image_path = "./input_data/103_1736046710_4648495.jpg"

    yolo_detect_and_sam_segment(
                                image_path=image_path,
                                yolo_model_path=yolo_model_path,
                                sam_model_path=sam_model_path,
                                device=device,
                                output_dir=output_folder,
                                category_dict=category_dict_yolo,
                                conf=0.5
                                )


if __name__ == "__main__":
    
    main()
