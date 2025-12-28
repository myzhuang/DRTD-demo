from ultralytics import YOLO
import os
from pathlib import Path

def rename_yolo_images_before_run(drtd_root='', modality="event"):
    drtd_root = Path(drtd_root)
    if os.path.exists(drtd_root / 'images'):
        msg = 'images file is exist, please check the path.'
        print(msg)
        raise FileExistsError(msg)
    else:
        os.rename(drtd_root / modality, drtd_root / 'images')
        print('rename ', drtd_root / modality, '--->', drtd_root / 'images')

def rename_back_after_run(drtd_root='', modality="event"):
    drtd_root = Path(drtd_root)
    os.rename(drtd_root / 'images', drtd_root / modality)
    print(drtd_root / 'images', '--->', 'rename back', drtd_root / modality,)

if __name__ == '__main__':

    modality = 'RGB'
    model_type = 'yolo11n'

    # training parameters
    args = {
            'data': 'DRTD_cfg.yaml',
            'epochs': 20,
            'imgsz': 640,
            'batch': 64,
            'device': 'cuda',  
            'workers': 8,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'save': True,
            'save_period': 20, 
            'project': 'output_data',
            'name': 'yolov11_rgb',
            'exist_ok': True,
            'verbose': True,
            'amp': False,     
            }
    
    # start to single image inference 
    best_model = YOLO('output_data/'+args['name']+'/weights/best.pt')
    results = best_model.predict(
                                source='./input_data/rgb_images_from_test/',  
                                imgsz=640,
                                conf=0.25,
                                iou=0.7,
                                save=True,                    
                                project=args['project'],
                                name=args['name'],
                                exist_ok=True
                                )



