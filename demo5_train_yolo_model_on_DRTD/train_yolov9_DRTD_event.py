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

    DRTD_root_path = '/home2/zmy/projects/20240821_evcar/dataset/aligned/'
    modality = 'event'
    model_type = 'yolov9t'

    rename_yolo_images_before_run(drtd_root = DRTD_root_path, modality = modality)

    # # training parameters
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
            'name': 'yolov9_event',
            'exist_ok': True,
            'verbose': True,
            'amp': False,  
            'cache': False   
            }
    
    # start to train
    model = YOLO('weights/' + model_type + '.pt')
    results = model.train(**args)

    # start to test 
    test_results_json_path = args['project']+"/"+args['name']+"/"+model_type+"_test_metrics.json"
    best_model = YOLO('output_data/'+args['name']+'/weights/best.pt')
    metrics = best_model.val(data=args['data'], split='test')

    rename_back_after_run(drtd_root = DRTD_root_path, modality = modality)

