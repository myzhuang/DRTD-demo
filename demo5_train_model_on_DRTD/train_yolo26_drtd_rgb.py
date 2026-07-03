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

    print(
        drtd_root / 'images',
        '--->',
        'rename back',
        drtd_root / modality,
    )


if __name__ == '__main__':

    DRTD_root_path = '/home2/zmy/projects/20240821_evcar/dataset/aligned_rgb/'
    modality = 'rgb'

    # YOLO26 
    model_type = 'yolo26s'

    rename_yolo_images_before_run(
        drtd_root=DRTD_root_path,
        modality=modality
    )

    args = {
        'data': 'DRTD_cfg_rgb.yaml',

        'epochs': 30,
        'imgsz': 640,
        'batch': 32,

        'device': 'cuda',
        'workers': 8,

        'optimizer': 'AdamW',

        'lr0': 0.001,
        'lrf': 0.01,        #  lr = lr0 * lrf
        'cos_lr': True,     
        'warmup_epochs': 1, 

        'save': True,
        'save_period': 20,

        'project': 'output_data',
        'name': 'yolo26_rgb_seed0',

        'exist_ok': True,
        'verbose': True,

        'amp': False,
        'cache': False,

        'seed': 0
    }

    # train
    model = YOLO(f'weights/{model_type}.pt')

    results = model.train(**args)

    # test
    best_model = YOLO(
        f"output_data/{args['name']}/weights/best.pt"
    )

    metrics = best_model.val(
        data=args['data'],
        split='test'
    )

    rename_back_after_run(
        drtd_root=DRTD_root_path,
        modality=modality
    )