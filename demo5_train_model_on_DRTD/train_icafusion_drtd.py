import argparse
import logging
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo_test import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader_rgb_ir
from utils.general import logger, labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

from utils.datasets import RandomSampler
import global_var
import utils.datasets as _ds  # Patch target for the label-path resolver function

# =====================================================================
# Global random seed settings (ensure reproducibility)
# =====================================================================
RANDOM_SEED = 0          # Random seed; just change this value

DETERMINISTIC = False    # When True, enables fully deterministic algorithms (more reproducible, but some CUDA ops may error/become slower)

# =====================================================================
# Completely disable wandb (set environment variables before any wandb call)
# =====================================================================
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"


# =====================================================================
# Label-path resolver patch
# The original img2label_paths only recognizes 'visible' / 'infrared' directory names.
# The DRTD dataset uses 'rgb' / 'event' directory names; this extends its support,
# so training can proceed directly without renaming the data directories.
# Convention: replace the modality directory name in the image path with 'labels' to get the label path,
# i.e., the rgb and event modalities share the same set of labels (aligning with the standard dataset practice).
# =====================================================================
def _img2label_paths_drtd(img_paths):
    sb = 'labels'
    modality_dirs = ('visible', 'infrared', 'rgb', 'event', 'ir', 'thermal', 'lwir')
    out = []
    for x in img_paths:
        parts = x.replace('\\', '/').split('/')
        sa = None
        for m in modality_dirs:
            if m in parts:
                sa = m
                break
        if sa is None:
            raise ValueError(
                f"img2label_paths: modality directory not found in path (visible/infrared/rgb/event/ir/thermal): {x}")
        # Take the last occurrence of the modality directory to avoid false matches when a parent directory happens to contain the same substring
        idx = len(parts) - 1 - parts[::-1].index(sa)
        parts[idx] = sb
        new = '/'.join(parts)
        new = new.rsplit('.', 1)[0] + '.txt'
        out.append(new.replace('/', os.sep))
    return out


# Apply patch (replace the module-level function; calls from inside dataset classes will use this version)
_ds.img2label_paths = _img2label_paths_drtd


def train_rgb_ir(hyp, opt, device, tb_writer=None):
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(seed=RANDOM_SEED + (rank if rank > 0 else 0), deterministic=DETERMINISTIC)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming


    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    #pretrained = False
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        new_state_dict = state_dict
        for key in list(state_dict.keys()):
            new_state_dict[key[:6] + str(int(key[6])+10) + key[7:]] = state_dict[key]
        model.load_state_dict(new_state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path_rgb = data_dict['train_rgb']
    test_path_rgb = data_dict['val_rgb']
    train_path_ir = data_dict['train_ir']
    test_path_ir = data_dict['val_ir']
    labels_path = data_dict['path'] + '/labels/test'
    labels_list = os.listdir(labels_path)
    labels_list.sort()

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(pg0)} weight, {len(pg1)} weight (no decay), {len(pg2)} bias")
    del pg0, pg1, pg2

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # print("nl", nl)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, batch_size, gs, opt,
                                                   hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                                   world_size=opt.world_size, workers=opt.workers,
                                                   image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader, testdata = create_dataloader_rgb_ir(test_path_rgb, test_path_ir,imgsz_test, 32, gs, opt,
                                                        hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                                        rank=-1, world_size=opt.world_size, workers=opt.workers,
                                                        pad=0.5, prefix=colorstr('val: '))

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    MRresult = 0.0
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'rank', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            imgs_rgb = imgs[:, :3, :, :]
            imgs_ir = imgs[:, 3:, :, :]

            # FQY my code: visualize training data
            flage_visual = global_var.get_value('flag_visual_training_dataset')
            if flage_visual:
                from torchvision import transforms
                unloader = transforms.ToPILImage()
                for num in range(batch_size):
                    image = imgs[num, :3, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = unloader(image)
                    image.save('example_%s_%s_%s_color.jpg'%(str(epoch), str(i), str(num)))
                    image = imgs[num, 3:, :, :].cpu().clone()  # clone the tensor
                    image = image.squeeze(0)  # remove the fake batch dimension
                    image = unloader(image)
                    image.save('example_%s_%s_%s_ir.jpg'%(str(epoch), str(i), str(num)))

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                # pred = model(imgs)  # forward
                pred = model(imgs_rgb, imgs_ir)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                if ni < 3:
                    f1 = save_dir / f'train_batch{ni}_vis.jpg'
                    f2 = save_dir / f'train_batch{ni}_inf.jpg'
                    Thread(target=plot_images, args=(imgs_rgb, targets, paths, f1), daemon=True).start()
                    Thread(target=plot_images, args=(imgs_ir, targets, paths, f2), daemon=True).start()

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, MRresult, times = test.test(data_dict,
                                                           batch_size=32,
                                                           imgsz=imgsz_test,
                                                           model=ema.ema,
                                                           single_cls=opt.single_cls,
                                                           dataloader=testloader,
                                                           save_dir=save_dir,
                                                           save_txt=True,
                                                           save_conf=True,
                                                           verbose=nc < 50 and final_epoch,
                                                           plots=plots and final_epoch,
                                                           wandb_logger=wandb_logger,
                                                           compute_loss=compute_loss,
                                                           is_coco=is_coco,
                                                           labels_list=labels_list,
                                                           )

            # log
            keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/rank_loss',  # train loss
                    'TP', 'FP', 'FN', 'F1', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'val/rank_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2',  # learning rate
                    'MR_all', 'MR_day', 'MR_night', 'MR_near', 'MR_medium', 'MR_far', 'MR_none', 'MR_partial', 'MR_heavy', 'Recall_all'  # MR
                    ]
            vals = list(mloss) + list(results) + lr + MRresult
            dicts = {k: v for k, v in zip(keys, vals)}  # dict
            file = save_dir / 'results.csv'
            n = len(dicts) + 1  # number of cols
            s = '' if file.exists() else (('%s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            #wandb_logger.end_epoch(best_result=best_fitness == fi)
            # fi = MRresult[0]
            # if fi < best_fitness:
            #     best_fitness = fi

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    t1 = time.time()
    t = t1 - t0
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(file=save_dir / 'results.csv')  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        for m in (last, best) if best.exists() else (last):  # speed, mAP tests
            results, _, MRresult, _ = test.test(opt.data,
                                                batch_size=32,
                                                imgsz=imgsz_test,
                                                conf_thres=0.001,
                                                iou_thres=0.5,
                                                model=attempt_load(m, device).half(),
                                                single_cls=opt.single_cls,
                                                dataloader=testloader,
                                                save_dir=save_dir,
                                                save_txt=True,
                                                save_conf=True,
                                                save_json=False,
                                                plots=False,
                                                is_coco=is_coco,
                                                labels_list=labels_list,
                                                verbose=nc > 1,
                                                )

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    # =================================================================
    #  All training parameters are defined directly here; no command-line input is required.
    #  Just run:  python train.py
    #
    #  By default, the standard configuration from the ICAFusion paper/official repo is used:
    #    - Backbone/fusion model: yolov5l + Transfusion (ICAFusion bidirectional cross-attention fusion)
    #    - Optimizer: SGD, lr0=0.01, momentum=0.937, weight_decay=5e-4
    #    - Input size: 640x640, epochs=60
    #    (the above hyperparameters come from data/hyp.scratch.yaml)
    # =================================================================
    opt = argparse.Namespace(
        # ---------- Model and data ----------
        weights='yolov5l.pt',                                            # COCO pretrained weights (auto-downloaded); set to '' to disable pretraining
        cfg='./models/transformer/yolov5l_Transfusion_FLIR.yaml',        # ICAFusion fusion model structure (nc will be overridden by the data yaml)
        data='./data/multispectral/DRTD_config_for_ICAFusion.yaml',              # This file already contains the DRTD configuration (nc=4)
        hyp='./data/hyp.scratch.yaml',                                   # Default hyperparameters for training from scratch (SGD/lr0=0.01...)

        # ---------- Training scale ----------
        # epochs=60,                       # Default training epochs from the paper
        epochs=30,                       # Default training epochs from the paper
        batch_size=32,                    # Total batch size per single GPU; reduce (e.g. to 4) if VRAM is insufficient
        img_size=[640, 640],             # [train, test] input resolution
        workers=8,                       # Number of dataloader workers
        device='0',                      # GPU index; multi-GPU e.g. '0,1'; use 'cpu' for CPU

        # ---------- Optimizer ----------
        adam=False,                      # False=SGD (paper default); True=Adam
        linear_lr=False,                 # False=cosine (OneCycle), consistent with the paper
        label_smoothing=0.0,
        sync_bn=False,
        multi_scale=False,

        # ---------- Data loading / augmentation ----------
        rect=False,
        cache_images=False,
        image_weights=False,
        quad=False,
        single_cls=False,                # Multi-class (DRTD has 4 classes); keep False
        noautoanchor=False,              # Keep automatic anchor check enabled

        # ---------- Saving / evaluation ----------
        project='runs/train',
        name='DRTD_ICAFusion_seed0',

        exist_ok=False,
        nosave=False,
        notest=False,
        save_period=-1,

        # ---------- Misc (defaults are fine) ----------
        resume=False,
        evolve=False,
        bucket='',
        entity=None,
        upload_dataset=False,
        bbox_interval=-1,
        artifact_alias='latest',
        local_rank=-1,
    )

    # Toggle for visualizing paired training images (disabled by default)
    global_var._init()
    global_var.set_value('flag_visual_training_dataset', False)

    # ---------------- DDP / distributed variables (defaults are fine for single-GPU training) ----------------
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # ---------------- Check file paths ----------------
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)
    assert len(opt.cfg) or len(opt.weights), 'Either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # Pad to [train, test]
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # ---------------- Select device ----------------
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:  # Multi-GPU DDP (no need to worry about this for single-GPU)
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be divisible by the number of GPUs'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # ---------------- Read hyperparameters ----------------
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)

    # ---------------- Set global random seed (also set at the main entry, ensures dataloader etc. are reproducible) ----------------
    init_seeds(seed=RANDOM_SEED, deterministic=DETERMINISTIC)
    logger.info(f'Random seed = {RANDOM_SEED}, deterministic = {DETERMINISTIC}')

    # ---------------- Start training ----------------
    logger.info(opt)
    tb_writer = None
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)
    train_rgb_ir(hyp, opt, device, tb_writer)
