import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# =====================================================================
# Training configuration: change it here, then just run `python train.py` (no command-line arguments needed)
# =====================================================================
DATA_PATH         = '/home2/zmy/projects/20240821_evcar/dataset/rvt_stream_data/'  # Parent directory containing train/val/test
MODEL_SCALE       = 'tiny'      # tiny (RVT-T, 4.4M) / small (RVT-S, 9.9M) / base (RVT-B, 18.5M)
NUM_CLASSES       = 4           # = max class_id in all your annotations + 1
GPUS              = '0'         # Single GPU: '0'; multi-GPU example: '[0,1]'
DIST_BACKEND      = 'nccl'      # Use 'nccl' on Linux; DDP is not actually used for single-GPU

BATCH_TRAIN       = 8           # Starting value for V100 32GB + tiny; see notes below
BATCH_EVAL        = 2
NUM_WORKERS_TRAIN = 4           # On Linux can be >0; mixed sampling requires >=2
NUM_WORKERS_EVAL  = 2
SAMPLING          = 'mixed'     # Use paper default 'mixed' on Linux; fall back to 'random' if issues occur

LEARNING_RATE     = 2e-4        # Scale as lr = 2e-4 * sqrt(batch/8) with batch size
MAX_STEPS         = 250000      # Don't use 400k for small datasets; stop early based on the val curve
VAL_CHECK_INTERVAL = 5000       # Validation interval in steps

RUN_NAME          = 'rvt_tiny_250k_s0'   # Name of this run (subdirectory name for logs/ckpts)
OUTPUT_DIR        = './rvt_logs'     # Root directory for CSV logs and checkpoints (no wandb)
RESUME_CKPT       = None             # Resume from a local .ckpt; None = train from scratch

SEED              = 0          # Random seed; set to None to disable fixing
DETERMINISTIC     = False       # True = cuDNN deterministic algorithms (more reproducible but slower)

# ---- Build Hydra overrides from the variables above, inject into sys.argv (equivalent to passing command-line args) ----
_OVERRIDES = [
    'model=rnndet',
    'dataset=gen4',
    f'+experiment/gen4={MODEL_SCALE}',
    f'dataset.path={DATA_PATH}',
    f'hardware.gpus={GPUS}',
    f'hardware.dist_backend={DIST_BACKEND}',
    f'batch_size.train={BATCH_TRAIN}',
    f'batch_size.eval={BATCH_EVAL}',
    f'hardware.num_workers.train={NUM_WORKERS_TRAIN}',
    f'hardware.num_workers.eval={NUM_WORKERS_EVAL}',
    f'dataset.train.sampling={SAMPLING}',
    f'training.learning_rate={LEARNING_RATE}',
    f'training.max_steps={MAX_STEPS}',
    f'validation.val_check_interval={VAL_CHECK_INTERVAL}',
    # wandb is disabled, but these fields are required by the config schema; placeholder values are fine
    'wandb.project_name=RVT',
    f'wandb.group_name={RUN_NAME}',
    f'reproduce.deterministic_flag={DETERMINISTIC}',
    # Disable wandb-specific high-dim visualization callbacks (log_images), otherwise they crash when wandb is unavailable
    'logging.train.high_dim.enable=False',
    'logging.validation.high_dim.enable=False',
]
if SEED is not None:
    _OVERRIDES.append(f'reproduce.seed_everything={SEED}')
os.environ['WANDB_MODE'] = 'disabled'   # Completely disable wandb (prevent any leftover calls from going online/erroring)
sys.argv = [sys.argv[0]] + _OVERRIDES + sys.argv[1:]
# =====================================================================

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning.loggers import CSVLogger

from callbacks.custom import get_ckpt_callback
from config.modifier import dynamically_modify_train_config
from data.utils.types import DatasetSamplingMode
from modules.utils.fetch import fetch_data_module, fetch_model_module


@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Override the hardcoded number of classes in modifier with NUM_CLASSES from the top of this file (no need to edit modifier.py)
    config.model.head.num_classes = NUM_CLASSES
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # Reproducibility
    # ---------------------
    dataset_train_sampling = config.dataset.train.sampling
    assert dataset_train_sampling in iter(DatasetSamplingMode)
    seed = config.reproduce.seed_everything
    if seed is not None:
        assert isinstance(seed, int)
        print(f'USING pl.seed_everything WITH seed={seed}')
        pl.seed_everything(seed=seed, workers=True)
        if dataset_train_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.MIXED):
            print('Note: under stream/mixed sampling, the shuffle of the streaming branch still has residual nondeterminism; '
                  'the seed has fixed the remaining random sources such as model initialization / data augmentation / random sampling / dataloader workers.')

    # ---------------------
    # DDP
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints (local CSV, no wandb)
    # ---------------------
    logger = CSVLogger(save_dir=OUTPUT_DIR, name=RUN_NAME)
    ckpt_path = RESUME_CKPT          # Resume training from a local .ckpt (full state restoration), or None to train from scratch

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    callbacks.append(get_ckpt_callback(config))   # Still save the best ckpt locally based on val/AP
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(ModelSummary(max_depth=2))
    # Removed (all wandb-specific, will error when wandb is unavailable):
    #   GradFlowLogCallback -> passes matplotlib figures to log_metrics
    #   DetectionVizCallback -> logger.log_images
    #   logger.watch(...)    -> wandb's gradient / computation graph monitoring

    # ---------------------
    # Training
    # ---------------------

    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch
    assert val_check_interval is None or check_val_every_n_epoch is None

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir=OUTPUT_DIR,
        devices=gpus,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        logger=logger,
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        strategy=strategy,
        sync_batchnorm=False if strategy is None else True,
        move_metrics_to_cpu=False,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
    )
    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)


if __name__ == '__main__':
    main()
