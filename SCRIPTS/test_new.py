from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import fire
import numpy as np
import ignite
import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_transform, get_metrics, get_train_test_datasets, get_test_datasets, get_model
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    PiecewiseLinear,
    global_step_from_engine,
)
from ignite.utils import manual_seed, setup_logger
from torch import nn
from torch.amp import autocast
from torch.amp import GradScaler
from utils.quant_utils.save_handler_qonnx import CheckpointQONNX, DiskSaverQONNX
from utils.loss import TrainingLoss
from torchvision import tv_tensors


def test_only(local_rank, config):
    rank = idist.get_rank()
    device = idist.device()

    logger = setup_logger(name="ALCD_CLOUD_TEST")
    log_basic_info(logger, config)

    # Setup dataflow (we only need test_loader here)
    _, test_loader = get_dataflow(config)

    # Initialize model, optimizer, criterion, scheduler (same as training)
    model, optimizer, criterion, lr_scheduler = initialize(config)

    # Load checkpoint if resume_from is provided
    if config.get("resume_from", None):
        checkpoint_fp = config["resume_from"]
        logger.info(f"Resuming from checkpoint: {checkpoint_fp}")
        checkpoint = torch.load(checkpoint_fp, map_location=device)
    else:
        raise ValueError("No checkpoint provided in config['resume_from'].")

    # Get test transform
    _, test_transform = get_transform(
        mean=config["img_mean"], std=config["img_rescale"]
    )

    # Define metrics and evaluator
    class_count = 2
    metrics = get_metrics(class_count=class_count, criterion=criterion)
    evaluator = create_evaluator(checkpoint, test_transform, metrics=metrics, config=config)

    # Run evaluation on test set
    state = evaluator.run(test_loader)

    # Log and return results
    log_metrics(logger, 0, state.times["COMPLETED"], "Test", state.metrics)

    if rank == 0:
        logger.info("=== Final Test Results ===")
        for name, value in state.metrics.items():
            logger.info(f"{name}: {value}")

    return state.metrics


def run(
    seed: int = 12345,
    data_path: str = "pvc/",
    csv_paths: dict = {
        "train": "/pvc/train.csv",
        "test": "/pvc/valid.csv",
    },
    output_path: str = "./output-alcd-cloud/",
    input_size=(512, 512),
    img_mean: List[float] = [0.0, 0, 0.0],
    img_rescale: List[float] = [8657.0, 8657.0, 8657.0],
    model: str = "ags_tiny_unet_100k",
    quant_config: str = None,
    class_weights: List[float] = [0.1, 0.9],
    batch_size: int = 28,
    weight_decay: float = 2e-4,
    num_workers: int = 8,
    num_epochs: int = 100, #1000
    learning_rate: float = 1e-3,
    num_warmup_epochs: int = 5,
    validate_every: int = 3,
    checkpoint_every: int = 1000,
    backend: Optional[str] = None,
    resume_from: Optional[str] = None,
    log_every_iters: int = 5,
    nproc_per_node: Optional[int] = None,
    stop_iteration: Optional[int] = None,
    with_amp: bool = False,
    **spawn_kwargs: Any,
):
    """Main entry to train an model on CIFAR10 dataset.

    Args:
        seed (int): random state seed to set. Default, 12345.
        data_path (str): input dataset path. Default, "DATA".
        output_path (str): output path. Default, "./output-alcd-cloud/".
        input_size (Tuple[int,int]): model input shape = (256, 256),
        img_mean: List[float]: Mean to be substracted to each channel,
        img_rescale: List[float]: std used to rescale images,
        model (str): model name (from torchvision) to setup model to train. Default, "ags_tiny_unet_100k".
        quant_config: str: quantization method to apply. Default, None. Possible values, "8bit_fix", None.
        batch_size (int): total batch size. Default, 16.
        weight_decay (float): weight decay. Default, 1e-4.
        num_workers (int): number of workers in the data loader. Default, 8.
        num_epochs (int): number of epochs to train the model. Default, 1000.
        learning_rate (float): peak of piecewise linear learning rate scheduler. Default, 1e-4.
        num_warmup_epochs (int): number of warm-up epochs before learning rate decay. Default, 5.
        validate_every (int): run model's validation every ``validate_every`` epochs. Default, 3.
        checkpoint_every (int): store training checkpoint every ``checkpoint_every`` iterations. Default, 1000.
        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl", "xla-tpu",
            "gloo" etc. Default, None.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
            when main python process is spawning training as child processes.
        resume_from (str, optional): path to checkpoint to use to resume the training from. Default, None.
        log_every_iters (int): argument to log batch loss every ``log_every_iters`` iterations.
            It can be 0 to disable it. Default, 5.
        stop_iteration (int, optional): iteration to stop the training. Can be used to check resume from checkpoint.
        with_amp (bool): if True, enables native automatic mixed precision. Default, False.
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes

    """
    # check to see if the num_epochs is greater than or equal to num_warmup_epochs
    if num_warmup_epochs >= num_epochs:
        raise ValueError(
            "num_epochs cannot be less than or equal to num_warmup_epochs, please increase num_epochs or decrease "
            "num_warmup_epochs"
        )

    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(test_only, config)

def get_dataflow(config):
    # - Get train/test datasets
    with idist.one_rank_first(local=True):
        train_dataset, test_dataset = get_train_test_datasets(
            config["data_path"], csv_paths=config["csv_paths"]
        )

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    test_loader = idist.auto_dataloader(
        test_dataset,
        batch_size=2 * config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, test_loader

def get_test_dataflow(config):
    # - Get train/test datasets
    with idist.one_rank_first(local=True):
        test_dataset = get_test_datasets(
            config["data_path"], csv_paths=config["csv_paths"]
        )

    test_loader = idist.auto_dataloader(
        test_dataset,
        batch_size=2 * config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        pin_memory=True,
    )
    return test_loader


def initialize(config, train_loader=None):
    # Model
    model = get_model(config["model"], quant_config=config["quant_config"])
    model = idist.auto_model(model)

    # Criterion
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(config["class_weights"]).to(idist.device()))

    optimizer, lr_scheduler = None, None
    if train_loader is not None:  # training mode
        optimizer = torch.optim.Adam(model.parameters())
        steps_per_epoch = len(train_loader)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"] * steps_per_epoch
        )

    return model, optimizer, criterion, lr_scheduler


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"Epoch[{epoch}] - Evaluation time (seconds): {elapsed:.3f}\n - {tag} metrics:\n {metrics_output}"
    )


def log_basic_info(logger, config):
    logger.info(f"Train {config['model']} on CIFAR10")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(
            f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
        )
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def create_trainer(
    model, transform, optimizer, criterion, lr_scheduler, train_sampler, config, logger
):
    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    with_amp = config["with_amp"]
    scaler = GradScaler('cuda',enabled=with_amp)

    def train_step(engine, batch):
        x, y = batch[0], batch[1]

        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        new_images = []
        new_targets = []
        for i in range(len(x)):
            image = tv_tensors.Image(x[i])
            mask = tv_tensors.Mask(y[i])
            image, mask = transform((image, mask))
            new_images.append(image)
            new_targets.append(mask)
        x, y = torch.stack(new_images), torch.stack(new_targets)

        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        model.train()

        with autocast("cuda", enabled=with_amp):
            y_pred = model(x)

            loss = criterion(y_pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return {
            "batch loss": loss.item(),
        }

    trainer = Engine(train_step)
    trainer.logger = logger

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    metric_names = [
        "batch loss",
    ]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=metric_names if config["log_every_iters"] > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    resume_from = config["resume_from"]
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert (
            checkpoint_fp.exists()
        ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        to_save = {"trainer": trainer, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def create_evaluator(model, transform, metrics, config, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine: Engine, batch):
        model.eval()
        x, y = batch[0], batch[1]

        new_images = []
        new_targets = []
        for i in range(len(x)):
            image = tv_tensors.Image(x[i])
            mask = tv_tensors.Mask(y[i])
            image, mask = transform((image, mask))
            new_images.append(image)
            new_targets.append(mask)
        x, y = torch.stack(new_images), torch.stack(new_targets)

        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        with autocast("cuda", enabled=with_amp):
            output = model(x)
        return output, y

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def get_save_handler(config):
    return DiskSaver(config["output_path"], require_empty=False)


def getSaveHandlerQONNX(config: Dict[str, Any]):
    return DiskSaverQONNX(config["output_path"], require_empty=False)


if __name__ == "__main__":
    fire.Fire({"run": run})
