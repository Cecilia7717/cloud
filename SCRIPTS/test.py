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


def test_only(
    seed: int = 12345,
    data_path: str = "/pvc",
    csv_paths: str = "/pvc/valid.csv",
    output_path: str = "./output-test/",
    input_size=(512, 512),
    img_mean: List[float] = [0.0, 0.0, 0.0],
    img_rescale: List[float] = [8657.0, 8657.0, 8657.0],
    model: str = "ags_tiny_unet_100k",
    quant_config: int = 8,
    class_weights: List[float] = [0.1, 0.9],
    learning_rate: float = 1e-3,
    batch_size: int = 28,
    weight_decay: float = 2e-4,
    num_workers: int = 8,
    num_epochs: int = 100,
    num_warmup_epochs: int = 5,
    validate_every: int = 3,
    checkpoint_every: int = 1000,
    resume_from: Optional[str] = None,
    with_amp: bool = False,
    **spawn_kwargs: Any,
):
    """Run evaluation only on the test set."""

    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]
    with idist.Parallel(backend="nccl" if torch.cuda.is_available() else "gloo") as parallel:
        def _eval_only(local_rank, config):
            device = idist.device()
            manual_seed(config["seed"] + idist.get_rank())

            # Dataflow (test only)
            test_loader = get_test_dataflow(config)
            # Model + criterion
            model, _, criterion, _ = initialize(config, train_loader=None)

            # Load checkpoint
            if config["resume_from"] is None:
                raise ValueError("--resume_from must be specified for test_only")
            checkpoint_fp = Path(resume_from)
            ckpt = torch.load(config["resume_from"], map_location="cpu")
            if "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                # Plain state dict (your case for best_model_*.pt)
                state_dict = ckpt
            # Evaluator
            model.load_state_dict(state_dict, strict=False)
            metrics = get_metrics(class_count=2, criterion=criterion)
            test_transform = get_transform(
                mean=config["img_mean"], std=config["img_rescale"]
            )[1]  # second element is test_transform
            evaluator = create_evaluator(model, test_transform, metrics, config)

            # Run evaluation
            state = evaluator.run(test_loader)
            print("=== Test Metrics ===")
            for k, v in state.metrics.items():
                print(f"{k}: {v}")

        parallel.run(_eval_only, config)


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
    fire.Fire({"test_only": test_only})
