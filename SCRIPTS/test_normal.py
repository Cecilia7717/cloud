from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
import sys, os
sys.path.append("/PyLandscape")
import fire
import numpy as np
import ignite
import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import get_transform, get_metrics, get_train_test_datasets, get_model
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
from PyLandscape.benchmarks.bit_flip import BitFlip

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


def initialize(config):
    model = get_model(config["model"], config["quant_config"])
    # Adapt model for distributed settings if configured
    model = idist.auto_model(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    optimizer = idist.auto_optim(optimizer)
    criterion = TrainingLoss(config["class_weights"]).to(idist.device())

    le = config["num_iters_per_epoch"]
    milestones_values = [
        (0, 0.0),
        (le * config["num_warmup_epochs"], config["learning_rate"]),
        (le * config["num_epochs"], 0.0),
    ]
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
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

def make_iterable(value: Any):
    if isinstance(value, list):
        return value
    return [value]

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

# ==== NEW: helper to robustly load model weights from various checkpoint formats ====
def load_model_from_checkpoint(model, ckpt_path, logger=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Case 1: Ignite-style
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        model.load_state_dict(state_dict, strict=False)
        if logger:
            logger.info(f"Loaded Ignite checkpoint from {ckpt_path}")

    # Case 2: Standard PyTorch checkpoint
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if logger:
            logger.info(f"Loaded PyTorch checkpoint from {ckpt_path}")

    # Case 3: Raw state_dict (your second one)
    elif all(isinstance(k, str) for k in ckpt.keys()):
        model.load_state_dict(ckpt, strict=False)
        if logger:
            logger.info(f"Loaded raw state_dict from {ckpt_path}")

    else:
        raise ValueError(f"Unrecognized checkpoint structure in {ckpt_path}")

    return model

# ==== NEW: evaluation worker ====
def evaluate_only(local_rank, config):
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name="ALCD_CLOUD_EVAL")
    log_basic_info(logger, config)

    # ---------- Data ----------
    train_loader, test_loader = get_dataflow(config)

    # ---------- Model & metrics ----------
    model = get_model(config["model"], config["quant_config"])
    model = idist.auto_model(model)
    model.to(device)

    if not config["resume_from"]:
        raise ValueError("run_eval requires --resume_from to point to a checkpoint file")
    load_model_from_checkpoint(model, config["resume_from"], logger)

    # get transforms & metrics (your get_metrics expects a criterion; for eval we can pass a no-op or reuse TrainingLoss)
    train_transform, test_transform = get_transform(
        mean=config["img_mean"], std=config["img_rescale"]
    )

    # If your get_metrics needs a criterion for loss tracking, reuse your TrainingLoss
    # (will be used only for metrics; model is in eval mode)
    criterion = TrainingLoss(config["class_weights"]).to(device)
    model.criterion = criterion
    class_count = 2
    metrics = get_metrics(class_count=class_count, criterion=criterion)
    
    if config['bit_flip']:
        bit_flip = BitFlip(model, test_loader, device)
        num_bits = make_iterable([1,5,10,20])
        strategies = ["fkeras", "random"]
        for strategy in strategies:
            perturbed_models = bit_flip.attack(num_bits, strategy)
            for perturbed_model, n_bits in zip(perturbed_models, num_bits):
                # evaluate the bit flipped models
                # ---------- Evaluator ----------
                evaluator = create_evaluator(perturbed_model, test_transform, metrics=metrics, config=config)
                print(f"evaluate: {strategy}_bit_flip_{n_bits}")
                # Optional: simple log hook
                @evaluator.on(Events.COMPLETED)
                def _log_final(engine):
                    logger.info(f"Evaluation completed on {len(test_loader)} batches.")
                    for k, v in engine.state.metrics.items():
                        logger.info(f"{k}: {v}")

                # ---------- Run ----------
                state = evaluator.run(test_loader)
                log_metrics(logger, 0, state.times["COMPLETED"], "Test", state.metrics)
    else:
        # ---------- Evaluator ----------
        evaluator = create_evaluator(model, test_transform, metrics=metrics, config=config)
        # Optional: simple log hook
        @evaluator.on(Events.COMPLETED)
        def _log_final(engine):
            logger.info(f"Evaluation completed on {len(test_loader)} batches.")
            for k, v in engine.state.metrics.items():
                logger.info(f"{k}: {v}")

        # ---------- Run ----------
        state = evaluator.run(test_loader)
        log_metrics(logger, 0, state.times["COMPLETED"], "Test", state.metrics)


# ==== NEW: Fire entrypoint for evaluation only ====
def run_eval(
    seed: int = 12345,
    data_path: str = "/pvc/",
    csv_paths: dict = {
        "train": "/pvc/train.csv",
        "test": "/pvc/valid.csv",
    },
    bit_flip: bool = False,
    # output_path not strictly needed for eval but kept for consistency/logging
    output_path: str = "./output-alcd-cloud-noisy/",
    input_size=(512, 512),
    img_mean: List[float] = [0.0, 0.0, 0.0],
    img_rescale: List[float] = [8657.0, 8657.0, 8657.0],
    model: str = "ags_tiny_unet_100k",
    quant_config: int = 8,  # or your string variant
    class_weights: List[float] = [0.1, 0.9],
    batch_size: int = 28,
    num_workers: int = 8,
    backend: Optional[str] = None,
    resume_from: Optional[str] = None,  # REQUIRED: path to checkpoint
    nproc_per_node: Optional[int] = None,
    with_amp: bool = False,  # respected by create_evaluator (autocast)
    **spawn_kwargs: Any,
):
    """
    Evaluate a saved checkpoint on the test set.
    Example:
      python your_script.py eval --resume_from /path/to/checkpoint.pt
    """
    # Catch all local parameters into config (mirror run())
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(evaluate_only, config)
    
if __name__ == "__main__":
    fire.Fire({"eval": run_eval})
