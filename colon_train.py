import random
import time
import datetime
import numpy as np
import argparse
import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from pathlib import Path
from tqdm import tqdm

from timm.models import create_model
from timm.utils import NativeScaler
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy

from engine import train_one_epoch, evaluate
from datasets.colon_dataset import ColonCancerDataset
from utils import save
import models


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Performer training and evaluation script on Colon data",
        add_help=False)
    parser.add_argment("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=300, type=int)

    # Model parameters
    parser.add_argument("--model",
                        default="performer_tiny_patch25_500",
                        type=str,
                        metavar="MODEL",
                        help="Name of model to train")
    parser.add_argument("input_size",
                        default=500,
                        type=int,
                        help="images input size")

    parser.add_argument("--drop",
                        type=float,
                        default=0.0,
                        metavar="PCT",
                        help="Dropout rate (default: 0.)")
    parser.add_argument("--drop_path",
                        type=float,
                        default=0.1,
                        metavar="PCT",
                        help="Drop path rate (default: 0.1)")

    parser.add_argument("--model_ema",
                        action="store_true")
    parser.add_argument("--no_model_ema",
                        action="store_false",
                        dest="model_ema")
    parser.set_defaults(model_ema=True)
    parser.add_argument("--model_ema_decay",
                        type=float,
                        default=0.99996,
                        help='')

    # Optimiser parameters
    parser.add_argument("--opt",
                        default="adamw",
                        type=str,
                        meatavar="OPTIMISER",
                        help="Optimiser (default: 'adamw'")
    parser.add_argument("--opt_eps",
                        default=1e-8,
                        type=float,
                        meatvar="EPSILON",
                        help="Optimiser Epsilon (default: 1e-8)")
    parser.add_argument("--opt_betas",
                        default=None,
                        type=float,
                        nargs="+",
                        meatvar="BETA",
                        help="Optimiser Betas (defaults: None, use opt default)")
    parser.add_argument("--clip_grad",
                        type=float,
                        default=None,
                        metavar="NORM",
                        help="Clip gradient norm (default: None, no clipping)")
    parser.add_argument("--momentum",
                        type=float,
                        default=0.9,
                        meatvar="M",
                        help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.05,
                        help="Weight decay (default: 0.05)")

    # Learning rate schuedule parameters
    parser.add_argument("--sched",
                        default="cosine",
                        type=str,
                        metavar="SCHEDULER",
                        help="LR scheduler (default: 'cosine'")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-4,
                        metavar="LR",
                        help="learning rate (default: 5e-4)")
    parser.add_argument("--lr_noise",
                        type=float,
                        nargs="+",
                        default=None,
                        metavar="pct, pct",
                        help="learning rate noise on/off epoch percentage")
    parser.add_argument("--lr_noise_pct",
                        type=float,
                        default=0.67,
                        metavar="PERCENT",
                        help="learning rate noise limit percent (default: 0.67)")
    parser.add_argument("--lr_noise_std",
                        type=float,
                        default=1.0,
                        metavar="STDDEV",
                        help="learning rate noise std-dev (default: 1.0)")
    parser.add_argument("--warmup_lr",
                        type=float,
                        default=1e-6,
                        metavar="LR",
                        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)")
    parser.add_argument("--min_lr",
                        type=float,
                        default=1e-5,
                        metavar="LR",
                        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)")

    parser.add_argument("--decay_epochs",
                        type=float,
                        default=30,
                        metavar="N",
                        help="epoch interval to decay LR")
    parser.add_argument("--warmup_epochs",
                        type=int,
                        default=5,
                        metavar="N",
                        help="epochs to warmup LR, if scheduler supports")
    parser.add_argument("--cooldown_epochs",
                        type=int,
                        default=10,
                        metavar="N",
                        help="epochs to cooldown LR at min_lr, after cyclic schedule ends")
    parser.add_argument("--patience_epochs",
                        type=int,
                        default=10,
                        metavar="N",
                        help="patience epochs for Plateau LR scheduler (default: 10")
    parser.add_argument("--deacy_rate",
                        "--dr",
                        type=float,
                        default=0.1,
                        meatavar="RATE",
                        help="LR decay rate (default: 0.1)")

    # Dataset parameters
    parser.add_argument("--data_path",
                        default='/',
                        type=str,
                        help="dataset path")
    parser.add_argument("--output_dir",
                        default="model_checkpoints/colon/",
                        help="path where to save, empty for no saving")
    parser.add_argument("--device",
                        default="cuda:0",
                        help="device to use for training/testing")
    parser.add_argument("--seed",
                        default=0,
                        type=int)
    parser.add_argument("--start_epoch",
                        default=0,
                        type=int,
                        metavar="N",
                        help="start epoch")
    parser.add_argument("--eval",
                        action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--num_workers",
                        default=10,
                        type=int,
                        help="Num workers for data loading")
    parser.add_argument("--pin_mem",
                        action="store_true",
                        help="Pin CPU memory in Dataloader for more efficient transfer to GPU (hopefully)")
    parser.add_argument("--no_pin_mem",
                        action="store_false",
                        dest="pin_mem",
                        help='')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):

    print(args)

    device = torch.device(args.device)

    # fix seed for reproducability
    print("Setting random seed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    data_directory = args.data_path
    print("Loading data")
    train_dataset = ColonCancerDataset(data_directory,
                                       train=True,
                                       seed=args.seed)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=args.pin_mem,
                              drop_last=True)

    val_dataset = ColonCancerDataset(data_directory,
                                     train=False,
                                     seed=args.manual_seed)
    val_loader = DataLoader(val_dataset,
                            batch_size=int(1.5 * args.batch_size),
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_mem,
                            drop_last=False)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=2,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    model.to(device)

    # model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model__ema_decay,
    #         device='cpu' if args.model_ema_force_cpu else '',
    #         resume=''
    #     )

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    linear_scaled_lr = args.lr * args.batch_size / 512.0
    args.lr = linear_scaled_lr
    optimiser = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimiser)

    criterion = LabelSmoothingCrossEntropy()

    output_dir = Path(args.output_dir)

    wandb.watch(model, criterion, log='all', log_freq=10)

    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        train_loss, train_metrics = train_one_epoch(model,
                                                    criterion,
                                                    train_loader,
                                                    optimiser,
                                                    device,
                                                    set_training_mode=args.finetune == '')

        lr_scheduler.step(epoch)
        # TODO add in resuming training

        val_loss, val_metrics = evaluate(
            val_loader,
            model,
            device)

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                save({
                    "model": model_without_ddp.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "scaler": loss_scaler.state_dict(),
                    "args": args,
                }, checkpoint_path)

        wandb.log({
            "epoch": epoch,
            "train loss": train_loss,
            "val loss": val_loss,
            "train acc": train_metrics["accuracy"],
            "train f1": train_metrics["f1 score"],
            "train prec": train_metrics["precision"],
            "train recall": train_metrics["recall"],
            "val acc": val_metrics["accuracy"],
            "val f1": val_metrics["f1 score"],
            "val prec": val_metrics["precision"],
            "val recall": val_metrics["recall"]
        })

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Comp Pathology Transformer(s) training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="vision-transformer-experiments",
        entity="r_j",
        tags=["COLON", "performer"],
        group="early tests",
        config=args,
        reinit=True
    )
    wandb.config.update(args)
    main(args)
