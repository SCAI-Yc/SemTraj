"""
A minimal training script for SemTraj using PyTorch DDP.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torch.utils.data import TensorDataset, DataLoader

from model.model import SemTraj
from model.diffusion import create_diffusion
from length_generator import LengthVAE


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    """
    Trains a new SemTraj model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."   

    seed = args.seed
    device = 0
    print("seed >>>", seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = "test_model"
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    model = SemTraj(
        in_channels=2,
        hidden_size=128,
        depth=28,
        head_nums=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        traj_length=200,
        learn_sigma=True
    ).to(device)

    # create length generator
    lg_model = LengthVAE(cond_dim=3).to(device)

    # Note that parameter initialization is done within the SemTraj constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    lg_ema = deepcopy(lg_model).to(device)
    requires_grad(lg_ema, False)


    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"SemTraj Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    lg_opt = torch.optim.AdamW(lg_model.parameters(), lr=1e-4, weight_decay=0)

    source_data_path = args.data_path

    trajs = np.load(f"{source_data_path}/trajs_xian.npy")
    heads = np.load(f"{source_data_path}/heads_xian.npy")

    trajs = np.swapaxes(trajs, 1, 2)
    trajs = torch.from_numpy(trajs).float()
    heads = torch.from_numpy(heads).float()
    dataset = TensorDataset(trajs, heads)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=8)
    
    logger.info(f"Dataset contains {len(dataset):,} trajectories ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    update_ema(lg_ema, lg_model, decay=0)  # Ensure EMA is initialized with synced weights
    lg_model.train()  # important! This enables embedding dropout for classifier-free guidance
    lg_ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for _, (trainx, head) in enumerate(dataloader):
            x, y = trainx, head
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            lg_opt.zero_grad()
            L_pred, mu, logvar = lg_model(y)
            L_true = y[..., -1] if y.shape[-1] > 1 else y.squeeze(-1)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            recon_loss = torch.nn.functional.mse_loss(L_pred, L_true)
            lg_loss = recon_loss + kld
            lg_loss.backward()
            lg_opt.step()
            update_ema(lg_ema, lg_model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SemTraj checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                # if rank == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        break

    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='./data')
    parser.add_argument("--results-dir", type=str, default="SemTraj_results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    args = parser.parse_args()
    main(args)