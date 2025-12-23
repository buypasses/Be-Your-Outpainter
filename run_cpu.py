"""Wrapper to force CPU for Be-Your-Outpainter"""
import os
import sys

# Monkey-patch torch to disable MPS before any imports
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

# Also patch accelerate's device detection
import accelerate.utils.environment
original_check = accelerate.utils.environment.check_cuda_p2p_ib_support
accelerate.utils.environment.check_cuda_p2p_ib_support = lambda: False

# Now run the training script
from scripts.train_outpaint import main
from omegaconf import OmegaConf

# Parse args like the original script
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--base_config", type=str, required=True)
args = parser.parse_args()

base_config = OmegaConf.load(args.base_config).outpaint
config = OmegaConf.load(args.config)

for key in config.keys():
    print(f"Running: {key}")
    main(base_config, config[key])
