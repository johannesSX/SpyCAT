import argparse
import logging
import pathlib

import pytorch_lightning as pl  # 'L' renamed to 'pl' for clarity

from pl import run_train  # Assumes run_train is defined in pl.py

# Set up module-level logger
logger = logging.getLogger(__name__)

def str2bool(v):
    """Convert command-line string to boolean (for argparse)."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train VAE model on MRI data"
    )
    parser.add_argument("--data_dir", type=str, default="../results/", help="Path to dataset directory")
    parser.add_argument("--mni_mask_path", type=str, default="../data/masks/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii")
    parser.add_argument("--mni_grid_path", type=str, default="../data/grid/mni_grid.nii.gz")
    parser.add_argument("--seq", type=str, default='t2',  help="MRI sequence type (e.g., swi, t1)")
    parser.add_argument("--epochs", type=int, default=46, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker threads for data loading")
    parser.add_argument('--num_samples_per_epoch_and_node', type=int, default=10_000_000, help="Samples per epoch per node")
    parser.add_argument('--n_cuda', type=int, default=0, help="CUDA device index (default: 0, disables if no GPU)")
    return parser.parse_args()

def run_vae():
    """Main function to set up data and start training."""
    args = parse_args()
    args.data_dir = pathlib.Path(args.data_dir)  # Convert to Path object

    pl.seed_everything(42, workers=True)

    # Prepare DataModule (expects your custom MRIDataModule class)
    from build_dataset import MRIDataModule
    data_module = MRIDataModule(args)
    data_module.setup('fit')

    # Start training
    device = f'cuda:{args.n_cuda}' if args.n_cuda >= 0 else 'cpu'
    run_train(
        args,
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        device
    )

if __name__ == "__main__":
    run_vae()