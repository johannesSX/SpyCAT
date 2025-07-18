import argparse
import logging
import pathlib
import torch
import torchio as tio
import pytorch_lightning as L
from pl import run_train, run_validation_large, run_anno_map
from build_dataset import MRIDataModule
from dataset import ValDataset_2

logger = logging.getLogger(__name__)

# Helper for boolean argparse arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Performer model on MRI data"
    )
    parser.add_argument("--data_dir", type=str, default="../results/")
    parser.add_argument("--vae_ckpt", type=int, default=19)
    parser.add_argument("--val_ckpt", type=int, default=19)
    parser.add_argument("--seq", type=str, default='t2')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--mni_mask_path", type=str, default="../data/masks/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii")
    parser.add_argument("--mni_grid_path", type=str, default="../data/grid/mni_grid.nii.gz")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--num_samples_per_epoch_and_node', type=int, default=1000000)
    parser.add_argument('--with_cuda', default=False, action='store_true', help='use CPU if no GPU available')
    parser.add_argument('--num_emb', type=int, default=8192)
    parser.add_argument('--mode', type=str, default='CREATE_ANNO_MAP')  # Either FIT or VALANDTEST
    return parser.parse_args()

def run_perf():
    args = parse_args()
    args.data_dir = pathlib.Path(args.data_dir)

    # Set random seed for reproducibility
    L.seed_everything(42, workers=True)

    # Choose operation mode
    if args.mode == 'FIT':
        dm = MRIDataModule(args)
        dm.setup('fit')
        run_train(args, dm.train_dataloader(), dm.val_dataloader(), dm.tio_mask_full, 'cuda:0')
    elif args.mode == 'VALANDTEST':
        dm = MRIDataModule(args)
        dm.setup('validate')
        run_validation_large(args, dm.val_dataloader(), dm.tio_mask_full, 'cuda:0')
    elif args.mode == 'CREATE_ANNO_MAP':
        # CREATE
        dataset_lst = [{'seq': '../data/sample/BraTS2021_00000_t2.nii.gz'}]
        dataset_val = ValDataset_2(
            dataset_lst,
            tio.ScalarImage(args.mni_grid_path),
            tio.ScalarImage(args.mni_mask_path),
            patch_size=16
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            shuffle=False,  # False if self.args.dist else True,
            pin_memory=True,
            batch_size=args.batch_size_eval,
            num_workers=args.num_workers,  # muss null sein
        )
        run_anno_map(args, val_dataloader, tio.ScalarImage(args.mni_mask_path), 'cuda:0')

if __name__ == "__main__":
    run_perf()