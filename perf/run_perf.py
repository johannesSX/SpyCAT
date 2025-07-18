import argparse
import logging
import pathlib
import pytorch_lightning as L
from pl import run_train, run_validation_large

logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_perf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../results/")
    parser.add_argument("--ext", type=str2bool, default=False)
    parser.add_argument("--vae_ckpt", type=int, default=19)
    parser.add_argument("--val_ckpt", type=int, default=19)
    parser.add_argument("--seq", type=str, default='t2')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--mni_mask_path", type=str, default="../data/masks/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii")
    parser.add_argument("--mni_grid_path", type=str, default="../data/grid/mni_grid.nii.gz")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--num_samples_per_epoch_and_node', default=1000000, type=int) # 1000000
    parser.add_argument('--with_cuda', default=False, action='store_true', help='use CPU in case there\'s no GPU support')
    parser.add_argument('--num_emb', type=int, default=8192)
    parser.add_argument('--mode', type=str, default='VALANDTEST') # REST3 oder FIT
    args, _ = parser.parse_known_args()
    args.data_dir = pathlib.Path(args.data_dir)

    L.seed_everything(42, workers=True)

    from build_dataset import MRIDataModule
    dm = MRIDataModule(args)
    if args.mode == 'FIT':
        dm.setup('fit')
        run_train(args, dm.train_dataloader(), dm.val_dataloader(), dm.tio_mask_full, 'cuda:0') # run_validation(dm.val_dataloader())
    elif args.mode == 'VALANDTEST':
        dm.setup('validate')
        run_validation_large(args, dm.val_dataloader(),  dm.tio_mask_full, 'cuda:0')


if __name__ == "__main__":
    run_perf()
