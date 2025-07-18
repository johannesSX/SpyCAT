import numpy as np
import torch
import torchio as tio
import pathlib
import tqdm
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model import VQVAE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def run_train(args, data_loader, val_data_loader, device):
    logger = SummaryWriter(log_dir=args.data_dir / 'lightning_logs' / f"vae_{args.seq}")

    vae_model = VQVAE().to(device)
    start_idx = 0

    optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.lr) # 0.0005
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    run_validation(args, start_idx, logger, vae_model, val_data_loader, device)
    for idx_epoch in tqdm.tqdm(range(start_idx, args.epochs), desc='TRAIN VAE - EPOCH'):
        if idx_epoch != 0 and idx_epoch % 1 == 0:
            path_to_save = pathlib.Path(logger.log_dir) / 'checkpoints' / f'val_after_epoch_{idx_epoch}.ckpt'
            pathlib.Path(path_to_save).parent.mkdir(parents=True, exist_ok=True)
            torch.save(vae_model.state_dict(), path_to_save)
            run_validation(args, idx_epoch, logger, vae_model, val_data_loader, device)

        vae_model.train()
        loss_rec_epoch, loss_KL_epoch, total_loss_epoch, loss_idx = 0, 0, 0, 0
        for idx_batch, in_data in enumerate(pbar := tqdm.tqdm(data_loader, desc='TRAIN VAE - BATCH', leave=False)):
            optimizer.zero_grad()
            in_data = in_data.to(device)
            y, _, vq_loss = vae_model(in_data)

            loss = vae_model.loss_function(y, in_data, vq_loss)

            # Measure loss
            loss_rec_batch = loss['Reconstruction_Loss']
            loss_KL_batch = loss['VQ_Loss']
            total_loss_batch = loss['loss']

            # Optimize
            loss['loss'].backward()
            optimizer.step()

            loss_rec_epoch += loss_rec_batch.item()
            loss_KL_epoch += loss_KL_batch.item()
            total_loss_epoch += total_loss_batch.item()
            loss_idx += 1

            pbar.set_description("rec-loss {:.4f} --- kl {:.4f} --- sum {:.4f}".format(loss_rec_batch.item(), loss_KL_batch.item(), total_loss_batch.item()))
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        logger.add_scalar('train/rec', loss_rec_epoch  / loss_idx, idx_epoch)
        logger.add_scalar('train/kl', loss_KL_epoch  / loss_idx, idx_epoch)
        logger.add_scalar('train/sum', total_loss_epoch  / loss_idx, idx_epoch)
        logger.add_scalar('train/lr', lr, idx_epoch)
        logger.flush()
    logger.close()


def run_validation(args, idx_epoch, logger, vae_model, data_loader, device):
    f_train = []
    f_gt = []
    f_entity = []
    vae_model.eval()
    loss_rec_epoch, loss_KL_epoch, total_loss_epoch, loss_idx = 0, 0, 0, 0
    for idx, _ in enumerate(tqdm.tqdm(data_loader.dataset.nonzero_all, desc='VAL VAE', leave=True)):
        data_loader.dataset.set_global_tile_idx(idx)
        for i, (in_data, idcs_nonzero, _, nonzeros, _, not_healthy, t_entity) in enumerate(data_loader):
            if nonzeros[0][0] != -1:
                batch_images = in_data.to(device)
                y, mu, vq_loss = vae_model(batch_images)

                # Measure loss
                loss = vae_model.loss_function(y, batch_images, vq_loss)
                loss_rec_batch = loss['Reconstruction_Loss']
                loss_KL_batch = loss['VQ_Loss']
                total_loss_batch = loss_rec_batch + loss_KL_batch

                loss_rec_epoch += loss_rec_batch.item()
                loss_KL_epoch += loss_KL_batch.item()
                total_loss_epoch += total_loss_batch.item()
                loss_idx += 1

                f_train.append(mu.flatten().cpu().detach().sigmoid().numpy())
                f_gt.append(not_healthy[0].item())
                f_entity.append(t_entity[0].item())

                subject = data_loader.dataset.subject_boxes[i]

                uid_name_1 = str(pathlib.Path(subject['seq']).name).replace('.nii.gz', '')
                uid_parent_1 = str(pathlib.Path(subject['seq']).parent.name)
                path_to_in_1 = logger.log_dir / f"{idx_epoch}_mris" / uid_parent_1 / f'{not_healthy[0].item()}_{idx}_{uid_name_1}_in.nii.gz'
                path_to_out_1 = logger.log_dir / f"{idx_epoch}_mris" / uid_parent_1 / f'{not_healthy[0].item()}_{idx}_{uid_name_1}_out.nii.gz'
                path_to_out_1.parent.mkdir(parents=True, exist_ok=True)

                tio.ScalarImage(tensor=in_data[0][0].cpu().detach().unsqueeze(0)).save(path_to_in_1)
                tio.ScalarImage(tensor=y[0][0].cpu().detach().unsqueeze(0)).save(path_to_out_1)

    logger.add_scalar('val/rec', loss_rec_epoch / loss_idx, idx_epoch)
    logger.add_scalar('val/kl', loss_KL_epoch / loss_idx, idx_epoch)
    logger.add_scalar('val/sum', total_loss_epoch / loss_idx, idx_epoch)

    f_train = np.asarray(f_train)
    f_entity = np.asarray(f_entity)
    clf_pca = PCA(n_components=2, random_state=42)
    x_pca = clf_pca.fit_transform(f_train)
    clf_umap = umap.UMAP(random_state=42)
    x_umap = clf_umap.fit_transform(f_train)

    for entity in np.unique(f_entity):
        if entity != -1:
            fig, ax = plt.subplots()
            ax.scatter(x_pca[:, 0][f_entity == -1], x_pca[:, 1][f_entity == -1], c='r')
            ax.scatter(x_pca[:, 0][f_entity == entity], x_pca[:, 1][f_entity == entity], c='b')
            path_to_save = logger.log_dir / f"pca" / f"{idx_epoch}_{entity}.png"
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(path_to_save))
            plt.close('all')

            fig, ax = plt.subplots()
            ax.scatter(x_umap[:, 0][f_entity == -1], x_umap[:, 1][f_entity == -1], c='r')
            ax.scatter(x_umap[:, 0][f_entity == entity], x_umap[:, 1][f_entity == entity], c='b')
            path_to_save = logger.log_dir / f"umap" / f"{idx_epoch}_{entity}.png"
            path_to_save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(path_to_save))
            plt.close('all')

    vae_model.train()