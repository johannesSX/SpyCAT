import copy
import numpy as np
import torch
import torchio as tio
import pathlib
import torchmetrics as tom
import tqdm
from scipy.ndimage import gaussian_filter
from torch.utils.tensorboard import SummaryWriter
import raster_geometry as rg
from nilearn import plotting as nilearn_plotting
import json
import scipy as sc


def run_train(args, train_data_loader, pred_data_loader, tio_mask_full, device):
    logger = SummaryWriter(log_dir=args.data_dir / 'lightning_logs' / f"perf_{args.seq}")

    from model import Performer
    performer = Performer(
        args,
        sum(tio_mask_full.data.shape[1:])
    ).to(device)

    optim = torch.optim.Adam(performer.parameters(), lr=1e-4)

    # run_validation_short(args, 0, performer, pred_data_loader, tio_mask_full, device)
    for idx_epoch in tqdm.tqdm(range(args.epochs), desc='TRAIN PERF - EPOCH'):
        if idx_epoch != 0 and idx_epoch % 1 == 0:
            path_to_save = pathlib.Path(logger.log_dir) / 'checkpoints' / f'val_after_epoch_{idx_epoch}.ckpt'
            pathlib.Path(path_to_save).parent.mkdir(parents=True, exist_ok=True)
            torch.save(performer.state_dict(), path_to_save)
            run_validation_short(args, idx_epoch, performer, pred_data_loader, tio_mask_full, device)

        loss_per_epoch = []
        performer.train()
        for idx_batch, (in_data, in_neighbors, idcs_nonzero_perf) in enumerate(pbar := tqdm.tqdm(train_data_loader, desc='TRAIN PERF - BATCH', leave=False)):
            in_data = in_data.to(device)
            in_neighbors = in_neighbors.to(device)

            idcs_nonzero_perf = args.num_emb + idcs_nonzero_perf.to(device) # 4096
            loss = performer(in_data, in_neighbors, idcs_nonzero_perf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(performer.parameters(), 0.5)
            optim.step()
            optim.zero_grad()
            pbar.set_description("loss {:.4f}".format(loss.item()))
            loss_per_epoch.append(loss.detach().cpu())
        logger.add_scalar('train/loss', torch.hstack(loss_per_epoch).mean(), idx_epoch)
        logger.flush()
    logger.close()


def run_validation_short(args, idx_epoch, performer, val_data_loader, tio_mask_full, device):
    performer.eval()

    for idx_subject, data_dict in enumerate(tqdm.tqdm(val_data_loader.dataset.subject_boxes, desc='NO REST - BATCH', total=len(val_data_loader.dataset.subject_boxes))):
        tio_img = tio.ScalarImage(data_dict['seq'])
        tio_gen = tio.ScalarImage(data_dict['seq'])
        tio_gen.data = torch.zeros_like(tio_gen.data)

        val_data_loader.dataset.set_global_subject_idx(idx_subject)
        for in_data, in_neighbors, idcs_nonzero_perf, nonzero in tqdm.tqdm(val_data_loader, leave=False):
            in_neighbors = in_neighbors.to(device)
            idcs_nonzero_perf = args.num_emb + idcs_nonzero_perf.to(device)
            out_data_fake = performer.forward_pred(in_neighbors, idcs_nonzero_perf)

            for (_, x_idx, y_idx, z_idx), _out_data_fake in zip(nonzero, out_data_fake):
                tio_gen.data[:, x_idx - 8: x_idx + 8, y_idx - 8: y_idx + 8, z_idx - 8: z_idx + 8] = _out_data_fake.detach().cpu() # _out_data_fake.detach().cpu()
                #tio_gen.data[:, x_idx - 2: x_idx + 2, y_idx - 2: y_idx + 2, z_idx - 2: z_idx + 2] = _out_data_fake[:, 6: 10, 6: 10, 6: 10].detach().cpu()  # _out_data_fake.detach().cpu()

        uid_name = str(pathlib.Path(data_dict['seq']).name).replace('.nii.gz', '')
        uid_parent = str(pathlib.Path(data_dict['seq']).parent.name)
        path_to_save_g = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"preds" / f"{idx_epoch}" / uid_parent / f'g_{uid_name}.nii.gz'
        path_to_save_o = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"preds" / f"{idx_epoch}" / uid_parent / f'o_{uid_name}.nii.gz'

        path_to_save_g.parent.mkdir(parents=True, exist_ok=True)
        path_to_save_o.parent.mkdir(parents=True, exist_ok=True)

        tio_gen.save(path_to_save_g)
        tio_img.save(path_to_save_o)
    performer.train()


def run_anno_map(args, val_data_loader, tio_mask_full, device):
    from model import Performer
    performer = Performer(args, sum(tio_mask_full.data.shape[1:])).to(device)
    performer.load_state_dict(torch.load(f"../results/lightning_logs/perf_{args.seq}/checkpoints/val_after_epoch_{args.val_ckpt}.ckpt"))
    performer.eval()

    for idx_subject, data_dict in enumerate(tqdm.tqdm(val_data_loader.dataset.subject_boxes, desc='NO REST - BATCH', total=len(val_data_loader.dataset.subject_boxes))):
        tio_img = tio.ScalarImage(data_dict['seq'])
        tio_anno = copy.deepcopy(tio_img)
        tio_anno.data = torch.zeros_like(tio_anno.data)

        val_data_loader.dataset.set_global_subject_idx(idx_subject)
        for in_data, in_neighbors, idcs_nonzero_perf, nonzero in tqdm.tqdm(val_data_loader, leave=False):
            in_data = in_data.to(device)
            in_neighbors = in_neighbors.to(device)
            idcs_nonzero_perf = args.num_emb + idcs_nonzero_perf.to(device)

            out_data_org, out_data_fake, _ = performer.forward_eval(in_data, in_neighbors, idcs_nonzero_perf, custom_thrs=None)
            out_data_diff = out_data_fake - out_data_org

            for (_, x_idx, y_idx, z_idx), _out_data_diff in zip(nonzero, out_data_diff):
                assert tio_anno.data[:, x_idx - 8: x_idx + 8, y_idx - 8: y_idx + 8, z_idx - 8: z_idx + 8].max() == 0
                tio_anno.data[:, x_idx - 8: x_idx + 8, y_idx - 8: y_idx + 8, z_idx - 8: z_idx + 8] = _out_data_diff.detach().cpu()  # _out_data_diff[:, 4: 12, 4: 12, 4: 12].detach().cpu()

        uid_name = str(pathlib.Path(data_dict['seq']).name).replace('.nii.gz', '')
        uid_parent = str(pathlib.Path(data_dict['seq']).parent.name)
        path_to_save_a = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"anno_map" / uid_parent / f'a_{uid_name}.nii.gz'
        path_to_save_a.parent.mkdir(parents=True, exist_ok=True)
        tio_anno.data = torch.from_numpy(sc.ndimage.gaussian_filter(tio_anno.data.numpy(), sigma=1)) # 1
        tio_anno.save(path_to_save_a)


def run_validation_large(args, val_data_loader, tio_mask_full, device, r=9, topk=10):
    tio_mask_small = tio.ScalarImage(args.mni_mask_path)
    tio_mask_small.data = torch.from_numpy(sc.ndimage.binary_erosion(input=tio_mask_small.data.numpy(), structure=np.ones((1, 3, 3, 3)), iterations=3)).int() # vorher 5

    from model import Performer
    performer = Performer(args,  sum(tio_mask_full.data.shape[1:])).to(device)
    performer.load_state_dict(torch.load( f"../results/lightning_logs/perf_{args.seq}/checkpoints/val_after_epoch_{args.val_ckpt}.ckpt"))
    performer.eval()

    score_img_level = []
    for idx_subject, data_dict in enumerate(tqdm.tqdm(val_data_loader.dataset.subject_boxes, desc='VAL - BATCH', total=len(val_data_loader.dataset.subject_boxes))):
        tio_img = tio.ScalarImage(data_dict['seq'])
        if 'seg' in list(data_dict.keys()) and data_dict['seg'] is not None:
            tio_seg = tio.ScalarImage(data_dict['seg'])
            tio_seg.data[ tio_seg.data > 1] = 1
        else:
            tio_seg = None
        tio_anno = copy.deepcopy(tio_img)
        tio_anno.data = torch.zeros_like(tio_anno.data)
        tio_mask = copy.deepcopy(tio_img)
        tio_mask.data = torch.zeros_like(tio_mask.data)
        tio_topkand = copy.deepcopy(tio_img)
        tio_topkand.data = torch.zeros_like(tio_topkand.data)
        tio_org = copy.deepcopy(tio_img)
        tio_org.data = torch.zeros_like(tio_org.data)
        tio_minmask = copy.deepcopy(tio_img)
        tio_minmask.data = torch.zeros_like(tio_minmask.data)

        val_data_loader.dataset.set_global_subject_idx(idx_subject)
        for in_data, in_neighbors, idcs_nonzero_perf, nonzero in tqdm.tqdm(val_data_loader, leave=False):
            in_data = in_data.to(device)
            in_neighbors = in_neighbors.to(device)
            idcs_nonzero_perf = args.num_emb + idcs_nonzero_perf.to(device)

            out_data_org, out_data_fake, _ = performer.forward_eval(in_data, in_neighbors, idcs_nonzero_perf, custom_thrs=None)
            out_data_diff = out_data_fake - out_data_org

            for (_, x_idx, y_idx, z_idx), _out_data_diff in zip(nonzero, out_data_diff):
                assert tio_anno.data[:, x_idx - 8: x_idx + 8, y_idx - 8: y_idx + 8, z_idx - 8: z_idx + 8].max() == 0
                tio_anno.data[:, x_idx - 8: x_idx + 8, y_idx - 8: y_idx + 8, z_idx - 8: z_idx + 8] = _out_data_diff.detach().cpu() # _out_data_diff[:, 4: 12, 4: 12, 4: 12].detach().cpu()

        uid_name = str(pathlib.Path(data_dict['seq']).name).replace('.nii.gz', '')
        uid_parent = str(pathlib.Path(data_dict['seq']).parent.name)
        path_to_save_a = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / uid_parent / f'a_{uid_name}.nii.gz'
        path_to_save_i = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / uid_parent / f'i_{uid_name}.nii.gz'
        path_to_save_o = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / uid_parent / f'o_{uid_name}.nii.gz'
        path_to_save_k = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / uid_parent / f'k_{uid_name}.nii.gz'
        path_to_save_m = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / uid_parent / f'm_{uid_name}.nii.gz'
        path_to_save_ga = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / uid_parent / f'ga_{uid_name}.png'
        path_to_save_go = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / uid_parent / f'go_{uid_name}.png'
        path_to_save_gk = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / uid_parent / f'gk_{uid_name}.png'
        path_to_save_sa = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / uid_parent / f'sa_{uid_name}.png'
        path_to_save_a.parent.mkdir(parents=True, exist_ok=True)

        tio_anno.data = tio_anno.data * tio_mask_small.data
        tio_anno.data = torch.from_numpy(sc.ndimage.gaussian_filter(tio_anno.data.numpy(), sigma=1)) # 1
        thr_val = 1.5
        tio_minmask.data[tio_img.data > thr_val] = 1

        tio_anno.save(path_to_save_a)
        tio_img.save(path_to_save_i)
        tio_minmask.save(path_to_save_m)

        _tio_anno = copy.deepcopy(tio_anno)
        _, sx, sy, sz = tio_anno.shape
        lst_pred = []
        lst_pred_value = []
        lst_pred_share_vol = []
        for k in range(topk):
            tio_single = copy.deepcopy(tio_img)
            tio_single.data = torch.zeros_like(tio_single.data)
            max_val = _tio_anno.data.flatten().max()

            _, x, y, z = (_tio_anno.data == max_val).nonzero()[0] # hier assert einfügen
            tio_topkand.data[:, x - r: x + r + 1, y - r: y + r + 1, z - r: z + r + 1] = torch.tensor(rg.sphere((2 * r) + 1, r)) * ((topk + 1) - k)
            tio_single.data[:, x - r: x + r + 1, y - r: y + r + 1, z - r: z + r + 1] = torch.tensor(rg.sphere((2 * r) + 1, r)) * ((topk + 1) - k)
            _tio_anno.data[tio_topkand.data.nonzero(as_tuple=True)] = _tio_anno.data.min()
            lst_pred.append(torch.tensor([x, y, z]))
            lst_pred_value.append(tio_anno.data[tio_single.data.nonzero(as_tuple=True)].max())
            if data_dict['not_healthy'] == True and tio_seg is None:
                _x, _y, _z = data_dict['annotations'][0]['anchor3DX'], data_dict['annotations'][0]['anchor3DY'], data_dict['annotations'][0]['anchor3DZ']
                tio_gt = copy.deepcopy(tio_img)
                tio_gt.data = torch.zeros_like(tio_gt.data)
                tio_gt.data[:, _x - r: _x + r + 1, _y - r: _y + r + 1, _z - r: _z + r + 1] = torch.tensor( rg.sphere((2 * r) + 1, r))
                lst_pred_share_vol.append(torch.logical_and(tio_single.data, tio_gt.data).float().sum())
            else:
                lst_pred_share_vol.append(torch.tensor(-1.0))

        tio_topkand.save(path_to_save_k)

        nilearn_plotting.plot_glass_brain(
            str(path_to_save_k),
            output_file=str(path_to_save_gk),
            title=f"Prediction -> {data_dict['not_healthy']}",
            black_bg=True,
            display_mode="lyrz",
            threshold="auto",
            radiological=True,
        )
        nilearn_plotting.plot_glass_brain(
            str(path_to_save_a),
            output_file=str(path_to_save_ga),
            title=f"Prediction -> {data_dict['not_healthy']}",
            black_bg=True,
            display_mode="lyrz",
            threshold="auto",
            radiological=True,
        )

        def _norm(x, _max, _min):
            return (x - _min) / (_max - _min)

        if data_dict['not_healthy'] == 1:
            tio_org.data = copy.deepcopy(tio_seg.data)
            tio_org.save(path_to_save_o)
            nilearn_plotting.plot_glass_brain(
                str(path_to_save_o),
                output_file=str(path_to_save_go),
                title=f"Ground Truth -> {data_dict['not_healthy']}",
                black_bg=True,
                display_mode="lyrz",
                threshold="auto",
                radiological=True,
            )
        tio_topkand.data[tio_topkand.data >= 1] = 1
        if data_dict['not_healthy'] == True:
            assert len(data_dict['annotations']) == 1

        score_img_level.append({
            'study_uid': uid_parent,
            'not_healthy': data_dict['not_healthy'],
            'anno_dict': data_dict['annotations'][0] if data_dict['not_healthy'] == True else None,
            'preds': torch.vstack(lst_pred),
            'preds_val': torch.hstack(lst_pred_value),
            'preds_share_vol': torch.hstack(lst_pred_share_vol),
            'tio_topkand': copy.deepcopy(tio_topkand),
            'tio_anno': copy.deepcopy(tio_anno),
            'tio_seg': copy.deepcopy(tio_seg)
        })

    path_to_save_jimean = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / f'jimean.json'
    path_to_save_jimax = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / f'jimax.json'
    path_to_save_ja = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / f'ja.json'
    path_to_save_jaa = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / f'jaa.json'
    path_to_save_jlaa = args.data_dir / 'lightning_logs' / f"perf_{args.seq}" / f"{r}_{topk}_pred" / f'jlaa.json'

    _score_img_level = []
    _gt_img_level = []
    for data_dict in score_img_level:
        _score_img_level.append(data_dict['tio_anno'].data[data_dict['tio_topkand'].data.nonzero(as_tuple=True)].mean())
        _gt_img_level.append(data_dict['not_healthy'])
    metric_img_level = compute_stats(torch.hstack(_score_img_level).sigmoid(), torch.as_tensor(_gt_img_level))
    with open(str(path_to_save_jimean), 'w') as fp:
        json.dump(metric_img_level, fp)
    print(metric_img_level)

    _score_img_level = []
    _gt_img_level = []
    for data_dict in score_img_level:
        _score_img_level.append(data_dict['tio_anno'].data[data_dict['tio_topkand'].data.nonzero(as_tuple=True)].max())
        _gt_img_level.append(data_dict['not_healthy'])
    metric_img_level = compute_stats(torch.hstack(_score_img_level).sigmoid(), torch.as_tensor(_gt_img_level))
    with open(str(path_to_save_jimax), 'w') as fp:
        json.dump(metric_img_level, fp)
    print(metric_img_level)

    _pred_anno_level = []
    _gt_anno_level = []
    for data_dict in score_img_level:
        if data_dict['not_healthy'] == True:
            assert len(data_dict['anno_dict']['medicals']) == 1
            tio_gt = copy.deepcopy(data_dict['tio_topkand'])
            tio_gt.data = torch.zeros_like(tio_gt.data)
            if data_dict['tio_seg'] is None:
                x, y, z = torch.tensor([data_dict['anno_dict']['anchor3DX'], data_dict['anno_dict']['anchor3DY'], data_dict['anno_dict']['anchor3DZ']])
                tio_gt.data[:, x - r: x + r + 1, y - r: y + r + 1, z - r: z + r + 1] = torch.tensor(rg.sphere((2 * r) + 1, r))
            else:
                tio_gt.data = copy.deepcopy(data_dict['tio_seg'].data)
            if torch.logical_and(data_dict['tio_topkand'].data, tio_gt.data).float().sum() >= 1:
                _pred_anno_level.append(1)
            else:
                _pred_anno_level.append(0)
            _gt_anno_level.append(1)
    tp = torch.sum(torch.tensor(_pred_anno_level))
    fn = torch.abs(torch.tensor(_pred_anno_level).shape[0] - tp)
    tpfn = len(_pred_anno_level)
    metric_anno_level = {
        'tp': tp.item(),
        'fn': fn.item(),
        'tpfn': tpfn,
        'list': _pred_anno_level
    }
    with open(str(path_to_save_ja), 'w') as fp:
        json.dump(metric_anno_level, fp)
    print(metric_anno_level)


def calc_scores(lst_box_dict, entity):
    lst_score = []
    lst_gt = []
    lst_ent = []
    for score_dict in lst_box_dict:
        if 1 in score_dict['gt'] and entity in score_dict['ent']:
            lst_score.extend(torch.from_numpy(np.hstack(score_dict['score'])))
            lst_gt.extend(torch.as_tensor(score_dict['gt']))
            lst_ent.extend(torch.as_tensor(score_dict['ent']))
    score, gt, ent = torch.hstack(lst_score), torch.hstack(lst_gt), torch.hstack(lst_ent)

    metric = compute_stats(score, gt)

    metric_for_logger = {
        "r_acc": metric['r_acc'], "r_precision":  metric['r_precision'], "r_recall": metric['r_recall'], "r_specificity": metric['r_specificity'],
        "r_fbeta":  metric['r_fbeta'], "auroc": metric['auroc'], "avprec": metric['avprec'], "r_opt_thr":  metric['r_opt_thr'],
        "r_tn": metric['r_tn'], "r_fp": metric['r_fp'], "r_fn": metric['r_fn'], "r_tp": metric['r_tp']
    }
    return metric_for_logger



def compute_stats(score_lst, gt_lst):
    threshold_roc = calc_opt_roc_thr(score_lst, gt_lst)

    threshold_roc = max(0, threshold_roc - 1e-6)

    score_accuracy = tom.Accuracy(task='binary', threshold=threshold_roc).to(torch.device("cpu"))
    r_score_accuracy = score_accuracy(score_lst, gt_lst)
    score_accuracy.reset()

    score_auroc = tom.AUROC(task='binary').to(torch.device("cpu"))
    f_score_auroc = score_auroc(score_lst, gt_lst)
    score_auroc.reset()

    score_average_precision = tom.AveragePrecision(task="binary").to(torch.device("cpu"))
    f_score_average_precision = score_average_precision(score_lst, gt_lst)
    score_average_precision.reset()

    score_fbeta = tom.FBetaScore(task="binary", beta=0.5, threshold=threshold_roc).to(torch.device("cpu"))
    r_score_fbeta = score_fbeta(score_lst, gt_lst)
    score_fbeta.reset()

    score_precision = tom.Precision(task="binary", threshold=threshold_roc).to(torch.device("cpu"))
    r_score_precision = score_precision(score_lst, gt_lst)
    score_precision.reset()

    score_recall = tom.Recall(task="binary", threshold=threshold_roc).to(torch.device("cpu"))
    r_score_recall = score_recall(score_lst, gt_lst)
    score_recall.reset()

    score_specificity = tom.Specificity(task="binary", threshold=threshold_roc).to(torch.device("cpu"))
    r_score_specificity = score_specificity(score_lst, gt_lst)
    score_specificity.reset()

    confusion = tom.ConfusionMatrix(task='binary', threshold=threshold_roc).to(torch.device("cpu"))
    r_confusion = confusion(score_lst, gt_lst).numpy().astype(int)
    confusion.reset()

    metric = {
        "r_acc": float(r_score_accuracy),
        "r_precision": float(r_score_precision),
        "r_recall": float(r_score_recall),
        "r_specificity": float(r_score_specificity),
        "r_fbeta": float(r_score_fbeta),
        "auroc": float(f_score_auroc),
        "avprec": float(f_score_average_precision),
        "r_opt_thr": float(threshold_roc),
        "r_tn": float(r_confusion[0][0]),
        "r_fp": float(r_confusion[0][1]),
        "r_fn": float(r_confusion[1][0]),
        "r_tp": float(r_confusion[1][1])
    }
    return metric


def calc_opt_roc_thr(score_lst, gt_lst):
    score_roc = tom.ROC(task='binary')
    score_roc.update(score_lst, gt_lst)
    fpr, tpr, thresholds = score_roc.compute()
    score_roc.reset()
    optimal_idx = torch.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return float(optimal_threshold)
