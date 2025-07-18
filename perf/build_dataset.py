import copy
import torch
import torchio as tio
import random
import tqdm
import pytorch_lightning as pl
from dataset import TileDataset, ValDataset_2
from sklearn import model_selection as sk_model_selection
from lib_ext import DatasetIXI, DatasetBrats


from sklearn import \
    utils as sk_utils


class MRIDataModule(pl.LightningDataModule):
    def __init__(self, args, tile=8):
        super().__init__()
        self.args = args
        self.tile = tile
        self.f_tile = tile * 2

        self.lst_train_nl, self.lst_train_l_1, self.lst_val_nl, self.lst_val_l, self.lst_test_nl = self.extern_data()

    def extern_data(self):
        assert self.args.data_dir.exists(), f'provided path {self.args.data_dir} does not exist'
        lst_dicts_ixi = DatasetIXI(prefix='HISTO').read()
        _lst_dicts_ext = DatasetBrats(prefix='HISTO').read()

        lst_dicts_ext = []
        for _data_dict in tqdm.tqdm(_lst_dicts_ext[:581], 'LOAD EXT DATASET'): # random.sample(_lst_dicts_ext, 581)
            tio_seg = tio.ScalarImage(_data_dict['path_to_nii_seg'][0])
            seg_data = tio_seg.data
            nonzero = seg_data.nonzero()

            idx_nonzero = torch.tensor(random.sample(range(nonzero.size(0)), 1))[0]
            _, x_idx, y_idx, z_idx = nonzero[idx_nonzero]
            tio_seg.data.nonzero()

            _seg_data = seg_data[:,
                          max(x_idx - self.tile, 0): min(x_idx + self.tile, seg_data.shape[1] - 1),
                          max(y_idx - self.tile, 0): min(y_idx + self.tile, seg_data.shape[2] - 1),
                          max(z_idx - self.tile, 0): min(z_idx + self.tile, seg_data.shape[3] - 1)
            ]
            while torch.unique(torch.as_tensor(_seg_data.shape[1:])).shape[0] != 1:
                idx_nonzero = torch.tensor(random.sample(range(self.nonzero_all.size(0)), 1))[0]
                _, x_idx, y_idx, z_idx = self.nonzero_all[idx_nonzero]
                _seg_data = seg_data[:,
                              max(x_idx - self.tile, 0): min(x_idx + self.tile, seg_data.shape[1] - 1),
                              max(y_idx - self.tile, 0): min(y_idx + self.tile, seg_data.shape[2] - 1),
                              max(z_idx - self.tile, 0): min(z_idx + self.tile, seg_data.shape[3] - 1)
                ]

            _data_dict['annotations'] = [{
                'anchor3DX': x_idx,
                'anchor3DY': y_idx,
                'anchor3DZ': z_idx,
                'medicals': [{'medicalEntityID': 1}]
            }]
            lst_dicts_ext.append(copy.deepcopy(_data_dict))
        train_ixi, test_ixi = sk_model_selection.train_test_split(
            lst_dicts_ixi, test_size=0.6, random_state=42
        )
        test_ixi, val_ixi = sk_model_selection.train_test_split(
            test_ixi, test_size=0.5, random_state=42
        )
        train_ext, test_ext = sk_model_selection.train_test_split(
            lst_dicts_ext, test_size=0.6, random_state=42
        )
        test_ext, val_ext = sk_model_selection.train_test_split(
            test_ext, test_size=0.5, random_state=42
        )
        lst_train_nl = train_ixi
        lst_train_l = train_ext
        lst_val_l = val_ext
        lst_val_nl = val_ixi
        lst_test_nl = test_ixi
        return lst_train_nl, lst_train_l, lst_val_nl, lst_val_l, lst_test_nl # [:40]


    def create_sampler_grid(self):
        tio_mask_full = tio.LabelMap(self.args.mni_mask_path)

        tio_mask_all = tio.ScalarImage(self.args.mni_grid_path)

        return tio_mask_full, tio_mask_all


    def build_kk_1(self, dataset_kk_lst, mul_neg_label, stage):
        lst_kk_data_dict_boxes = []
        for data_dict in tqdm.tqdm(dataset_kk_lst, desc="BUILD BOXES KK"):
            if 'city' in list(data_dict.keys()): # wegen extern
                if data_dict['city'] == 'BOCHUM':
                    city = 0
                elif data_dict['city'] == 'BOTTROP':
                    city = 1
                elif data_dict['city'] == 'ISERLOHN':
                    city = 2
            else:
                city = 3
            if len(data_dict['annotations']) > 0:
                for anno_dict in data_dict['annotations']:
                    _data_dict = copy.deepcopy(data_dict)
                    _data_dict = {
                        'seq': str(data_dict[f'path_to_nii_{self.args.seq}'][0]),
                        'seg': _data_dict[f'path_to_nii_seg'][0],
                        'annotations': [copy.deepcopy(anno_dict)],
                        'not_healthy': 1,
                        'xyz': [0, anno_dict['anchor3DX'], anno_dict['anchor3DY'], anno_dict['anchor3DZ']],
                        'weight': 1.0,
                        'city': city
                    }
                    lst_kk_data_dict_boxes.append(_data_dict)
            else:
                if len(data_dict[f'path_to_nii_{self.args.seq}']) > 0:
                    _data_dict = {
                        'seq': str(data_dict[f'path_to_nii_{self.args.seq}'][0]),
                        'seg': None,
                        'not_healthy': 0,
                        'weight': mul_neg_label,
                        'city': city
                    }
                    lst_kk_data_dict_boxes.append(_data_dict)
        return lst_kk_data_dict_boxes

    def calc_weights(self, dataset_lst):
        weights = []
        for data_dict in tqdm.tqdm(dataset_lst, desc="EXTRACT WEIGHTS"):
            weights.append(data_dict['weight'])
        return weights


    def build_subjectdataset(self, stage, max_2=False):
        tio_mask_full, tio_mask_all = self.create_sampler_grid()
        self.tio_mask_full = tio_mask_full

        if stage == 'fit':
            dataset_lst = self.lst_train_nl[:10]
            training = True
        if stage == 'validate':
            if max_2:
                dataset_lst = self.lst_train_l_1[:2] + self.lst_val_nl[:2]
            else:
                dataset_lst = self.lst_train_l_1[:2] + self.lst_val_nl[:2]
            training = False

        if stage == 'fit':
            dataset_lst = sk_utils.shuffle(dataset_lst, random_state=42)
            dataset_lst = self.build_kk_1(dataset_lst, 1, stage)
            weights = self.calc_weights(dataset_lst=dataset_lst)
            dataset_lst, weights = sk_utils.shuffle(dataset_lst, weights, random_state=42)
            queue = TileDataset(
                dataset_lst,
                tio_mask_full,
                patch_size=self.f_tile,
                training=training
            )
            return {
                'queue': queue,
                'weights': weights,
            }
        if stage == 'validate':
            dataset_lst = self.build_kk_1(dataset_lst, 1, stage)
            # lst_data_dict_boxes_kk = lst_data_dict_boxes_kk[:10]
            queue = ValDataset_2(
                self.args.ext,
                dataset_lst,
                tio_mask_all,
                tio_mask_full,
                patch_size=self.f_tile
            )
            return {
                'queue': queue,
            }


    def setup(self, stage: str):
        if stage == 'fit':
            self.dataset_train = self.build_subjectdataset(stage)
            self.dataset_val = self.build_subjectdataset('validate', max_2=True)
        if stage == 'validate':
            self.dataset_val = self.build_subjectdataset('validate')


    def train_dataloader(self):
        patches_loader = torch.utils.data.DataLoader(
            self.dataset_train['queue'],
            shuffle=None, # False if self.args.dist else True,
            pin_memory=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            sampler=torch.utils.data.sampler.WeightedRandomSampler(
                weights=self.dataset_train['weights'],
                num_samples=self.args.num_samples_per_epoch_and_node,
                replacement=True,
            )
        )
        return patches_loader


    def val_dataloader(self):
        patches_loader = torch.utils.data.DataLoader(
            self.dataset_val['queue'],
            shuffle=False,  # False if self.args.dist else True,
            pin_memory=True,
            batch_size=self.args.batch_size_eval,
            num_workers=self.args.num_workers,  # muss null sein
        )
        # patches_loader = None
        return patches_loader

