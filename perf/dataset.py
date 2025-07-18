import torch
import torchio as tio
import random
import tqdm
from typing import TypeVar, Optional, Iterator, List

T_co = TypeVar('T_co', covariant=True)


def _help_extract_neighbors(img_data, x_idx, y_idx, z_idx, f_ps, ps, mni_x=193, mni_y=229, mni_z=193):
    x_idx, y_idx, z_idx = x_idx, y_idx, (2 + 16) + z_idx
    _img_data = torch.zeros((1, mni_x, mni_y, 2 + mni_z + 8))
    _img_data[:, 0 : mni_x, 0 : mni_y, 2 : 2 + mni_z] = img_data

    ps_max = torch.as_tensor(_img_data.shape) - 1
    _neighbor_data_3 = _img_data[
                       :,
                       max(x_idx - ps, 0): min(x_idx + ps, ps_max[1]),
                       max(y_idx - ps, 0): min(y_idx + ps, ps_max[2]),
                       max(z_idx + f_ps - ps, 0): min(z_idx + f_ps + ps, ps_max[3])
                       ]
    assert _neighbor_data_3.shape == (1, f_ps, f_ps, f_ps)
    return torch.stack([
        _neighbor_data_3
    ])


class ValDataset_2(torch.utils.data.Dataset):

    def  __init__(self, ext, subject_boxes, tio_mask_sel, tio_mask_full, patch_size=32):
        super(ValDataset_2, self).__init__()

        self.ext = ext
        self.ps = patch_size // 2
        self.f_ps = patch_size

        self.nonzero_sel = tio_mask_sel.data.nonzero()
        self.nonzero_full = tio_mask_full.data.nonzero()

        self.subject_boxes = subject_boxes

        paths_nii = self._extract_paths(self.subject_boxes)
        self.ram_dict_nii = self._init_ram_dict(paths_nii)

        self.tio_ref = tio.LabelMap('../data/masks/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii')

    def _extract_paths(self, subject_boxes):
        paths_nii = []
        for subject in tqdm.tqdm(subject_boxes, desc='EXTRACT PATH FOR RAM INIT'):
            if subject['seq'] not in paths_nii:
                paths_nii.append(subject['seq'].replace('/media/johsch/newtondata/phd_newton/', '../'))
        return paths_nii

    def _init_ram_dict(self, paths_nii):
        ram_dict_nii = {}
        for path_nii in tqdm.tqdm(paths_nii, desc='INIT RAM DICT', total=len(paths_nii)):
            tmp_nii = {
                path_nii: tio.ScalarImage(path_nii).data,  # sitk.tio.ScalarImage.as_sitk(tio.ScalarImage(path_nii))
            }
            ram_dict_nii.update(tmp_nii)
        print(f'LOADED {len(paths_nii)} INTO RAM DICT')
        return ram_dict_nii

    def extract_img(self, subject, x_idx, y_idx, z_idx, ps=8):
        img_data = self.ram_dict_nii[subject['seq'].replace('/media/johsch/newtondata/phd_newton/', '../')]
        _img_data = img_data[
                    :,
                    x_idx - ps: x_idx + ps,
                    y_idx - ps: y_idx + ps,
                    z_idx - ps: z_idx + ps
                    ]
        _neighbors = _help_extract_neighbors(img_data, x_idx, y_idx, z_idx, self.f_ps, self.ps)
        return _img_data, _neighbors

    def set_global_subject_idx(self, idx):
        self.global_subject_idx = idx

    def __len__(self):
        return len(self.nonzero_sel)

    def __getitem__(self, idx):
        subject = self.subject_boxes[ self.global_subject_idx]
        _, x_idx, y_idx, z_idx = self.nonzero_sel[idx]
        # _, x_idx, y_idx, z_idx = self.nonzero_full.max(0).values

        img_data, neighbors = self.extract_img(subject, x_idx, y_idx, z_idx)

        idcs_nonzero_perf = torch.tensor([
            x_idx,  # 0 -> 144
            y_idx + self.tio_ref.data.shape[1],
            z_idx + self.tio_ref.data.shape[1] + self.tio_ref.data.shape[2],
        ])

        return [img_data, neighbors, idcs_nonzero_perf, self.nonzero_sel[idx]]


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, subject_boxes, tio_mask_full, patch_size=32, training=False):
        super(TileDataset, self).__init__()
        print('INIT TILEDATSET TEACHER')
        self.tio_mask_full = tio_mask_full
        self.nonzero_full = tio_mask_full.data.nonzero()
        # self.nonzero_sel = tio_mask_sel.data.nonzero()

        self.subject_boxes = subject_boxes

        self.ps = patch_size // 2
        self.f_ps = patch_size
        self.training = training

        paths_nii = self._extract_paths(self.subject_boxes)
        self.ram_dict_nii = self._init_ram_dict(paths_nii)

    def _help_extract_img_0(self, img_data):
        idx_nonzero = torch.tensor(random.sample(range(self.nonzero_full.size(0)), 1))[0]
        _, x_idx, y_idx, z_idx = self.nonzero_full[idx_nonzero]

        _idcs_nonzero_perf = torch.tensor([
            x_idx,  # 0 -> 144
            y_idx + img_data.shape[1],
            z_idx + img_data.shape[1] + img_data.shape[2],
        ])
        _img_data = img_data[:,
                    max(x_idx - self.ps, 0): min(x_idx + self.ps, img_data.shape[1] - 1),
                    max(y_idx - self.ps, 0): min(y_idx + self.ps, img_data.shape[2] - 1),
                    max(z_idx - self.ps, 0): min(z_idx + self.ps, img_data.shape[3] - 1)
                    ]
        while _img_data.unique().shape[0] < 10 or torch.unique(torch.as_tensor(_img_data.shape[1:])).shape[0] != 1:
            idx_nonzero = torch.tensor(random.sample(range(self.nonzero_full.size(0)), 1))[0]
            _, x_idx, y_idx, z_idx = self.nonzero_full[idx_nonzero]
            _idcs_nonzero_perf = torch.tensor([
                x_idx,  # 0 -> 144
                y_idx + img_data.shape[1],
                z_idx + img_data.shape[1] + img_data.shape[2],
            ])
            _img_data = img_data[:,
                        max(x_idx - self.ps, 0): min(x_idx + self.ps, img_data.shape[1] - 1),
                        max(y_idx - self.ps, 0): min(y_idx + self.ps, img_data.shape[2] - 1),
                        max(z_idx - self.ps, 0): min(z_idx + self.ps, img_data.shape[3] - 1)
                        ]
        _neighbors = _help_extract_neighbors(img_data, x_idx, y_idx, z_idx, self.f_ps, self.ps)
        return _img_data, _neighbors, _idcs_nonzero_perf

    def _extract_img_0(self, subject):
        img_data = self.ram_dict_nii[subject['seq']]
        _img_data, _neighbors, _idcs_nonzero_perf = self._help_extract_img_0(img_data)
        assert list(_img_data.shape) == [1, self.f_ps, self.f_ps, self.f_ps]
        return _img_data, _neighbors, _idcs_nonzero_perf

    def set_ram_dict_with_indices(self, indices):
        sub_subject_boxes = [self.subject_boxes[index] for index in indices]
        paths_nii = self._extract_paths(sub_subject_boxes)
        print(f'START LOADING {len(paths_nii)} INTO RAM DICT, S-INDEX {indices[0]} -> E-INDEX {indices[-1]}')
        # print(indices)
        self.ram_dict_nii = self._init_ram_dict(paths_nii)

    def _extract_paths(self, subject_boxes):
        paths_nii = []
        for subject in tqdm.tqdm(subject_boxes, desc='EXTRACT PATH FOR RAM INIT'):
            if subject['seq'] not in paths_nii:
                paths_nii.append(subject['seq'])
        return paths_nii


    def _init_ram_dict(self, paths_nii):
        ram_dict_nii = {}
        for path_nii in tqdm.tqdm(paths_nii, desc='INIT RAM DICT', total=len(paths_nii)):
            tmp_nii = {
                path_nii: tio.ScalarImage(path_nii).data,
            }
            ram_dict_nii.update(tmp_nii)
        print(f'LOADED {len(paths_nii)} INTO RAM DICT')
        return ram_dict_nii

    def __len__(self):
        return len(self.subject_boxes)

    def __getitem__(self, idx):
        subject = self.subject_boxes[idx]

        if subject['not_healthy'] == 0:
            in_data, _neighbors, _idcs_nonzero_perf = self._extract_img_0(subject)
        return in_data, _neighbors, _idcs_nonzero_perf