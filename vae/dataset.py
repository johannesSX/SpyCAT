import torch
import torchio as tio
import numpy as np
import random
import tqdm
from perlin_numpy import generate_perlin_noise_3d

from typing import TypeVar, Optional, Iterator, List

T_co = TypeVar('T_co', covariant=True)

class LesionNoise:
    """
        Applies synthetic lesion-like noise (via Perlin noise) to a 3D patch of an image.
    """
    def __init__(self, size_range=[3, 5], iters=[0, 1, 2], f_ps=16):
        super(LesionNoise, self).__init__()
        self.size_range = size_range
        # self.iters = iters
        self.f_ps = f_ps

    def transform(self, img):
        """
        Apply lesion-like Perlin noise patch to the input image tensor.
        """
        # Generate Perlin noise of patch size
        noise = generate_perlin_noise_3d(
            (self.f_ps, self.f_ps, self.f_ps), (4, 4, 4), tileable=(False, False, False)
        )
        noise = torch.as_tensor(noise, dtype=img.dtype).unsqueeze(0)
        _noise = noise.clone()

        # Insert noise where noise >= 0, paste minimum elsewhere
        noise[_noise < 0] = img.min()
        noise[_noise >= 0] = img[_noise >= 0]

        # Randomly determine and place lesion region
        s = random.choice(self.size_range)
        ss = int(np.floor(s / 2))
        x = random.randint(0 + ss, self.f_ps - ss - 1)
        y = random.randint(0 + ss, self.f_ps - ss - 1)
        z = random.randint(0 + ss, self.f_ps - ss - 1)
        img[:, x - ss: x + ss + 1, y - ss: y + ss + 1, z - ss: z + ss + 1] = noise[:, x - ss: x + ss + 1, y - ss: y + ss + 1, z - ss: z + ss + 1]
        return img


class CentroidsDataset(torch.utils.data.Dataset):

    def  __init__(self, subject_boxes, tio_mask_all, patch_size=32, training=False):
        super(CentroidsDataset, self).__init__()
        print('INIT CENTROIDSDATSET')
        self.tio_mask_all = tio_mask_all
        self.nonzero_all = tio_mask_all.data.nonzero()
        self.subject_boxes = self._pre_compute_nonzero_idx(subject_boxes)  # [0:1000]
        self.ps = patch_size // 2
        self.f_ps = patch_size
        self.training = training
        self.tio_ref = tio.LabelMap('../data/masks/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii')

        paths = self._extract_paths(self.subject_boxes)
        self.ram_dict = self._init_ram_dict(paths)

        self.global_tile_idx = None

    def _pre_compute_nonzero_idx(self, subject_boxes):
        """
        Pre-compute closest nonzero mask voxels for each subject.
        """
        _subject_boxes = []
        for subject in subject_boxes:
            if subject['not_healthy'] == 0:
                subject['idx_nonzero'] = None
            elif subject['not_healthy'] == 1:
                c_dist = torch.cdist(self.nonzero_all[:, 1:].float(), torch.tensor(subject['xyz'][1:]).unsqueeze(0).float(), p=2)
                idx_nonzero = torch.argmin(c_dist)
                # assert c_dist.min() == 0.0
                # _, x_idx, y_idx, z_idx = self.nonzero_all[idx_nonzero]
                subject['idx_nonzero'] = idx_nonzero
            _subject_boxes.append(subject)
        return _subject_boxes

    def _extract_img(self, subject, x_idx, y_idx, z_idx):
        """
        Return the patch centered at (x_idx, y_idx, z_idx) for the given subject.
        """
        img_data = self.ram_dict[subject['seq'].replace('/media/johsch/newtondata/phd_newton/', '../')]
        _img_data = img_data[
                   :,
                   x_idx - self.ps: x_idx + self.ps,
                   y_idx - self.ps: y_idx + self.ps,
                   z_idx - self.ps: z_idx + self.ps
        ]
        assert list(_img_data.shape) == [1, self.f_ps, self.f_ps, self.f_ps]
        return _img_data

    def _extract_paths(self, subject_boxes):
        """
        Collect unique image paths for memory mapping, standardize path prefix.
        """
        paths_nii = []
        for subject in tqdm.tqdm(subject_boxes, desc='EXTRACT PATH FOR RAM INIT'):
            if subject['seq'] not in paths_nii:
                paths_nii.append(subject['seq'].replace('/media/johsch/newtondata/phd_newton/', '../'))
        return paths_nii

    def _init_ram_dict(self, paths):
        """
        Pre-load image volumes to a RAM dictionary.
        """
        ram_dict = {}
        for path in tqdm.tqdm(paths, desc='INIT RAM DICT'):
            tmp = {
                path: tio.ScalarImage(path).data,
            }
            ram_dict.update(tmp)
        print(f'LOADED {len(paths)} INTO RAM DICT')
        return ram_dict

    def __len__(self):
        return len(self.subject_boxes)

    def set_global_tile_idx(self, idx):
        self.global_tile_idx = idx

    def __getitem__(self, idx):
        """
        Returns a patch for given subject.
        """
        subject = self.subject_boxes[idx]
        x_idx, y_idx, z_idx = -1, -1, -1
        x_idx_org, y_idx_org, z_idx_org = [0, 0, 0]
        entity = -1
        if subject['not_healthy'] == 0:
            _, x_idx, y_idx, z_idx = self.nonzero_all[self.global_tile_idx]
        elif subject['not_healthy'] == 1 and subject['idx_nonzero'] == self.global_tile_idx:
            _, x_idx, y_idx, z_idx = self.nonzero_all[self.global_tile_idx] # 117, 181, 70
            _, x_idx_org, y_idx_org, z_idx_org = subject['xyz']
            if subject['not_healthy'] == 1:
                assert len(subject['annotations']) == 1
                assert len(subject['annotations'][0]['medicals']) == 1
                entity = subject['annotations'][0]['medicals'][0]['medicalEntityID']
        if subject['not_healthy'] == 1 and subject['idx_nonzero'] != self.global_tile_idx:
            in_data = torch.zeros((1, self.f_ps, self.f_ps, self.f_ps))
        else:
            in_data = self._extract_img(subject, x_idx, y_idx, z_idx)
        if torch.unique(in_data).shape[0] < 10:
            x_idx, y_idx, z_idx = -1, -1, -1

        return [in_data, self.global_tile_idx, idx, torch.tensor([x_idx, y_idx, z_idx]), torch.tensor([x_idx_org, y_idx_org, z_idx_org]), subject['not_healthy'], entity]


class TileDataset(torch.utils.data.Dataset):
    """
        PyTorch Dataset for randomly sampling 3D patches from healthy data, with optional lesion augmentation.
    """

    def __init__(self, subject_boxes, tio_mask_all, patch_size=32, training=False):
        super(TileDataset, self).__init__()
        print('INIT TILEDATSET TEACHER')
        self.tio_mask_all = tio_mask_all
        self.nonzero_all = tio_mask_all.data.nonzero()
        self.subject_boxes = subject_boxes

        self.ps = patch_size // 2
        self.f_ps = patch_size
        self.training = training

        paths_nii = self._extract_paths(self.subject_boxes)
        self.ram_dict_nii = self._init_ram_dict(paths_nii)

        self.transform = LesionNoise()

    def _help_extract_img_0(self, img_data_1):
        """
        Randomly extracts a patch centered at a nonzero mask index.
        """
        idx_nonzero = torch.tensor(random.sample(range(self.nonzero_all.size(0)), 1))[0]
        _, x_idx, y_idx, z_idx = self.nonzero_all[idx_nonzero]
        _img_data_1 = img_data_1[:,
                    max(x_idx - self.ps, 0): min(x_idx + self.ps, img_data_1.shape[1] - 1),
                    max(y_idx - self.ps, 0): min(y_idx + self.ps, img_data_1.shape[2] - 1),
                    max(z_idx - self.ps, 0): min(z_idx + self.ps, img_data_1.shape[3] - 1)
                    ]
        while _img_data_1.unique().shape[0] < 10 or torch.unique(torch.as_tensor(_img_data_1.shape[1:])).shape[0] != 1:
            idx_nonzero = torch.tensor(random.sample(range(self.nonzero_all.size(0)), 1))[0]
            _, x_idx, y_idx, z_idx = self.nonzero_all[idx_nonzero]
            _img_data_1 = img_data_1[:,
                          max(x_idx - self.ps, 0): min(x_idx + self.ps, img_data_1.shape[1] - 1),
                          max(y_idx - self.ps, 0): min(y_idx + self.ps, img_data_1.shape[2] - 1),
                          max(z_idx - self.ps, 0): min(z_idx + self.ps, img_data_1.shape[3] - 1)
                          ]
        return _img_data_1

    def _extract_img_0(self, subject):
        """
        Return a randomly patch for given subject.
        """
        img_data_1 = self.ram_dict_nii[subject['seq']]
        _img_data_1 = self._help_extract_img_0(img_data_1)
        assert list(_img_data_1.shape) == [1, self.f_ps, self.f_ps, self.f_ps]
        return _img_data_1

    def set_ram_dict_with_indices(self, indices):
        """
        Loads a subset of subject volumes into RAM for fast access.
        """
        sub_subject_boxes = [self.subject_boxes[index] for index in indices]
        paths_nii = self._extract_paths(sub_subject_boxes)
        print(f'START LOADING {len(paths_nii)} INTO RAM DICT, S-INDEX {indices[0]} -> E-INDEX {indices[-1]}')
        self.ram_dict_nii = self._init_ram_dict(paths_nii)

    def _extract_paths(self, subject_boxes):
        paths_nii = []
        for subject in tqdm.tqdm(subject_boxes, desc='EXTRACT PATH FOR RAM INIT'):
            if subject['seq'] not in paths_nii:
                paths_nii.append(subject['seq'])
        return paths_nii

    def _init_ram_dict(self, paths_nii):
        """
        Pre-load image volumes for random access.
        """
        ram_dict_nii = {}
        for path_nii in tqdm.tqdm(paths_nii, desc='INIT RAM DICT', total=len(paths_nii)):
            tmp_nii = {
                path_nii: tio.ScalarImage(path_nii).data,  # sitk.tio.ScalarImage.as_sitk(tio.ScalarImage(path_nii))
            }
            ram_dict_nii.update(tmp_nii)
        print(f'LOADED {len(paths_nii)} INTO RAM DICT')
        return ram_dict_nii

    def __len__(self):
        return len(self.subject_boxes)

    def __getitem__(self, idx):
        """
        Returns a random, lesion-augmented patch for healthy subjects.
        """
        subject = self.subject_boxes[idx]

        if subject['not_healthy'] == 0:
            in_data = self._extract_img_0(subject)
            in_data = self.transform.transform(in_data)
        return in_data