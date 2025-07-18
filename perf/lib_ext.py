import copy
import glob
import os

class SuperDataset():

    def __init__(self):
        super(SuperDataset, self).__init__()

    def get_template_dict(self):
        template_dict = {
            "t1": [],
            "t1ks": [],
            "t2": [],
            "swi": [],
            "flair": [],
            "ct": [],
            "xray": [],
            "etc": [],
            "seg": [],
            "not_healthy": [],
            "class_label": [],
            "cdr": [],
            "mask": [],
            "keyword": [],
        }
        return template_dict


# -------------------------------- IXI --------------------------------
class DatasetIXI(SuperDataset):

    def __init__(self, prefix='SKULL'):
        super(DatasetIXI, self).__init__()
        self.file_cat = "IXI"
        self.prefix = prefix

    def read(self):
        template_dict = super().get_template_dict()
        lst_template_dict = []

        path_to_t1, path_to_t2 = "../data/{}/IXI/IXI-T1/*.nii.gz".format(self.prefix), "../data/{}/IXI/IXI-T2/".format(self.prefix)
        lst_files_t1 = glob.glob(path_to_t1)

        for t1 in lst_files_t1:
            name_t1 = os.path.basename(os.path.normpath(t1))[:-10]
            name_t2 = "{}/{}-T2.nii.gz".format(path_to_t2, name_t1)

            data_dict = copy.deepcopy(template_dict)
            data_dict["path_to_nii_t1"] = [t1]
            data_dict["path_to_nii_t2"] = []
            data_dict["not_healthy"] = [False]
            data_dict["annotations"] = []

            if os.path.isfile(name_t2):
                data_dict["path_to_nii_t2"] = [name_t2]

            data_dict["keyword"] = ["ixi"]
            lst_template_dict.append(data_dict)
        return lst_template_dict


# -------------------------------- BRATS --------------------------------
class DatasetBrats(SuperDataset):

    def __init__(self, prefix='HISTO', t1c_as_add_label=False):
        super(DatasetBrats, self).__init__()
        self.file_cat = "BRATS"
        self.prefix = prefix
        self.t1c_as_add_label = t1c_as_add_label

    def read(self, train_val=True):
        template_dict = super().get_template_dict()
        lst_template_dict = []

        if train_val:
            path_to_t1, path_to_t1c, path_to_t2, path_to_flair, path_to_seg = \
                "../data/{}/BRATS/BraTS2021_Training_Data/*/*_t1.nii.gz".format(self.prefix), \
                "../data/{}/BRATS/BraTS2021_Training_Data/*/*_t1ce.nii.gz".format(self.prefix), \
                "../data/{}/BRATS/BraTS2021_Training_Data/*/*_t2.nii.gz".format(self.prefix), \
                "../data/{}/BRATS/BraTS2021_Training_Data/*/*_flair.nii.gz".format(self.prefix), \
                "../data/{}/BRATS/BraTS2021_Training_Data/*/*_seg.nii.gz".format(self.prefix)
        else:
            path_to_t1, path_to_t1c, path_to_t2, path_to_flair, path_to_seg = \
                "../data/{}/BRATS/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/*/*_t1.nii.gz".format(self.prefix), \
                "../data/{}/BRATS/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/*/*_t1ce.nii.gz".format(self.prefix), \
                "../data/{}/BRATS/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/*/*_t2.nii.gz".format(self.prefix), \
                "../data/{}/BRATS/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/*/*_flair.nii.gz".format(self.prefix), \
                "../data/{}/BRATS/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/*/*_seg.nii.gz".format(self.prefix)

        lst_files_t1 = glob.glob(path_to_t1)
        lst_files_t1c = glob.glob(path_to_t1c)
        lst_files_t2 = glob.glob(path_to_t2)
        lst_files_flair = glob.glob(path_to_flair)
        lst_files_seg = glob.glob(path_to_seg)

        for t1, t1c, t2, flair, seg in zip(lst_files_t1, lst_files_t1c, lst_files_t2, lst_files_flair, lst_files_seg):
            data_dict = copy.deepcopy(template_dict)
            data_dict["path_to_nii_t1"] = [t1]
            data_dict["path_to_nii_t1ks"] = [t1c]
            data_dict["path_to_nii_t2"] = [t2]
            data_dict["path_to_nii_flair"] = [flair]
            data_dict["path_to_nii_seg"] = [seg]
            data_dict["not_healthy"] = [True]
            data_dict["keyword"] = ["brats"]

            lst_template_dict.append(data_dict)
        return lst_template_dict