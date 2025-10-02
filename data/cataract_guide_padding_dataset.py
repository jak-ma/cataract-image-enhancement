import os
from data.base_dataset import BaseDataset
from utils.image_folder import make_dataset

# =====数据集路径配置===== #
source_dir = ""
target_dir = ""
source_mask_dir = ""
target_mask_dir = ""
# ======================= #


class CataractGuidePaddingDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dir_source = os.path.join(opt.dataroot, source_dir)
        self.dir_target = os.path.join(opt.dataroot, target_dir)
        self.dir_source_mask = os.path.join(opt.dataroot, source_mask_dir)
        self.dir_target_mask = os.path.join(opt.dataroot, target_mask_dir)

        self.source_paths = sorted(make_dataset(self.dir_source, opt.max_dataset_size))
        self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))
        self.source_mask_paths = sorted(make_dataset(self.source_mask_paths, opt.max_dataset_size))
        self.target_mask_paths = sorted(make_dataset(self.target_mask_paths, opt.max_dataset_size))

        


