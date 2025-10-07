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
        # 整理数据集路径
        self.dir_source = os.path.join(opt.dataroot, source_dir)
        self.dir_target = os.path.join(opt.dataroot, target_dir)
        self.dir_source_mask = os.path.join(opt.dataroot, source_mask_dir)
        self.dir_target_mask = os.path.join(opt.dataroot, target_mask_dir)
        # 将数据集路径以列表形式存储
        self.source_paths = make_dataset(self.dir_source, opt.max_dataset_size)
        self.target_paths = make_dataset(self.dir_target, opt.max_dataset_size)
        self.source_mask_paths = make_dataset(self.dir_source_mask, opt.max_dataset_size)
        self.target_mask_paths = make_dataset(self.dir_target_mask, opt.max_dataset_size)

        self.target_size = len(self.target_paths)
        assert (opt.load_size >= opt.crop_size), 'load_size should bigger than crop_size, please check the file base_options.py'
        
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain

    



        


