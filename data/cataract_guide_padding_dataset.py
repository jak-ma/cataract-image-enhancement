import os
from data.base_dataset import BaseDataset, get_transform_six_channel, get_params
from utils.image_folder import make_dataset
from PIL import Image
import random

# ========数据集路径配置======== #
source_dir = "source_AB"        #
target_dir = "target"           #
source_mask_dir = "source_mask" #
target_mask_dir = "target_mask" #
# ============================= #


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

    



        self.target_size = len(self.target_paths)
        assert(self.opt.load_size>=self.opt.crop_size), 'crop_szie should be smaller than load_size'

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc 
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = self.opt.isTrain
    
    def __len__(self):
        return len(self.source_paths)
    
    # 负责单样本加载和数据预处理
    def __getitem__(self, index):

        source_path = self.source_paths[index]
        # TODO 后面需要修改
        source_mask_path = os.path.join(self.dir_source_mask, os.path.split(source_path)[-1].replace('jpg', 'png'))
        
        target_index = random.randint(0, self.target_size-1) if self.isTrain else index % self.target_size
        target_path = self.target_paths[target_index]
        target_mask_path = self.target_mask_paths[target_index]

        SAB = Image.open(source_path).convert('RGB')
        TA = Image.open(target_path).convert('RGB')

        SA_mask = Image.open(source_mask_path).convert('L')
        SB_mask = SA_mask
        TA_mask = Image.open(target_mask_path).convert('L')

        w, h = SAB.size
        w2 = int(w/2)
        SA = SAB.crop((0, 0, w2, h))
        SB = SAB.crop((w2, 0, w, h))

        source_transform_params = get_params(self.opt, SA.size)
        source_A_transform, source_A_mask_transform = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.input_nc==1))
        source_B_transform, _ = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.output_nc==1))
        ### TODO 这里可以提出一个问题
        target_transform_params = get_params(self.opt, TA.size)
        target_A_transform, target_A_mask_transform = get_transform_six_channel(self.opt, target_transform_params, grayscale=(self.input_nc==1))

        SA = source_A_transform(SA)
        S_mask = source_A_mask_transform(SA_mask)

        SB = source_B_transform(SB)

        TA = target_A_transform(TA)
        T_mask = target_A_mask_transform(TA_mask)

        return {'SA': SA, 'SB': SB, 'S_mask': S_mask, 'SA_path': source_path, 'SB_path': source_path, 
                'TA': TA, 'T_mask': T_mask, 'TA_path': target_path}