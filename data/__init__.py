import importlib
import torch 
from data.base_dataset import BaseDataset

def find_dataset_using_name(dataset_name):
    # 动态导入模块
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls
            break
    if dataset is None:
        raise NotImplementedError(f"In the {dataset_filename}.py, there should be a subclass of BaseDataset with class name that matchs {target_dataset_name} in lowercase.")
    
    return dataset

def get_options_setter(dataset_name):
    dataset = find_dataset_using_name(dataset_name)
    return dataset.modify_commandline_options

# 支持多线程数据加载的数据集包装器
class CustomDatasetDataLoader():
    def __init__(self, opt):
        
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print(f"dataset [{type(self.dataset).__name__}] was created")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads)
        )
    
    def load_data(self):
        return self
    
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i*self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

def create_data(opt):
    data_loader = CustomDatasetDataLoader(opt)
    return data_loader.load_data()