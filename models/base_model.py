import os
import torch
from abc import ABC, abstractmethod
from . import networks
from collections import OrderedDict

class BaseModel(ABC):
    ### 继承 父类 ABC 表明这个类是个抽象基类
    ### 使用 @abstractmethod 修饰器修饰的函数必须在子类中被实现
    # 初始化方法
    def __init__(self, opt):
        super().__init__()
        # 命令行获取
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids else torch.device('cpu')
        # 指定结果保存目录
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # 预处理图像方式：'scale_width' 会将输入的的图像变得尺寸不一致
        # 设置了 benchmark = True → cuDNN 会根据当前硬件和图像尺寸寻找出处理速度最快的卷积算法
        # 预处理图像方式：'scale_width' 会将输入的的图像变得尺寸不一致
        # 因此 cuDNN 需要为每种不同尺寸都重新搜索最优算法，严重影响了效率
        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        # 需要最终在 visdom 上进行可视化的图片类型
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        # 这个是一个监控指标值，当比如 监控损失值，当损失不再下降时，触发学习率下降
        self.metric = 0
    
    # 改变命令行参数函数
    @abstractmethod
    def modify_commandline_options(parser, is_train):
        # 这里可以根据具体模型的需要写一些新的需要在命令行指定的参数
        # 或者重写现有默认参数
        return parser
    
    # 使得 输入 "适应" 模型需要
    @abstractmethod
    def set_input(self, input):
        # input 作为一个字典输入
        # 包括 数据本身 和 数据的一些说明信息
        # 首先 unpack 来自 DataLoader 数据装载器的数据
        # 并且可以根据模型的不同，进行一些预处理操作
        # 如 transforms.Compose() 的一些操作
        pass

    # 前向传播函数
    @abstractmethod
    def forward(self):
        # called by: 被xxx调用
        pass

    # 参数优化
    @abstractmethod
    def optimize_parameters(self):
        # 计算 loss, grads; 更新 weights.
        pass

    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = f'{opt.load_iter}' if opt.load_iter > 0 else f'{opt.epoch}'
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)
        pass

    # test 时进行模型评估
    def eval(self):
        for name in self.model_names:
            # 类型判断
            if isinstance(name, str):
                # net = self.net'name'
                net = getattr(self, 'net' + name)
                net.eval()

    # 正常的 test 步骤
    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
    
    # 计算可视化的输出
    def compute_visuals(self):
        pass

    # 使用scheduler规划学习率，退化学习 (更新学习率的函数)
    # 感觉这个函数的设置具有很大的特殊性，没有较好的复用价值
    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        
        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'learning rate {old_lr:.7f} -> {lr:.7f}')

    # 返回当前的可视化图像
    # 训练逻辑中定义了通过 visdom 来展示
    # 并使用一个 html 文件来保存
    def get_current_visuals(self):
        # 有序字典 OrderedDict 按照插入的顺序排列
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_'+name)
        return errors_ret

    # 保存模型
    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f'{epoch}_net_{name}.pth'
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                # 先把模型参数移至CPU, 然后再保存
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    # 保存完了以后继续把模型转移到 GPU 进行训练
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.module.cpu().state_dict(), save_path)

    # 私有函数 (感觉没搞明白，嵌套的挺多的)
    def __path_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # 从磁盘中加载模型
    def load_networks(self, epoch):
        for name in self.model_names:
            load_filename = f'{epoch}_net_{name}.pth'
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net'+name)
 
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print(f'loading the model net{name} from {load_path}...')
            state_dict = torch.load(load_path, map_location=self.device)
            # TODO state_dict._metadata 包含了一些模型相关的辅助信息，在 state_dict() 时自动添加进去
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            for key in list(state_dict.keys()):  
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)

    
    # 输出网络的总参数 | 模型结构
    def print_networks(self, verbose):
        # verbose 是一个 bool 值
        print('--------------- Networks initialized ----------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    # param.numel() -> numel() 能够计算每个张量元素的个数
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(f'[NetWorks {name}] Total number of parameters: {num_params/1e6:.3f} M')
        print('----------------------------------------------------')

    # 设置模型是否需要 计算梯度
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            # 小tips，标准化设计
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad