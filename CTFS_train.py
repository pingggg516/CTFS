from dataset.sonar import augment_normalized
import argparse
from copy import deepcopy
import logging
import os
import pprint
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter

from util.monitor import PerformanceMonitor

# 可靠性分数计算相关类 - 直接集成到主文件中
class GridBasedReliability:
    """网格法可靠性分数计算器"""
    
    def __init__(self, grid_size=32, reliability_threshold=0.4):
        self.grid_size = grid_size
        self.reliability_threshold = reliability_threshold
    
    def get_reliability_mask(self, reliability_scores):
        """根据可靠性分数生成掩码"""
        return reliability_scores > self.reliability_threshold


class CrossTeacherReliabilityValidator:
    """跨教师模型可靠性交叉验证器"""
    
    def __init__(self, grid_size=32, reliability_threshold=0.4):
        self.grid_size = grid_size
        self.reliability_threshold = reliability_threshold
        self.general_reliability_calculator = GridBasedReliability(grid_size, reliability_threshold)
        self.sonar_reliability_calculator = GridBasedReliability(grid_size, reliability_threshold)
    
    def calculate_cross_teacher_reliability(self, image, model_ema_general, model_ema_sonar_a, model_ema_sonar_b,
                                          reliability_augmenter, teacher_mode):
        """
        计算跨教师模型的可靠性分数，集成教师间一致性
        Args:
            image: 原始图像 [B, C, H, W]
            model_ema_general: 通用教师模型
            model_ema_sonar_a: 声纳教师A模型
            model_ema_sonar_b: 声纳教师B模型
            reliability_augmenter: 增强器
            teacher_mode: 当前教师模式
        Returns:
            final_reliability_scores: 融合后的可靠性分数 [B, 1, H, W]
        """
        B, C, H, W = image.shape
        
        # 1. 计算各个教师的可靠性分数
        general_reliability_scores = self._calculate_teacher_reliability(
            image, model_ema_general, reliability_augmenter, 'general'
        )
        
        sonar_a_reliability_scores = self._calculate_teacher_reliability(
            image, model_ema_sonar_a, reliability_augmenter, 'sonar_a'
        )
        
        sonar_b_reliability_scores = self._calculate_teacher_reliability(
            image, model_ema_sonar_b, reliability_augmenter, 'sonar_b'
        )
        
        # 2. 计算教师间一致性分数
        teacher_consistency_scores = self._calculate_teacher_consistency(
            image, model_ema_general, model_ema_sonar_a, model_ema_sonar_b
        )
        
        # 3. 简单均匀融合各教师的可靠性分数
        weighted_reliability = (
            general_reliability_scores + 
            sonar_a_reliability_scores + 
            sonar_b_reliability_scores
        ) / 3.0
        
        # 5. 应用一致性惩罚：一致性低的区域降低整体可靠性
        # 一致性分数作为乘性因子，范围[0.5, 1.0]，避免过度惩罚
        consistency_penalty = 0.5 + 0.5 * teacher_consistency_scores
        final_reliability_scores = weighted_reliability * consistency_penalty
        
        return final_reliability_scores
    
    def _calculate_teacher_reliability(self, image, teacher_model, reliability_augmenter, teacher_mode):
        """计算单个教师的可靠性分数"""
        B, C, H, W = image.shape
        
        # 1. 对原始图像进行预测
        with torch.no_grad():
            original_pred = teacher_model(image).detach()
            original_prob = torch.softmax(original_pred, dim=1)  # [B, C, H, W]
        
        # 2. 生成专用的增强视图
        reliability_views = reliability_augmenter.generate_reliability_views(image, teacher_mode)
        
        # 3. 对每个增强视图进行预测并还原到原始尺寸和方位
        augmented_predictions = []
        
        for i, view in enumerate(reliability_views):
            with torch.no_grad():
                pred = teacher_model(view).detach()
                prob = torch.softmax(pred, dim=1)
                
                # 根据增强类型还原预测结果
                if teacher_mode == 'general':
                    if i == 0:  # 水平翻转视图
                        # 将预测结果翻转回来
                        prob = torch.flip(prob, dims=[3])
                    elif i == 1:  # 缩放视图
                        # 将预测结果插值回原图尺寸
                        prob = torch.nn.functional.interpolate(
                            prob, size=(H, W), mode='bilinear', align_corners=False
                        )
                else:  # 阴影和能量衰减不改变几何位置，无需还原方位
                    pass
                
                augmented_predictions.append(prob)
        
        # 4. 计算原图预测与每个增强视图预测的网格级余弦相似度
        grid_h = H // self.grid_size
        grid_w = W // self.grid_size
        
        # 如果图像尺寸不能被网格大小整除，调整网格大小
        if H % self.grid_size != 0:
            grid_h += 1
        if W % self.grid_size != 0:
            grid_w += 1
        
        # 初始化网格级别的可靠性分数
        grid_reliability = torch.zeros(B, grid_h, grid_w).to(image.device)
        
        # 对每个网格计算可靠性分数
        for i in range(grid_h):
            for j in range(grid_w):
                # 计算网格边界
                start_h = i * self.grid_size
                end_h = min((i + 1) * self.grid_size, H)
                start_w = j * self.grid_size
                end_w = min((j + 1) * self.grid_size, W)
                
                # 提取原图该网格的预测结果并计算网格级别的特征向量
                original_grid = original_prob[:, :, start_h:end_h, start_w:end_w]
                original_feature = original_grid.mean(dim=(2, 3))  # [B, C]
                
                # 计算原图与各个增强视图的余弦相似度
                similarities = []
                for aug_pred in augmented_predictions:
                    aug_grid = aug_pred[:, :, start_h:end_h, start_w:end_w]
                    aug_feature = aug_grid.mean(dim=(2, 3))  # [B, C]
                    
                    # 计算余弦相似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        original_feature, aug_feature, dim=1
                    )
                    similarities.append(cos_sim)
                
                # 计算所有相似度的平均值作为网格的可靠性分数
                if similarities:
                    grid_reliability[:, i, j] = torch.stack(similarities).mean(dim=0)
                else:
                    grid_reliability[:, i, j] = 1.0
        
        # 5. 将网格级别的可靠性分数扩展到像素级别
        pixel_reliability = torch.zeros(B, 1, H, W).to(image.device)
        
        for i in range(grid_h):
            for j in range(grid_w):
                start_h = i * self.grid_size
                end_h = min((i + 1) * self.grid_size, H)
                start_w = j * self.grid_size
                end_w = min((j + 1) * self.grid_size, W)
                
                # 将网格的可靠性分数赋给网格内所有像素
                pixel_reliability[:, 0, start_h:end_h, start_w:end_w] = grid_reliability[:, i, j].unsqueeze(-1).unsqueeze(-1)
        
        return pixel_reliability
    
    def _calculate_teacher_consistency(self, image, model_ema_general, model_ema_sonar_a, model_ema_sonar_b):
        """计算三个教师对原始图像预测的一致性"""
        B, C, H, W = image.shape
        
        with torch.no_grad():
            # 通用教师对原始图像的预测
            general_pred = model_ema_general(image).detach()
            general_prob = torch.softmax(general_pred, dim=1)
            
            # 声纳教师A对原始图像的预测
            sonar_a_pred = model_ema_sonar_a(image).detach()
            sonar_a_prob = torch.softmax(sonar_a_pred, dim=1)
            
            # 声纳教师B对原始图像的预测
            sonar_b_pred = model_ema_sonar_b(image).detach()
            sonar_b_prob = torch.softmax(sonar_b_pred, dim=1)
        
        # 计算网格级别的一致性
        grid_h = H // self.grid_size
        grid_w = W // self.grid_size
        
        # 如果图像尺寸不能被网格大小整除，调整网格大小
        if H % self.grid_size != 0:
            grid_h += 1
        if W % self.grid_size != 0:
            grid_w += 1
        
        # 初始化网格级别的一致性分数
        grid_consistency = torch.zeros(B, grid_h, grid_w).to(image.device)
        
        # 对每个网格计算一致性分数
        for i in range(grid_h):
            for j in range(grid_w):
                # 计算网格边界
                start_h = i * self.grid_size
                end_h = min((i + 1) * self.grid_size, H)
                start_w = j * self.grid_size
                end_w = min((j + 1) * self.grid_size, W)
                
                # 提取网格区域的预测结果
                general_grid = general_prob[:, :, start_h:end_h, start_w:end_w]
                sonar_a_grid = sonar_a_prob[:, :, start_h:end_h, start_w:end_w]
                sonar_b_grid = sonar_b_prob[:, :, start_h:end_h, start_w:end_w]
                
                # 计算网格级别的特征向量（对网格内所有像素取平均）
                general_feature = general_grid.mean(dim=(2, 3))  # [B, C]
                sonar_a_feature = sonar_a_grid.mean(dim=(2, 3))  # [B, C]
                sonar_b_feature = sonar_b_grid.mean(dim=(2, 3))  # [B, C]
                
                # 计算三个教师之间的余弦相似度
                cos_sim_ga = torch.nn.functional.cosine_similarity(
                    general_feature, sonar_a_feature, dim=1
                )
                cos_sim_gb = torch.nn.functional.cosine_similarity(
                    general_feature, sonar_b_feature, dim=1
                )
                cos_sim_ab = torch.nn.functional.cosine_similarity(
                    sonar_a_feature, sonar_b_feature, dim=1
                )
                
                # 取三个相似度的平均值作为一致性分数
                grid_consistency[:, i, j] = (cos_sim_ga + cos_sim_gb + cos_sim_ab) / 3.0
        
        # 将网格级别的一致性分数扩展到像素级别
        pixel_consistency = torch.zeros(B, 1, H, W).to(image.device)
        
        for i in range(grid_h):
            for j in range(grid_w):
                start_h = i * self.grid_size
                end_h = min((i + 1) * self.grid_size, H)
                start_w = j * self.grid_size
                end_w = min((j + 1) * self.grid_size, W)
                
                # 将网格的一致性分数赋给网格内所有像素
                pixel_consistency[:, 0, start_h:end_h, start_w:end_w] = grid_consistency[:, i, j].unsqueeze(-1).unsqueeze(-1)
        
        return pixel_consistency
    
    def get_reliability_mask(self, reliability_scores):
        """
        根据可靠性分数生成可靠性掩码
        
        Args:
            reliability_scores: 可靠性分数张量 (B, H, W)
            
        Returns:
            reliable_mask: 可靠性掩码 (B, H, W)，True表示可靠
        """
        return reliability_scores > self.reliability_threshold


class ReliabilityAugmentations:
    """可靠性分数计算专用的增强操作类"""
    
    def __init__(self):
        pass
    
    def generate_reliability_views(self, image, teacher_mode):
        """根据教师模式生成用于可靠性计算的增强视图"""
        if teacher_mode == 'general':
            return self._apply_general_augmentations(image)
        elif teacher_mode == 'sonar_a':
            return self._apply_sonar_a_augmentations(image)
        elif teacher_mode == 'sonar_b':
            return self._apply_sonar_b_augmentations(image)
        else:
            return [image.clone()]
    
    def _apply_general_augmentations(self, image):
        """通用教师的增强操作：水平翻转、尺寸缩放"""
        views = []
        
        # 1. 水平翻转视图
        flipped = torch.flip(image, dims=[3])
        views.append(flipped)
        
        # 2. 尺寸缩放视图（随机放大或缩小）
        B, C, H, W = image.shape
        # 随机缩放因子：0.7-1.3倍
        scale_factor = torch.rand(1).item() * 0.6 + 0.7  # 0.7到1.3之间
        
        # 分别计算缩放后的高度和宽度
        scaled_H = int(H * scale_factor)
        scaled_W = int(W * scale_factor)
        
        # 确保缩放后的尺寸能被patch尺寸（14）整除
        patch_size = 14
        scaled_H = (scaled_H // patch_size) * patch_size
        scaled_W = (scaled_W // patch_size) * patch_size
        
        # 确保最小尺寸不小于patch_size
        if scaled_H < patch_size:
            scaled_H = patch_size
        if scaled_W < patch_size:
            scaled_W = patch_size
        
        scaled = torch.nn.functional.interpolate(image, size=(scaled_H, scaled_W), mode='bilinear', align_corners=False)
        views.append(scaled)
        
        return views
    
    def _apply_sonar_a_augmentations(self, image):
        # Independent stochastic shadow views; no speckle or occlusion.
        return [augment_normalized(image, 'sonar_a') for _ in range(2)]
    
    def _apply_sonar_b_augmentations(self, image):
        # Deterministic Eq. 11: a repeated identical view adds no information.
        return [augment_normalized(image, 'sonar_b')]


def get_current_teacher_mode(epoch, warmup_epochs=15):
    """
    获取当前epoch应该使用的教师模型模式
    
    Args:
        epoch: 当前epoch (从0开始)
        warmup_epochs: 预热阶段的epoch数
    
    Returns:
        str: 'supervised' (预热阶段), 'general' (通用教师), 'sonar_a' (声纳教师A), 'sonar_b' (声纳教师B)
    """
    if epoch < warmup_epochs:
        return 'supervised'  # 预热阶段：全监督训练
    
    # 15个epoch后的交替逻辑：通用教师1个epoch，声纳教师A 1个epoch，声纳教师B 1个epoch
    cycle_epoch = epoch - warmup_epochs
    cycle_position = cycle_epoch % 3  # 3个epoch为一个周期（1+1+1）
    
    if cycle_position == 0:
        return 'general'   # 通用教师：1个epoch
    elif cycle_position == 1:
        return 'sonar_a'   # 声纳教师A：1个epoch
    else:
        return 'sonar_b'   # 声纳教师B：1个epoch


parser = argparse.ArgumentParser(description='CTFS: single-GPU training with three EMA teachers')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--resume', action='store_true', help='Resume from latest.pth in --save-path')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    resume_path = os.path.join(args.save_path, 'latest.pth')
    if args.resume:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError('Resume checkpoint not found: ' + resume_path)
    elif os.path.exists(args.save_path):
        raise FileExistsError('Use a new save directory, or pass --resume: ' + args.save_path)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # 单卡训练：注释分布式初始化
    # rank, world_size = setup_distributed(port=args.port)
    rank, world_size = 0, 1
    
    # Add file handler to save logs to txt file
    if rank == 0:
        log_file_path = os.path.join(args.save_path, 'training.log')
        os.makedirs(args.save_path, exist_ok=args.resume)
        file_handler = logging.FileHandler(
            log_file_path, mode='a' if args.resume else 'w', encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
        
        # Initialize Performance Monitor
        monitor = PerformanceMonitor(logger)
        monitor.start_training()
    else:
        monitor = None

    cudnn.enabled = True
    cudnn.benchmark = True

    if cfg['model'] == 'dpt':
        model_configs = {
            'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
        if not args.resume:
            state_dict = torch.load(cfg['pretrained_path'], map_location='cpu')
            model.backbone.load_state_dict(state_dict)
    elif cfg['model'] in ('deeplabv3', 'deeplabv3plus'):
        # The legacy module wraps torchvision DeepLabV3, not DeepLabV3+.
        from model.semseg.deeplabv3plus import DeepLabV3Plus as DeepLabV3
        model = DeepLabV3(nclass=cfg['nclass'], pretrained=not args.resume)
    else:
        raise NotImplementedError(f"Unsupported model '{cfg['model']}'")
        
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    # 单卡训练：注释分布式模型包装
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = 0
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    # )
    
    # 创建三个教师模型
    model_ema_general = deepcopy(model)  # 通用教师
    model_ema_general.eval()
    for param in model_ema_general.parameters():
        param.requires_grad = False
    
    model_ema_sonar_a = deepcopy(model)  # 声纳教师A
    model_ema_sonar_a.eval()
    for param in model_ema_sonar_a.parameters():
        param.requires_grad = False
    
    model_ema_sonar_b = deepcopy(model)  # 声纳教师B
    model_ema_sonar_b.eval()
    for param in model_ema_sonar_b.parameters():
        param.requires_grad = False
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(reduction='none', **cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    
    # 初始化跨教师模型可靠性交叉验证器和增强器
    cross_teacher_validator = CrossTeacherReliabilityValidator(grid_size=cfg['grid_size'], reliability_threshold=cfg['reliability_threshold'])
    reliability_augmenter = ReliabilityAugmentations()
    
    # 保留原有的单教师可靠性计算器作为备用
    reliability_calculator = GridBasedReliability(grid_size=32, reliability_threshold=0.4)

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    # 单卡训练：注释分布式采样器
    # trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True
    )
    
    # trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True
    )
    
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, shuffle=False
    )
    
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    
    if args.resume:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        # 加载三个教师模型，兼容旧版本checkpoint
        if 'model_ema_general' in checkpoint and 'model_ema_sonar_a' in checkpoint and 'model_ema_sonar_b' in checkpoint:
            model_ema_general.load_state_dict(checkpoint['model_ema_general'])
            model_ema_sonar_a.load_state_dict(checkpoint['model_ema_sonar_a'])
            model_ema_sonar_b.load_state_dict(checkpoint['model_ema_sonar_b'])
        elif 'model_ema_general' in checkpoint and 'model_ema_sonar' in checkpoint:
            # 兼容双教师版本：使用原有的model_ema_sonar初始化两个声纳教师
            model_ema_general.load_state_dict(checkpoint['model_ema_general'])
            model_ema_sonar_a.load_state_dict(checkpoint['model_ema_sonar'])
            model_ema_sonar_b.load_state_dict(checkpoint['model_ema_sonar'])
        else:
            # 兼容旧版本：使用原有的model_ema初始化三个教师模型
            if 'model_ema' in checkpoint:
                model_ema_general.load_state_dict(checkpoint['model_ema'])
                model_ema_sonar_a.load_state_dict(checkpoint['model_ema'])
                model_ema_sonar_b.load_state_dict(checkpoint['model_ema'])
            else:
                # 如果没有EMA模型，使用当前学生模型初始化
                model_ema_general.load_state_dict(checkpoint['model'])
                model_ema_sonar_a.load_state_dict(checkpoint['model'])
                model_ema_sonar_b.load_state_dict(checkpoint['model'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            monitor.start_epoch()

        # 记录epoch开始时间
        epoch_start_time = time.time()
        
        # 获取当前epoch的教师模型模式
        teacher_mode = get_current_teacher_mode(epoch, cfg['warmup_epochs'])
        
        # 更新数据集的teacher_mode
        trainset_u.teacher_mode = teacher_mode
        trainset_l.teacher_mode = teacher_mode
        
        # 在预热阶段结束时（epoch=15），用训练好的学生模型初始化三个教师模型
        if epoch == cfg['warmup_epochs']:
            if rank == 0:
                logger.info('===========> Warmup phase completed! Initializing three teacher models with student model parameters...')
            
            # 用当前学生模型参数初始化三个教师模型
            model_ema_general.load_state_dict(model.state_dict())
            model_ema_sonar_a.load_state_dict(model.state_dict())
            model_ema_sonar_b.load_state_dict(model.state_dict())
            
            if rank == 0:
                logger.info('===========> Three teacher models initialized successfully!')
        
        if rank == 0:
            logger.info('===========> Epoch: {:}, Teacher mode: {}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, teacher_mode, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        # 单卡训练：注释分布式采样器的epoch设置
        # trainloader_l.sampler.set_epoch(epoch)
        # trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        for i, ((img_x, mask_x),
                (img_u_w_standard, img_u_w_sonar_a, img_u_w_sonar_b, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w_standard, img_u_w_sonar_a, img_u_w_sonar_b = img_u_w_standard.cuda(), img_u_w_sonar_a.cuda(), img_u_w_sonar_b.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            # 根据教师模式选择相应的教师模型和增强策略
            if teacher_mode == 'supervised':
                # 预热阶段：不使用伪标签，跳过无监督损失
                pred_u_w = None
                conf_u_w = None
                mask_u_w = None
            else:
                with torch.no_grad():
                    # 1. 伪标签生成（保持原有逻辑）
                    if teacher_mode == 'general':
                        # 使用通用教师模型，输入标准弱增强图像生成伪标签
                        pred_u_w = model_ema_general(img_u_w_standard).detach()
                    elif teacher_mode == 'sonar_a':
                        # 使用声纳教师A模型，输入声纳专用弱增强图像生成伪标签
                        pred_u_w = model_ema_sonar_a(img_u_w_sonar_a).detach()
                    elif teacher_mode == 'sonar_b':
                        # 使用声纳教师B模型，输入声纳专用弱增强图像生成伪标签
                        pred_u_w = model_ema_sonar_b(img_u_w_sonar_b).detach()
                    
                    conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                    mask_u_w = pred_u_w.argmax(dim=1)
                    
                    # 2. 跨教师模型可靠性交叉验证（独立的过程）
                    # 使用标准弱增强图像作为基础图像
                    reliability_base_image = img_u_w_standard
                    
                    # 计算跨教师模型的可靠性分数
                    reliability_scores = cross_teacher_validator.calculate_cross_teacher_reliability(
                        reliability_base_image, 
                        model_ema_general, 
                        model_ema_sonar_a,
                        model_ema_sonar_b,
                        reliability_augmenter, 
                        teacher_mode
                    )
                    
                    # 获取可靠性掩码
                    reliable_mask = cross_teacher_validator.get_reliability_mask(reliability_scores)
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=False).chunk(2)

            # Eq. 2: ignored/padded pixels have zero CE, with full-grid denominator.
            loss_x = criterion_l(pred_x, mask_x).mean()
            
            # 根据教师模式计算损失
            if teacher_mode == 'supervised':
                # 预热阶段：只使用监督损失
                loss_u_s = torch.tensor(0.0).cuda()
                loss = loss_x
            else:
                # 半监督阶段：计算无监督损失
                mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
                ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
                
                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
                ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]

                # 应用cutmix到可靠性掩码和可靠性分数
                # 确保reliable_mask形状正确，去掉多余的维度
                reliable_mask_squeezed = reliable_mask.squeeze(1)  # [B, H, W]
                reliable_mask_cutmixed1 = reliable_mask_squeezed.clone()
                reliable_mask_cutmixed2 = reliable_mask_squeezed.clone()
                reliable_mask_cutmixed1[cutmix_box1 == 1] = reliable_mask_squeezed.flip(0)[cutmix_box1 == 1]
                reliable_mask_cutmixed2[cutmix_box2 == 1] = reliable_mask_squeezed.flip(0)[cutmix_box2 == 1]
                
                # 应用cutmix到可靠性分数
                # 确保reliability_scores形状正确，去掉多余的维度
                reliability_scores_squeezed = reliability_scores.squeeze(1)  # [B, H, W]
                reliability_scores_cutmixed1 = reliability_scores_squeezed.clone()
                reliability_scores_cutmixed2 = reliability_scores_squeezed.clone()
                reliability_scores_cutmixed1[cutmix_box1 == 1] = reliability_scores_squeezed.flip(0)[cutmix_box1 == 1]
                reliability_scores_cutmixed2[cutmix_box2 == 1] = reliability_scores_squeezed.flip(0)[cutmix_box2 == 1]

                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                # 使用配置阈值过滤和可靠性分数权重机制
                reliability_threshold = cfg['reliability_threshold']
                
                # 创建可靠性权重：低于阈值的设为0（不参与训练），高于阈值的使用原始可靠性分数作为权重
                reliability_weights1 = torch.where(
                    reliability_scores_cutmixed1 > reliability_threshold,
                    reliability_scores_cutmixed1,  # 使用实际的可靠性分数作为权重
                    torch.zeros_like(reliability_scores_cutmixed1)  # 低于阈值的设为0
                )
                
                # 应用可靠性权重和忽略掩码
                valid_mask1 = (reliability_weights1 > 0) & (ignore_mask_cutmixed1 != 255)
                loss_u_s1 = loss_u_s1 * reliability_weights1 * valid_mask1.float()
                # Eq. 17: average over B*H*W, including rejected pixels as zero.
                loss_u_s1 = loss_u_s1.mean()
                
                loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                # 使用配置阈值过滤和可靠性分数权重机制
                reliability_weights2 = torch.where(
                    reliability_scores_cutmixed2 > reliability_threshold,
                    reliability_scores_cutmixed2,  # 使用实际的可靠性分数作为权重
                    torch.zeros_like(reliability_scores_cutmixed2)  # 低于阈值的设为0
                )
                
                # 应用可靠性权重和忽略掩码
                valid_mask2 = (reliability_weights2 > 0) & (ignore_mask_cutmixed2 != 255)
                loss_u_s2 = loss_u_s2 * reliability_weights2 * valid_mask2.float()
                # Eq. 17: average over B*H*W, including rejected pixels as zero.
                loss_u_s2 = loss_u_s2.mean()
                
                loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
                loss = loss_x + cfg['lambda_u'] * loss_u_s
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            
            # 计算mask ratio，在预热阶段设为0
            if teacher_mode == 'supervised':
                mask_ratio = 0.0
            else:
                # 使用配置阈值的可靠性分数计算mask ratio
                reliability_threshold = cfg['reliability_threshold']
                reliable_pixels = (reliability_scores.squeeze(1) > reliability_threshold) & (ignore_mask != 255)
                mask_ratio = reliable_pixels.sum().item() / max(1, (ignore_mask != 255).sum().item())
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            # EMA更新：根据教师模式选择更新的教师模型
            if teacher_mode != 'supervised':  # 只在非预热阶段更新EMA
                ema_ratio = min(1 - 1 / (iters + 1), 0.996)
                
                if teacher_mode == 'general':
                    # 更新通用教师模型
                    for param, param_ema in zip(model.parameters(), model_ema_general.parameters()):
                        param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
                    for buffer, buffer_ema in zip(model.buffers(), model_ema_general.buffers()):
                        buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
                elif teacher_mode == 'sonar_a':
                    # 更新声纳教师A模型
                    for param, param_ema in zip(model.parameters(), model_ema_sonar_a.parameters()):
                        param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
                    for buffer, buffer_ema in zip(model.buffers(), model_ema_sonar_a.buffers()):
                        buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
                elif teacher_mode == 'sonar_b':
                    # 更新声纳教师B模型
                    for param, param_ema in zip(model.parameters(), model_ema_sonar_b.parameters()):
                        param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
                    for buffer, buffer_ema in zip(model.buffers(), model_ema_sonar_b.buffers()):
                        buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % max(1, len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))
        
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        
        # Clear CUDA cache before evaluation
        torch.cuda.empty_cache()
        
        # Standard evaluation
        mult = 14 if cfg['model'] == 'dpt' else None
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=mult)
        
        # 评估三个教师模型
        mIoU_ema_general, iou_class_ema_general = evaluate(model_ema_general, valloader, eval_mode, cfg, multiplier=mult)
        mIoU_ema_sonar_a, iou_class_ema_sonar_a = evaluate(model_ema_sonar_a, valloader, eval_mode, cfg, multiplier=mult)
        mIoU_ema_sonar_b, iou_class_ema_sonar_b = evaluate(model_ema_sonar_b, valloader, eval_mode, cfg, multiplier=mult)
        
        # 选择最佳的教师模型结果作为EMA结果
        if mIoU_ema_general >= mIoU_ema_sonar_a and mIoU_ema_general >= mIoU_ema_sonar_b:
            mIoU_ema, iou_class_ema = mIoU_ema_general, iou_class_ema_general
            best_teacher = 'general'
        elif mIoU_ema_sonar_a >= mIoU_ema_sonar_b:
            mIoU_ema, iou_class_ema = mIoU_ema_sonar_a, iou_class_ema_sonar_a
            best_teacher = 'sonar_a'
        else:
            mIoU_ema, iou_class_ema = mIoU_ema_sonar_b, iou_class_ema_sonar_b
            best_teacher = 'sonar_b'
        
        # Clear CUDA cache after evaluation
        torch.cuda.empty_cache()
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'General EMA: {:.2f}, Sonar A EMA: {:.2f}, Sonar B EMA: {:.2f}, Best EMA: {:.2f} ({})'.format(
                                cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, 
                                iou_class_ema_general[cls_idx], iou_class_ema_sonar_a[cls_idx], 
                                iou_class_ema_sonar_b[cls_idx], iou_class_ema[cls_idx], best_teacher))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, General EMA: {:.2f}, '
                        'Sonar A EMA: {:.2f}, Sonar B EMA: {:.2f}, Best EMA: {:.2f} ({})'.format(
                            eval_mode, mIoU, mIoU_ema_general, mIoU_ema_sonar_a, mIoU_ema_sonar_b, mIoU_ema, best_teacher))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema_general', mIoU_ema_general, epoch)
            writer.add_scalar('eval/mIoU_ema_sonar_a', mIoU_ema_sonar_a, epoch)
            writer.add_scalar('eval/mIoU_ema_sonar_b', mIoU_ema_sonar_b, epoch)
            writer.add_scalar('eval/mIoU_ema_best', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema_general' % (CLASSES[cfg['dataset']][i]), iou_class_ema_general[i], epoch)
                writer.add_scalar('eval/%s_IoU_ema_sonar_a' % (CLASSES[cfg['dataset']][i]), iou_class_ema_sonar_a[i], epoch)
                writer.add_scalar('eval/%s_IoU_ema_sonar_b' % (CLASSES[cfg['dataset']][i]), iou_class_ema_sonar_b[i], epoch)

        is_best = mIoU >= previous_best
        is_best_ema = mIoU_ema >= previous_best_ema
        
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema_general': model_ema_general.state_dict(),
                'model_ema_sonar_a': model_ema_sonar_a.state_dict(),
                'model_ema_sonar_b': model_ema_sonar_b.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema,
                'teacher_mode': teacher_mode,
                'best_teacher': best_teacher,
                # All class scores below belong to this checkpoint's epoch.
                'class_names': list(CLASSES[cfg['dataset']]),
                'metrics': {
                    'student': {'mIoU': float(mIoU), 'iou_class': iou_class.tolist()},
                    'general': {'mIoU': float(mIoU_ema_general), 'iou_class': iou_class_ema_general.tolist()},
                    'sonar_a': {'mIoU': float(mIoU_ema_sonar_a), 'iou_class': iou_class_ema_sonar_a.tolist()},
                    'sonar_b': {'mIoU': float(mIoU_ema_sonar_b), 'iou_class': iou_class_ema_sonar_b.tolist()},
                    'best_ema': {
                        'teacher': best_teacher,
                        'mIoU': float(mIoU_ema),
                        'iou_class': iou_class_ema.tolist()
                    }
                }
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            if is_best_ema:
                torch.save(checkpoint, os.path.join(args.save_path, 'best_ema.pth'))
        
        # 计算并打印epoch耗时
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if rank == 0:
            logger.info('***** Epoch {} completed in {:.2f} seconds ({:.2f} minutes) *****'.format(
                epoch, epoch_duration, epoch_duration / 60.0))
            monitor.end_epoch(epoch)
            monitor.log_performance_summary()

    if rank == 0:
        writer.close()


if __name__ == '__main__':
    main()
