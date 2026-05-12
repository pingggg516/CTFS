import argparse
from copy import deepcopy
import logging
import os
import pprint

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

from util.dist_helper import setup_distributed



class GridBasedReliability:


    def __init__(self, grid_size=32, reliability_threshold=0.7):
        self.grid_size = grid_size
        self.reliability_threshold = reliability_threshold

    def get_reliability_mask(self, reliability_scores):

        return reliability_scores > self.reliability_threshold


class CrossTeacherReliabilityValidator:


    def __init__(self, grid_size=32, reliability_threshold=0.3):
        self.grid_size = grid_size
        self.reliability_threshold = reliability_threshold
        self.general_reliability_calculator = GridBasedReliability(grid_size, reliability_threshold)
        self.sonar_reliability_calculator = GridBasedReliability(grid_size, reliability_threshold)

    def calculate_cross_teacher_reliability(self, image, model_ema_general, model_ema_sonar,
                                          reliability_augmenter, teacher_mode):
        B, C, H, W = image.shape


        general_reliability_scores = self._calculate_teacher_reliability(
            image, model_ema_general, reliability_augmenter, 'general'
        )


        sonar_reliability_scores = self._calculate_teacher_reliability(
            image, model_ema_sonar, reliability_augmenter, 'sonar'
        )


        consistency_scores = self._calculate_teacher_consistency(
            image, model_ema_general, model_ema_sonar
        )


        final_reliability_scores = (
            0.4 * general_reliability_scores +
            0.4 * sonar_reliability_scores +
            0.2 * consistency_scores
        )

        return final_reliability_scores

    def _calculate_teacher_reliability(self, image, teacher_model, reliability_augmenter, teacher_mode):
        B, C, H, W = image.shape


        with torch.no_grad():
            original_pred = teacher_model(image).detach()
            original_prob = torch.softmax(original_pred, dim=1)


        reliability_views = reliability_augmenter.generate_reliability_views(image, teacher_mode)


        augmented_predictions = []

        for i, view in enumerate(reliability_views):
            with torch.no_grad():
                pred = teacher_model(view).detach()
                prob = torch.softmax(pred, dim=1)


                if teacher_mode == 'general':
                    if i == 0:

                        prob = torch.flip(prob, dims=[3])
                    elif i == 1:

                        prob = torch.nn.functional.interpolate(
                            prob, size=(H, W), mode='bilinear', align_corners=False
                        )
                else:
                    pass

                augmented_predictions.append(prob)


        grid_h = H // self.grid_size
        grid_w = W // self.grid_size


        if H % self.grid_size != 0:
            grid_h += 1
        if W % self.grid_size != 0:
            grid_w += 1


        grid_reliability = torch.zeros(B, grid_h, grid_w).to(image.device)


        for i in range(grid_h):
            for j in range(grid_w):

                start_h = i * self.grid_size
                end_h = min((i + 1) * self.grid_size, H)
                start_w = j * self.grid_size
                end_w = min((j + 1) * self.grid_size, W)


                original_grid = original_prob[:, :, start_h:end_h, start_w:end_w]
                original_feature = original_grid.mean(dim=(2, 3))


                similarities = []
                for aug_pred in augmented_predictions:
                    aug_grid = aug_pred[:, :, start_h:end_h, start_w:end_w]
                    aug_feature = aug_grid.mean(dim=(2, 3))


                    cos_sim = torch.nn.functional.cosine_similarity(
                        original_feature, aug_feature, dim=1
                    )
                    similarities.append(cos_sim)


                if similarities:
                    grid_reliability[:, i, j] = torch.stack(similarities).mean(dim=0)
                else:
                    grid_reliability[:, i, j] = 1.0


        pixel_reliability = torch.zeros(B, 1, H, W).to(image.device)

        for i in range(grid_h):
            for j in range(grid_w):
                start_h = i * self.grid_size
                end_h = min((i + 1) * self.grid_size, H)
                start_w = j * self.grid_size
                end_w = min((j + 1) * self.grid_size, W)


                pixel_reliability[:, 0, start_h:end_h, start_w:end_w] = grid_reliability[:, i, j].unsqueeze(-1).unsqueeze(-1)

        return pixel_reliability

    def _calculate_teacher_consistency(self, image, model_ema_general, model_ema_sonar):

        B, C, H, W = image.shape

        with torch.no_grad():

            general_pred = model_ema_general(image).detach()
            general_prob = torch.softmax(general_pred, dim=1)


            sonar_pred = model_ema_sonar(image).detach()
            sonar_prob = torch.softmax(sonar_pred, dim=1)


        grid_h = H // self.grid_size
        grid_w = W // self.grid_size


        if H % self.grid_size != 0:
            grid_h += 1
        if W % self.grid_size != 0:
            grid_w += 1


        grid_consistency = torch.zeros(B, grid_h, grid_w).to(image.device)


        for i in range(grid_h):
            for j in range(grid_w):

                start_h = i * self.grid_size
                end_h = min((i + 1) * self.grid_size, H)
                start_w = j * self.grid_size
                end_w = min((j + 1) * self.grid_size, W)


                general_grid = general_prob[:, :, start_h:end_h, start_w:end_w]
                sonar_grid = sonar_prob[:, :, start_h:end_h, start_w:end_w]


                general_feature = general_grid.mean(dim=(2, 3))
                sonar_feature = sonar_grid.mean(dim=(2, 3))


                cos_sim = torch.nn.functional.cosine_similarity(
                    general_feature, sonar_feature, dim=1
                )

                grid_consistency[:, i, j] = cos_sim


        pixel_consistency = torch.zeros(B, 1, H, W).to(image.device)

        for i in range(grid_h):
            for j in range(grid_w):
                start_h = i * self.grid_size
                end_h = min((i + 1) * self.grid_size, H)
                start_w = j * self.grid_size
                end_w = min((j + 1) * self.grid_size, W)


                pixel_consistency[:, 0, start_h:end_h, start_w:end_w] = grid_consistency[:, i, j].unsqueeze(-1).unsqueeze(-1)

        return pixel_consistency

    def get_reliability_mask(self, reliability_scores):
        return reliability_scores >= self.reliability_threshold


class ReliabilityAugmentations:

    def __init__(self):
        pass

    def generate_reliability_views(self, image, teacher_mode):
        if teacher_mode == 'general':
            return self._apply_general_augmentations(image)
        elif teacher_mode == 'sonar':
            return self._apply_sonar_augmentations(image)
        else:
            return [image.clone()]

    def _apply_general_augmentations(self, image):
        views = []


        flipped = torch.flip(image, dims=[3])
        views.append(flipped)


        B, C, H, W = image.shape

        scale_factor = torch.rand(1).item() * 0.6 + 0.7
        scaled_size = int(H * scale_factor)

        scaled_size = (scaled_size // 14) * 14
        if scaled_size < 14:
            scaled_size = 14
        scaled = torch.nn.functional.interpolate(image, size=(scaled_size, scaled_size), mode='bilinear', align_corners=False)
        views.append(scaled)

        return views

    def _apply_sonar_augmentations(self, image):
        views = []


        noise = torch.randn_like(image) * 0.05
        noisy = torch.clamp(image + image * noise, 0, 1)
        views.append(noisy)


        occluded = image.clone()
        B, C, H, W = image.shape
        for b in range(B):

            num_occlusions = torch.randint(3, 6, (1,)).item()
            for _ in range(num_occlusions):

                center_x = torch.randint(0, W, (1,)).item()
                center_y = torch.randint(0, H, (1,)).item()

                radius = torch.randint(int(min(H, W) * 0.02), int(min(H, W) * 0.08), (1,)).item()


                y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

                a_ratio = torch.rand(1).item() * 0.5 + 0.5
                b_ratio = torch.rand(1).item() * 0.5 + 0.5
                ellipse_mask = ((x_indices - center_x) ** 2 / (radius * a_ratio) ** 2 +
                               (y_indices - center_y) ** 2 / (radius * b_ratio) ** 2) <= 1


                occluded[b, :, ellipse_mask] = 0

        views.append(occluded)

        return views


def get_current_teacher_mode(epoch, warmup_epochs=15):
    if epoch < warmup_epochs:
        return 'supervised'


    cycle_epoch = epoch - warmup_epochs
    cycle_position = cycle_epoch % 2

    if cycle_position == 0:
        return 'general'
    else:
        return 'sonar'


parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0



    rank, world_size = 0, 1


    if rank == 0:
        log_file_path = os.path.join(args.save_path, 'training.log')
        os.makedirs(args.save_path, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)

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



    local_rank = 0

    model.cuda()






    model_ema_general = deepcopy(model)
    model_ema_general.eval()
    for param in model_ema_general.parameters():
        param.requires_grad = False

    model_ema_sonar = deepcopy(model)
    model_ema_sonar.eval()
    for param in model_ema_sonar.parameters():
        param.requires_grad = False

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)


    cross_teacher_validator = CrossTeacherReliabilityValidator(grid_size=32, reliability_threshold=0.3)
    reliability_augmenter = ReliabilityAugmentations()


    reliability_calculator = GridBasedReliability(grid_size=32, reliability_threshold=0.7)

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )



    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True
    )


    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, shuffle=True
    )


    valloader = DataLoader(
        valset, batch_size=2, pin_memory=True, num_workers=2, drop_last=False, shuffle=False
    )

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])


        if 'model_ema_general' in checkpoint and 'model_ema_sonar' in checkpoint:
            model_ema_general.load_state_dict(checkpoint['model_ema_general'])
            model_ema_sonar.load_state_dict(checkpoint['model_ema_sonar'])
        else:

            if 'model_ema' in checkpoint:
                model_ema_general.load_state_dict(checkpoint['model_ema'])
                model_ema_sonar.load_state_dict(checkpoint['model_ema'])
            else:

                model_ema_general.load_state_dict(checkpoint['model'])
                model_ema_sonar.load_state_dict(checkpoint['model'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):

        teacher_mode = get_current_teacher_mode(epoch)


        trainset_u.teacher_mode = teacher_mode
        trainset_l.teacher_mode = teacher_mode


        if epoch == 15:
            if rank == 0:
                logger.info('===========> Warmup phase completed! Initializing dual teacher models with student model parameters...')


            model_ema_general.load_state_dict(model.state_dict())
            model_ema_sonar.load_state_dict(model.state_dict())

            if rank == 0:
                logger.info('===========> Dual teacher models initialized successfully!')

        if rank == 0:
            logger.info('===========> Epoch: {:}, Teacher mode: {}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, teacher_mode, previous_best, best_epoch, previous_best_ema, best_epoch_ema))

        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()





        loader = zip(trainloader_l, trainloader_u)

        model.train()

        for i, ((img_x, mask_x),
                (img_u_w_standard, img_u_w_sonar, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w_standard, img_u_w_sonar = img_u_w_standard.cuda(), img_u_w_sonar.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()


            if teacher_mode == 'supervised':

                pred_u_w = None
                conf_u_w = None
                mask_u_w = None
            else:
                with torch.no_grad():

                    if teacher_mode == 'general':

                        pred_u_w = model_ema_general(img_u_w_standard).detach()
                    elif teacher_mode == 'sonar':

                        pred_u_w = model_ema_sonar(img_u_w_sonar).detach()

                    conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                    mask_u_w = pred_u_w.argmax(dim=1)



                    reliability_base_image = img_u_w_standard


                    reliability_scores = cross_teacher_validator.calculate_cross_teacher_reliability(
                        reliability_base_image,
                        model_ema_general,
                        model_ema_sonar,
                        reliability_augmenter,
                        teacher_mode
                    )


                    reliable_mask = cross_teacher_validator.get_reliability_mask(reliability_scores)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)

            loss_x = criterion_l(pred_x, mask_x)


            if teacher_mode == 'supervised':

                loss_u_s = torch.tensor(0.0).cuda()
                loss = loss_x
            else:

                mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
                ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]

                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
                ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]


                reliable_mask_cutmixed1 = reliable_mask.clone()
                reliable_mask_cutmixed2 = reliable_mask.clone()

                cutmix_box1_expanded = cutmix_box1.unsqueeze(1)
                cutmix_box2_expanded = cutmix_box2.unsqueeze(1)
                reliable_mask_cutmixed1[cutmix_box1_expanded == 1] = reliable_mask.flip(0)[cutmix_box1_expanded == 1]
                reliable_mask_cutmixed2[cutmix_box2_expanded == 1] = reliable_mask.flip(0)[cutmix_box2_expanded == 1]


                reliability_scores_cutmixed1 = reliability_scores.clone()
                reliability_scores_cutmixed2 = reliability_scores.clone()
                reliability_scores_cutmixed1[cutmix_box1_expanded == 1] = reliability_scores.flip(0)[cutmix_box1_expanded == 1]
                reliability_scores_cutmixed2[cutmix_box2_expanded == 1] = reliability_scores.flip(0)[cutmix_box2_expanded == 1]

                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)

                reliability_threshold = 0.3


                reliability_weights1 = torch.where(
                    reliability_scores_cutmixed1 >= reliability_threshold,
                    reliability_scores_cutmixed1,
                    torch.zeros_like(reliability_scores_cutmixed1)
                )


                valid_mask1 = (reliability_weights1 > 0) & (ignore_mask_cutmixed1 != 255)
                loss_u_s1 = loss_u_s1 * reliability_weights1 * valid_mask1.float()
                loss_u_s1 = loss_u_s1.sum() / (valid_mask1.sum().item() + 1e-8)

                loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)

                reliability_weights2 = torch.where(
                    reliability_scores_cutmixed2 >= reliability_threshold,
                    reliability_scores_cutmixed2,
                    torch.zeros_like(reliability_scores_cutmixed2)
                )


                valid_mask2 = (reliability_weights2 > 0) & (ignore_mask_cutmixed2 != 255)
                loss_u_s2 = loss_u_s2 * reliability_weights2 * valid_mask2.float()
                loss_u_s2 = loss_u_s2.sum() / (valid_mask2.sum().item() + 1e-8)

                loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
                loss = (loss_x + loss_u_s) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())


            if teacher_mode == 'supervised':
                mask_ratio = 0.0
            else:

                reliability_threshold = 0.3
                reliable_pixels = (reliability_scores >= reliability_threshold) & (ignore_mask != 255)
                mask_ratio = reliable_pixels.sum().item() / (ignore_mask != 255).sum().item()
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']


            if teacher_mode != 'supervised':
                ema_ratio = min(1 - 1 / (iters + 1), 0.996)

                if teacher_mode == 'general':

                    for param, param_ema in zip(model.parameters(), model_ema_general.parameters()):
                        param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
                    for buffer, buffer_ema in zip(model.buffers(), model_ema_general.buffers()):
                        buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
                elif teacher_mode == 'sonar':

                    for param, param_ema in zip(model.parameters(), model_ema_sonar.parameters()):
                        param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
                    for buffer, buffer_ema in zip(model.buffers(), model_ema_sonar.buffers()):
                        buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg,
                                            total_loss_s.avg, total_mask_ratio.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'


        torch.cuda.empty_cache()


        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)


        mIoU_ema_general, iou_class_ema_general = evaluate(model_ema_general, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema_sonar, iou_class_ema_sonar = evaluate(model_ema_sonar, valloader, eval_mode, cfg, multiplier=14)


        if mIoU_ema_general >= mIoU_ema_sonar:
            mIoU_ema, iou_class_ema = mIoU_ema_general, iou_class_ema_general
            best_teacher = 'general'
        else:
            mIoU_ema, iou_class_ema = mIoU_ema_sonar, iou_class_ema_sonar
            best_teacher = 'sonar'


        torch.cuda.empty_cache()

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'General EMA: {:.2f}, Sonar EMA: {:.2f}, Best EMA: {:.2f} ({})'.format(
                                cls_idx, CLASSES[cfg['dataset']][cls_idx], iou,
                                iou_class_ema_general[cls_idx], iou_class_ema_sonar[cls_idx],
                                iou_class_ema[cls_idx], best_teacher))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, General EMA: {:.2f}, '
                        'Sonar EMA: {:.2f}, Best EMA: {:.2f} ({})'.format(
                            eval_mode, mIoU, mIoU_ema_general, mIoU_ema_sonar, mIoU_ema, best_teacher))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema_general', mIoU_ema_general, epoch)
            writer.add_scalar('eval/mIoU_ema_sonar', mIoU_ema_sonar, epoch)
            writer.add_scalar('eval/mIoU_ema_best', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema_general' % (CLASSES[cfg['dataset']][i]), iou_class_ema_general[i], epoch)
                writer.add_scalar('eval/%s_IoU_ema_sonar' % (CLASSES[cfg['dataset']][i]), iou_class_ema_sonar[i], epoch)

        is_best = mIoU >= previous_best

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
                'model_ema_sonar': model_ema_sonar.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema,
                'teacher_mode': teacher_mode,
                'best_teacher': best_teacher
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
