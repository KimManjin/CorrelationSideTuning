import os
import sys
import time
import argparse
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import torch.optim as optim
import numpy as np
from utils.utils import init_distributed_mode, AverageMeter, ListMeter, reduce_tensor, accuracy, correct_per_class, log_model_info, gpu_mem_usage
from utils.logger import setup_logger

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap
import wandb
import datetime
import shutil
from contextlib import suppress



from modules.video_clip import video_header
from utils.Augmentation import get_augmentation, multiple_samples_collate 
from utils.solver import _lr_scheduler


class VideoCLIP(nn.Module):
    def __init__(self, visual_model, fusion_model, config, args) :
        super(VideoCLIP, self).__init__()
        if 'dino' in args.config:
            self.visual = visual_model
        elif 'mae' in args.config:
            self.visual = visual_model
        else:   
            self.visual = visual_model.visual
        self.fusion_model = fusion_model
        self.n_seg = config.data.num_segments
        self.drop_out = nn.Dropout(p=config.network.drop_fc)
        self.fc = nn.Linear(config.network.n_emb, config.data.num_classes)

    def forward(self, image):
        bt = image.size(0)
        b = bt // self.n_seg
        image_emb = self.visual(image)
        if image_emb.size(0) != b: # no joint_st
            image_emb = image_emb.view(b, self.n_seg, -1)
            image_emb = self.fusion_model(image_emb)

        image_emb = self.drop_out(image_emb)
        logit = self.fc(image_emb)
        return logit

def epoch_saving(epoch, model, optimizer, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename


def update_dict(dict):
    def _rename_param(key):
        # This regex matches keys of the form:
        #   <prefix>selfy_layers<digits>.<layer_index><rest>
        # For example, it matches:
        #   visual.side_network.selfy_layers2.0.stss_extraction.conv0.1.weight
        # where:
        #   prefix = "visual.side_network."
        #   digits = "2"   (if present; if missing, treat as empty)
        #   layer_index = "0"
        #   rest = ".stss_extraction.conv0.1.weight"
        pattern = r'^(.*?)(selfy_layers)(\d*)\.(\d+)(\..+)$'
        m = re.match(pattern, key)
        if m:
            prefix = m.group(1)
            digits = m.group(3)
            layer_index = m.group(4)
            rest = m.group(5)
            # If digits is empty (i.e. key was 'selfy_layers' only), then use 0.
            new_enc_index = str(int(digits) - 1) if digits else "0"
            # Construct the new key:
            # (visual.side_network.selfy_layers<index>.<layer_index>.<...>)
            # -> (visual.side_network.moss_layers.<layer_index>.stss_encoders.<index-1>.<...>)
            new_key = f"{prefix}moss_layers.{layer_index}.stss_encoders.{new_enc_index}{rest}"
            return new_key
        else:
            return key
    new_dict = {}
    for k, v in dict.items():
        new_k = k.replace('module.', '')
        new_k = _rename_param(new_k)
        new_dict[new_k] = v
        if new_k != k:
            print(f"Renaming parameter: {k} -> {new_k}")
    return new_dict

def get_parser():
    parser = argparse.ArgumentParser('CLIP4Time training and evaluation script for video classification', add_help=False)
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--exp_name', default='default', type=str, help='experiment name')
    parser.add_argument('--root_dir', default='./exp', type=str, help='root directory for storing the experiment data')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )      
    args = parser.parse_args()
    return args




def main(args):
    global best_prec1
    """ Training Program """
    init_distributed_mode(args)
    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)

    # set up working directory
    root_dir = args.root_dir
    working_dir = os.path.join(root_dir, args.exp_name)

    if 'something' in config.data.dataset:
        from datasets.sth import Video_dataset
    elif 'diving' in config.data.dataset:
        from datasets.diving48 import Video_dataset
    elif 'finegym' in config.data.dataset:
        from datasets.finegym import Video_dataset
    else:
        from datasets.kinetics import Video_dataset

    if dist.get_rank() == 0:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir)
        shutil.copy('train_vision.py', working_dir)
        if 'eva' in args.config:
            shutil.copy('models/eva_clip/eva_vit_model.py', working_dir)
        elif 'dino' in args.config:
            shutil.copy('models/dino/vision_transformer.py', working_dir)
        elif 'mae' in args.config:
            shutil.copy('models/mae/models_vit.py', working_dir)
        else:
            shutil.copy('models/clip/model.py', working_dir)


    # build logger. If True, use Wandb to logging
    logger = setup_logger(output=working_dir,
                          distributed_rank=dist.get_rank(),
                          name=__name__)
    config.wandb.group_name = args.exp_name
    config.wandb.exp_name = os.path.join(args.exp_name, 'train')
    if dist.get_rank() == 0 and config.wandb.use_wandb and not args.debug:
        wandb.login(key=config.wandb.key)
        wandb.init(project=config.wandb.project_name,
                   name=config.wandb.exp_name,
                   group=config.wandb.group_name,
                   entity=config.wandb.entity,
                   job_type="train")

    # print env and config    
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))




    device = "cpu"
    if torch.cuda.is_available():        
        device = "cuda"
        cudnn.benchmark = True

    # fix the seed for reproducibility
    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


    # get fp16 model and weight
    model_name = config.network.arch
    if model_name in ["EVA02-CLIP-L-14", "EVA02-CLIP-L-14-336", "EVA02-CLIP-bigE-14", "EVA02-CLIP-bigE-14-plus"]:
        # TODO: add MOSS model
        # TODO: modify to take config argument
        # get evaclip model start ########
        weight_path = {
            "EVA02-CLIP-L-14": './pretrain/clip/EVA02_CLIP_L_psz14_s4B.pt',
            "EVA02-CLIP-L-14-336": './pretrain/clip/EVA02_CLIP_L_336_psz14_s6B.pt',
            "EVA02-CLIP-bigE-14":'./pretrain/clip/EVA02_CLIP_E_psz14_s4B.pt',
            "EVA02-CLIP-bigE-14-plus":'./pretrain/clip/EVA02_CLIP_E_psz14_plus_s9B.pt'
        }
        from models.eva_clip import create_model_and_transforms
        model, _, preprocess = create_model_and_transforms(model_name, pretrained=weight_path[model_name], force_custom_clip=True, T=config.data.num_segments, side_dim=config.network.side_dim)
        model_state_dict = model.state_dict()
        # get evaclip model end ########    
    elif model_name in ['DINO-ViT-B-16']:
        weight_path = {
            "DINO-ViT-B-16":'./pretrain/dino/dino_vitbase16_pretrain.pth'
        }
        from models.dino.build import build_model_from_checkpoints
        model = build_model_from_checkpoints(config, pretrained=weight_path[model_name])
        model_state_dict = model.state_dict()
    elif model_name in ['MAE-ViT-B-16']:
        weight_path = {
            "MAE-ViT-B-16":'./pretrain/mae/mae_pretrain_vit_base.pth'
        }
        from models.mae.build import build_model_from_checkpoints
        model = build_model_from_checkpoints(config, pretrained=weight_path[model_name])
        model_state_dict = model.state_dict()
    else:
        # get fp16 model and weight
        import models.clip as clip
        model, model_state_dict = clip.load(
            config,
            device='cpu',
            jit=False,
            download_root='./pretrain/clip') # Must set jit=False for training  ViT-B/32

    if args.precision == "amp" or args.precision == "fp32":
        model = model.float()

    video_head = video_header(
        config.network.sim_header,
        model_state_dict)
    model_onehot = VideoCLIP(model, video_head, config, args)

    # freeze model
    if config.network.my_fix_clip:
        for name, param in model_onehot.named_parameters():
            if 'corr' not in name and 'moss' not in name and 'visual' in name and 'side' not in name and 'ln_post' not in name and 'visual.proj' not in name or 'logit_scale' in name:
                param.requires_grad = False
                logger.info(name + ' False')
            else:
                param.requires_grad = True
                logger.info(name +' True')
            
    flops, params, tunable_params = None, 0.0, 0.0
    if dist.get_rank() == 0:
        flops, params, tunable_params = log_model_info(model_onehot, config, use_train_input=True)


    # Transform functions
    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)
    logger.info('train transforms: {}'.format(transform_train.transforms))
    logger.info('val transforms: {}'.format(transform_val.transforms))
    
    train_data = Video_dataset(
        config.data.train_root, config.data.train_list,
        config.data.label_list, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
        transform=transform_train, dense_sample=config.data.dense, num_sample=config.data.num_sample)

    ################ Few-shot data for training ###########
    if config.data.shot:
        cls_dict = {}
        for item  in train_data.video_list:
            if item.label not in cls_dict:
                cls_dict[item.label] = [item]
            else:
                cls_dict[item.label].append(item)
        import random
        select_vids = []
        K = config.data.shot
        for category, v in cls_dict.items():
            slice = random.sample(v, K)
            select_vids.extend(slice)
        n_repeat = len(train_data.video_list) // len(select_vids)
        train_data.video_list = select_vids * n_repeat
        # print('########### number of videos: {} #########'.format(len(select_vids)))
    ########################################################
    if config.data.num_sample > 1:
        collate_func = multiple_samples_collate
    else:
        collate_func = None


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)                       
    train_loader = DataLoader(train_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=train_sampler, drop_last=False, collate_fn=collate_func)

    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val, dense_sample=config.data.dense)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size,num_workers=config.data.workers,
        sampler=val_sampler, drop_last=False)

    ############# criterion #############
    mixup_fn = None
    if config.solver.mixup:
        logger.info("=> Using Mixup")
        from timm.loss import SoftTargetCrossEntropy
        criterion = SoftTargetCrossEntropy()     
        # smoothing is handled with mixup label transform
        from utils.mixup import Mixup
        mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=config.data.num_classes)
        #   
        # from utils.mixup_old import CutmixMixupBlending
        # mixup_fn = CutmixMixupBlending(num_classes=config.data.num_classes, 
        #                                smoothing=0.1, 
        #                                mixup_alpha=0.8, 
        #                                cutmix_alpha=1.0, 
        #                                switch_prob=0.5)   

    elif config.solver.smoothing:
        logger.info("=> Using label smoothing: 0.1")
        from timm.loss import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=config.solver.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            logger.info("=> loading checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            state_dict = update_dict(state_dict)
            if state_dict['fc.weight'].size(0) != config.data.num_classes:
                state_dict.pop('fc.weight')
                state_dict.pop('fc.bias')
                logger.info('=> pop last fc layer')
            model_onehot.load_state_dict(state_dict, strict=False)
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    clip_params = []
    other_params = []
    bn_params = []
    for name, param in model_onehot.named_parameters():
        if 'bn' in name and 'side' in name:
            bn_params.append(param)
        elif 'visual' in name and 'control_point' not in name and 'time_embedding' not in name:
            clip_params.append(param)
        else:
            other_params.append(param)

    if config.network.sync_bn:
        bn_lr = config.solver.lr
    else:
        bn_lr = config.solver.lr / config.solver.grad_accumulation_steps

    optimizer = optim.AdamW([{'params': clip_params, 'lr': config.solver.lr * config.solver.clip_ratio}, 
                            {'params': other_params, 'lr': config.solver.lr},
                            {'params': bn_params, 'lr': bn_lr}],
                            betas=config.solver.betas, lr=config.solver.lr, eps=1e-8,
                            weight_decay=config.solver.weight_decay) 
    
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cpu')
            checkpoint['model_state_dict'] = update_dict(checkpoint['model_state_dict'])
            model_onehot.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            start_epoch = checkpoint['epoch'] + 1
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.pretrain))


    lr_scheduler = _lr_scheduler(config, optimizer)
        
    if args.distributed:
        model_onehot = DistributedDataParallel(model_onehot.cuda(), device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model_onehot.module


    scaler = GradScaler() if args.precision == "amp" else None

    best_prec1 = 0.0
    if config.solver.evaluate:
        logger.info(("===========evaluate==========="))
        prec1 = validate(
            start_epoch,
            val_loader, device, 
            model_onehot, config, logger)
        return



    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)        
        cur_iter = train(model_onehot, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, mixup_fn, logger)

        if (epoch+1) % config.logging.eval_freq == 0 or (epoch+1) > config.solver.epochs - 5:  # and epoch>0
            if config.logging.skip_epoch is not None and epoch in config.logging.skip_epoch:
                continue
            prec1 = validate(epoch, val_loader, device, model_onehot, config, logger, cur_iter)

            if dist.get_rank() == 0:
                is_best = prec1 >= best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1,best_prec1))
                logger.info('Saving:')
                filename = "{}/last_model.pt".format(working_dir)

                epoch_saving(epoch, model_without_ddp, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model_without_ddp, optimizer)


def train(model, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, mixup_fn, logger):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()
    for i,(images, list_id) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])

        if mixup_fn is not None:
            images = images.transpose(1, 2)  # b t c h w -> b c t h w
            images, list_id = mixup_fn(images, list_id)
            images = images.transpose(1, 2)

        list_id = list_id.to(device)
        b,t,c,h,w = images.size()
        images= images.view(-1,c,h,w) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
    
        if (i + 1) % config.solver.grad_accumulation_steps != 0:
            with model.no_sync():
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, list_id)

                    # loss regularization
                    loss = loss / config.solver.grad_accumulation_steps    
                if scaler is not None:
                    # back propagation
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
        else:
            with autocast():
                logits = model(images)
                loss = criterion(logits, list_id)

                # loss regularization
                loss = loss / config.solver.grad_accumulation_steps            

            if scaler is not None:
                # back propagation
                scaler.scale(loss).backward()

                scaler.step(optimizer)  
                scaler.update()  
                optimizer.zero_grad()  # reset gradient
                    
            else:
                # back propagation
                loss.backward()
                optimizer.step()  # update param
                optimizer.zero_grad()  # reset gradient

        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        

        prec1, prec5 = accuracy(logits, list_id, topk=(1, 5))
        top1.update(prec1.item(), logits.size(0))
        top5.update(prec5.item(), logits.size(0))
        losses.update(loss.item(), logits.size(0))


        batch_time.update(time.time() - end)
        end = time.time()                


        cur_iter = epoch * len(train_loader) +  i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))        

        if i % config.logging.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Mem {mem_usage:.2f}GB\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, mem_usage=gpu_mem_usage(),
                             top1=top1,
                             loss=losses, lr=optimizer.param_groups[-1]['lr'])))  # TODO
            if dist.get_rank() == 0 and config.wandb.use_wandb and not args.debug:
                wandb.log({"train/loss": losses.avg,
                        "train/top1": top1.avg,
                        "train/top5": top5.avg,
                        "train/lr": optimizer.param_groups[-1]['lr']},
                        step=cur_iter)
    return cur_iter


def validate(epoch, val_loader, device, model, config, logger, cur_iter=0):
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_per_class = ListMeter(config.data.num_classes)
    top5_per_class = ListMeter(config.data.num_classes)
    model.eval()
    with torch.no_grad():
        for i, (image, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            logits = model(image_input)  # B 400

            # topk accuracy
            prec = accuracy(logits, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            # topk accuracy per class
            if config.logging.acc_per_class:
                correct_k, count = correct_per_class(logits, class_id, topk=(1, 5))
                correct_1_per_class = reduce_tensor(correct_k[0], average=False)
                correct_5_per_class = reduce_tensor(correct_k[1], average=False)
                count_per_class = reduce_tensor(count, average=False)

                top1_per_class.update(correct_1_per_class.cpu(), count_per_class.cpu())
                top5_per_class.update(correct_5_per_class.cpu(), count_per_class.cpu())

            if i % config.logging.print_freq == 0:
                base_log = ('Test: [{0}/{1}]\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), top1=top1, top5=top5))
                if config.logging.acc_per_class:
                    extra_log = ('\tmPrec@1 ({:.3f})\tmPrec@5 ({:.3f})'.format(top1_per_class.mean(), top5_per_class.mean()))
                else:
                    extra_log = ''
                logger.info(base_log + extra_log)
    base_log = 'Overall Prec@1 {:.03f}% Prec@5 {:.03f}%'.format(top1.avg, top5.avg)
    if config.logging.acc_per_class:
        extra_log = ' mPrec@1 ({:.03f}) mPrec@5 ({:.03f})'.format(top1_per_class.mean(), top5_per_class.mean())
    else:
        extra_log = ''
    logger.info(base_log + extra_log)
    if dist.get_rank() == 0 and config.wandb.use_wandb and not args.debug:
        base_log = {"val/top1": top1.avg, "val/top5": top5.avg}
        if config.logging.acc_per_class:
            extra_log = {"val/mtop1": top1_per_class.mean(), "val/mtop5": top5_per_class.mean()}
            base_log.update(extra_log)
        wandb.log(base_log, step=cur_iter)
    
    if 'finegym' in config.data.dataset:
        return top1_per_class.mean()
    else:
        return top1.avg


if __name__ == '__main__':
    args = get_parser() 
    main(args)

