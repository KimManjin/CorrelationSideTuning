import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import time
from utils.utils import init_distributed_mode, AverageMeter, ListMeter, reduce_tensor, accuracy, correct_per_class, log_model_info
from utils.logger import setup_logger
import clip

import yaml
from dotmap import DotMap
import wandb

from datasets.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample
from modules.video_clip import video_header

class VideoCLIP(nn.Module):
    def __init__(self, clip_model, fusion_model, config) :
        super(VideoCLIP, self).__init__()
        self.visual = clip_model.visual
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

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--root_dir', default='./exp', type=str, help='root directory for storing the experiment data')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )                        
    parser.add_argument('--test_crops', type=int, default=1)   
    parser.add_argument('--test_clips', type=int, default=1) 
    parser.add_argument('--dense', default=False, action="store_true",
                    help='use multiple clips for test')                     
    args = parser.parse_args()
    return args

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict


def main(args):
    init_distributed_mode(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    if 'something' in config['data']['dataset']:
        from datasets.sth import Video_dataset
    else:
        from datasets.kinetics import Video_dataset

    config = DotMap(config)

    # set up working directory
    working_dir = "/".join(args.weights.split('/')[:-1])
    exp_name = args.weights.split('/')[-2]
    # build logger. If True, use Wandb to logging
    logger = setup_logger(output=working_dir,
                          distributed_rank=dist.get_rank(),
                          name=__name__)
    config.wandb.group_name = exp_name
    config.wandb.exp_name = os.path.join(exp_name, 'test')
    if dist.get_rank() == 0 and config.wandb.use_wandb and not args.debug:
        wandb.login(key=config.wandb.key)
        wandb.init(project=config.wandb.project_name,
                   name=config.wandb.exp_name,
                   group=config.wandb.group_name,
                   entity=config.wandb.entity,
                   job_type="test")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    # get fp16 model and weight
    model_name = config.network.arch
    if model_name in ["EVA02-CLIP-L-14", "EVA02-CLIP-L-14-336", "EVA02-CLIP-bigE-14", "EVA02-CLIP-bigE-14-plus"]:
        # TODO: add SELFY model
        # TODO: modify to take config argument
        # get evaclip model start ########
        weight_path = {
            "EVA02-CLIP-L-14": './clip-pretrain/EVA02_CLIP_L_psz14_s4B.pt',
            "EVA02-CLIP-L-14-336": './clip-pretrain/EVA02_CLIP_L_336_psz14_s6B.pt',
            "EVA02-CLIP-bigE-14":'./clip-pretrain/EVA02_CLIP_E_psz14_s4B.pt',
            "EVA02-CLIP-bigE-14-plus":'./clip_pretrain/EVA02_CLIP_E_psz14_plus_s9B.pt'
        }
        from eva_clip import create_model_and_transforms
        model, _, preprocess = create_model_and_transforms(model_name, pretrained=weight_path[model_name], force_custom_clip=True, T=config.data.num_segments, side_dim=config.network.side_dim)
        clip_state_dict = model.state_dict()
        # get evaclip model end ########    
    else:
        # get fp16 model and weight
        import clip
        model, clip_state_dict = clip.load(
            config,
            device='cpu',
            jit=False,
            download_root='./clip_pretrain') # Must set jit=False for training  ViT-B/32

    video_head = video_header(
        config.network.sim_header,
        clip_state_dict)

    if args.precision == "amp" or args.precision == "fp32":
        model = model.float()


    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    # rescale size
    if 'something' in config.data.dataset:
        scale_size = (256, 320) 
    else:
        scale_size = 256 if config.data.input_size == 224 else config.data.input_size

    # crop size
    input_size = config.data.input_size

    # control the spatial crop
    if args.test_crops == 1: # one crop
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 3 crops (left right center)
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                )
        ])
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))


    val_data = Video_dataset(       
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        test_mode=True,
        transform=torchvision.transforms.Compose([
            cropping,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean,input_std),
        ]),
        dense_sample=args.dense,
        test_clips=args.test_clips)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
        batch_size=config.data.test_batch_size,num_workers=config.data.workers,
        sampler=val_sampler, pin_memory=True, drop_last=False)


    model_full = VideoCLIP(model, video_head, config)
    flops, params, tunable_params = None, 0.0, 0.0
    if dist.get_rank() == 0:
        flops, params, tunable_params = log_model_info(model_full, config, use_train_input=False)
    

    if os.path.isfile(args.weights):
        checkpoint = torch.load(args.weights, map_location='cpu')
        if dist.get_rank() == 0:
            logger.info('load model: epoch {}'.format(checkpoint['epoch']))

        model_full.load_state_dict(update_dict(checkpoint['model_state_dict']))
        del checkpoint

    if args.distributed:
        model_full = DistributedDataParallel(model_full.cuda(), device_ids=[args.gpu], find_unused_parameters=True)

    prec1, (top1_per_class, top5_per_class) = validate(
        val_loader, device, 
        model_full, config, args.test_crops, args.test_clips, logger=logger)
    
    if config.logging.acc_per_class:
            # Save per-class accuracies to file
            save_path = os.path.join(working_dir, 'per_class_accuracies.txt')
            with open(save_path, 'w') as f:
                f.write('Class\tTop1\tTop5\n')
                for i in range(len(top1_per_class)):
                    f.write(f'{i}\t{top1_per_class[i]:.2f}\t{top5_per_class[i]:.2f}\n')
            logger.info(f'Per-class accuracies saved to {save_path}')
    return



def validate(val_loader, device, model, config, test_crops, test_clips, logger=None):
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_per_class = ListMeter(config.data.num_classes)
    top5_per_class = ListMeter(config.data.num_classes)
    model.eval()
    proc_start_time = time.time()

    with torch.no_grad():
        for i, (image, class_id) in enumerate(val_loader):
            batch_size = class_id.numel()
            num_crop = test_crops

            num_crop *= test_clips  # 4 clips for testing when using dense sample

            class_id = class_id.to(device)
            n_seg = config.data.num_segments
            image = image.view((-1, n_seg, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            image_input = image.to(device).view(-1, c, h, w)


            logits = model(image_input)  # bt n_class


            cnt_time = time.time() - proc_start_time

            logits = logits.view(batch_size, -1, logits.size(1)).softmax(dim=-1)
            logits = logits.mean(dim=1, keepdim=False)      # bs n_class

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
    
            if i % config.logging.print_freq == 0 and dist.get_rank() == 0:
                runtime = float(cnt_time) / (i + 1) / (batch_size * dist.get_world_size())
                base_msg = ('Test: [{0}/{1}], average {runtime:.4f} sec/video \t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                               i, len(val_loader), runtime=runtime, top1=top1, top5=top5)
                if True:
                    top1_per_class_mean = top1_per_class.mean()
                    top5_per_class_mean = top5_per_class.mean()
                    base_msg += '\tmPrec@1 {:.3f}\tmPrec@5 {:.3f}'.format(
                        top1_per_class_mean, top5_per_class_mean)
                
                logger.info(base_msg)


    if dist.get_rank() == 0:
        logger.info('-----Evaluation is finished------')
        base_msg = 'Overall Prec@1 {:.03f}% Prec@5 {:.03f}%'.format(top1.avg, top5.avg)
        if config.logging.acc_per_class:
            extra_msg = '\tmPrec@1 ({:.3f})\tmPrec@5 ({:.3f})'.format(top1_per_class.mean(), top5_per_class.mean())
        else:
            extra_msg = ''
        logger.info(base_msg + extra_msg)
        if config.wandb.use_wandb and not args.debug:
            base_log = {"test/top1": top1.avg, "test/top5": top5.avg}
            if config.logging.acc_per_class:
                extra_log = {"test/mtop1": top1_per_class.mean(), "test/mtop5": top5_per_class.mean()}
                base_log.update(extra_log)
            wandb.log(base_log, step=0)

    return top1.avg, (top1_per_class.avg, top5_per_class.avg)



if __name__ == '__main__':
    args = get_parser() 
    main(args)

