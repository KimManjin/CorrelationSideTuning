"""
utils for clip
"""
import os

import torch
import torch.distributed as dist
import torch.distributed.nn as distnn
from torch import nn
import torch.nn.functional as F
import numpy as np
import json
import math
from utils.logger import get_logger

from collections import defaultdict
from typing import Any, Counter, DefaultDict, Dict, Optional, Tuple, Union
from fvcore.nn import FlopCountAnalysis

logger = get_logger(__name__)

def init_distributed_mode(args):
    """ init for distribute mode """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
        
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    '''
    This is commented due to the stupid icoding pylint checking.
    print('distributed init rank {}: {}'.format(args.rank, args.dist_url), flush=True)
    '''
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()


def ddp_all_reduce(*args):
    """ all reduce (op: sum) by ddp """
    t = torch.tensor([x for x in args], dtype=torch.float64, device='cuda')
    dist.barrier()
    dist.all_reduce(t)
    t = t.tolist()
    return t


def ddp_all_gather(*args):
    """ all gather by ddp, all gather don't have grad_fn by default """
    rets = []
    world_size = dist.get_world_size()
    for x in args:
        if type(x) is torch.Tensor:
            ret = [torch.zeros_like(x) for _ in range(world_size)]
            dist.barrier()
            dist.all_gather(ret, x)
        else:  # for any picklable object
            ret = [None for _ in range(world_size)]
            dist.barrier()
            dist.all_gather_object(ret, x)
        rets.append(ret)
    return rets if len(rets) > 1 else rets[0]




def gather_labels(labels):
    # We gather tensors from all gpus
    gathered_labels = ddp_all_gather(labels)
    all_labels = torch.cat(gathered_labels)
    return all_labels


# def gen_label(labels):
#     num = len(labels)
#     gt = np.zeros(shape=(num,num))
#     for i, label in enumerate(labels):
#         for k in range(num):
#             if labels[k] == label:
#                 gt[i,k] = 1
#     return gt

def gen_label(labels):
    num = len(labels)
    gt = torch.zeros(size=(num, num))
    labels_column = labels.reshape(-1, 1).repeat(1, num)
    labels_row = labels.repeat(num, 1)
    gt[labels_column == labels_row] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    # print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()



def gather_features(
        image_features, text_features,
        local_loss=False, gather_with_grad=False, rank=0, world_size=1):

    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(distnn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(distnn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
    return all_image_features, all_text_features



def create_logits(image_features, text_features, logit_scale, local_loss=False):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    if dist.get_world_size() > 1:
        all_image_features, all_text_features = gather_features(
            image_features, text_features,
            local_loss=local_loss, gather_with_grad=False, 
            rank=dist.get_rank(), world_size=dist.get_world_size())
            
        # cosine similarity as logits
        if local_loss:
            logits_per_image = logit_scale * image_features @ all_text_features.T
            logits_per_text = logit_scale * text_features @ all_image_features.T
        else:
            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T   

    else:
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T                 

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


def create_ds_config(args, working_dir, config):
    #args.deepspeed_config = os.path.join(working_dir, "deepspeed_config.json")
    args.deepspeed_config = "deepspeed_config.json"
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": config.data.batch_size * config.solver.grad_accumulation_steps * dist.get_world_size(),
            "train_micro_batch_size_per_gpu": config.data.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": config.solver.lr,
                    "weight_decay": config.solver.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def epoch_saving(epoch, model, video_head, optimizer, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'fusion_model_state_dict': video_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model, video_head, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': video_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename


def reduce_tensor(tensor, n=None, average=True):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt = rt / n
    return rt


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


class ListMeter:
    """Computes and stores the average and current values in torch.tensor (list)"""
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.val = torch.zeros(self.length)
        self.avg = torch.zeros(self.length) 
        self.sum = torch.zeros(self.length)
        self.count = torch.zeros(self.length)

    def update(self, val, n=1):
        """
        Args:
            val: torch.tensor of shape (length,) containing values for each item
            n: int or torch.tensor of shape (length,) containing sample counts
        """
        if isinstance(n, int):
            n = torch.full_like(val, n)
        
        assert val.shape == (self.length,), f"val must have shape ({self.length},), got {val.shape}"
        assert n.shape == (self.length,), f"n must have shape ({self.length},), got {n.shape}"
        
        self.val = val
        self.sum += val
        self.count += n
        self.avg = torch.where(self.count > 0, (self.sum / self.count) * 100, 0.)

    def mean(self):
        return torch.where(self.count > 0, self.avg, 0.).mean()
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        self.val = reduce_tensor(self.val.cuda(), world_size, average=False).cpu()
        self.sum = reduce_tensor(self.sum.cuda(), world_size, average=False).cpu()
        self.count = reduce_tensor(self.count.cuda(), world_size, average=False).cpu()
        self.avg = self.sum / self.count


def accuracy_epic_kitchens(verb_output, noun_output, verb_target, noun_target, topk=(1,)):
    """Computes the action accuracy for Epic-Kitchens where both verb and noun need to be correct
    """
    maxk = max(topk)
    batch_size = verb_target.size(0)
    
    # Verb top-k accuracy
    _, verb_pred = verb_output.topk(maxk, 1, True, True)  # (B, maxk)
    verb_pred = verb_pred.t()
    verb_target = verb_target.view(1, -1).expand_as(verb_pred)  # (B, maxk)
    verb_correct = verb_pred.eq(verb_target)  # (B, maxk)
    # Noun top-k accuracy
    _, noun_pred = noun_output.topk(maxk, 1, True, True)  # (B, maxk)
    noun_pred = noun_pred.t()
    noun_target = noun_target.view(1, -1).expand_as(noun_pred)  # (B, maxk)
    noun_correct = noun_pred.eq(noun_target)  # (B, maxk)
    # Action top-k accuracy
    action_correct = verb_correct & noun_correct  # (B, maxk)
    
    # Calculate top-k accuracies
    action_topk_acc = []
    verb_topk_acc = []
    noun_topk_acc = []  
    for k in topk:
        # For each k, check if any of the top-k predictions are correct
        action_correct_k = action_correct[:, :k].reshape(-1).float().sum(0)
        verb_correct_k = verb_correct[:, :k].reshape(-1).float().sum(0)
        noun_correct_k = noun_correct[:, :k].reshape(-1).float().sum(0)
        action_topk_acc.append(action_correct_k.mul_(100.0 / batch_size))
        verb_topk_acc.append(verb_correct_k.mul_(100.0 / batch_size))
        noun_topk_acc.append(noun_correct_k.mul_(100.0 / batch_size))
    return action_topk_acc, verb_topk_acc, noun_topk_acc

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # Top-k accuracy
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    topk_acc = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        topk_acc.append(correct_k.mul_(100.0 / batch_size))
    return topk_acc

def correct_per_class(output, target, topk=(1, )):
    """
    Get the number of correct predictions per class.
    """
    maxk = max(topk)
    batch_size = target.size(0)
    # one-hot label encoding
    one_hot_target = F.one_hot(target, num_classes=output.size(1))
    one_hot_target_k = one_hot_target.unsqueeze(0).repeat(max(topk), 1, 1)
    num_samples_per_class = one_hot_target.sum(dim=0).float()
    # top-k prediction
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).unsqueeze(-1)
    correct_k = one_hot_target_k * correct
    # topk correct per class
    topk_correct = []
    for k in topk:
        correct_k_per_class = correct_k[:k].float().sum(dim=1).sum(dim=0)
        topk_correct.append(torch.where(num_samples_per_class > 0,
                                        correct_k_per_class,
                                        0.0)
        )
    per_class_acc_info = (topk_correct, num_samples_per_class)
    return per_class_acc_info

def accuracy_per_sample(output, target):
    """
    Get per-sample prediction correctness
    """
    pred = output.argmax(dim=1)
    correct = (pred == target).float()
    return correct

def accuracy_per_sample_epic_kitchens(verb_output, noun_output, verb_target, noun_target):
    """
    Get per-sample prediction correctness for Epic-Kitchens where both verb and noun need to be correct
    """
    verb_pred = verb_output.argmax(dim=1)
    noun_pred = noun_output.argmax(dim=1)
    verb_correct = (verb_pred == verb_target)
    noun_correct = (noun_pred == noun_target)
    action_correct = (verb_correct & noun_correct)
    return action_correct.float(), verb_correct.float(), noun_correct.float()

from torchnet import meter
def mean_average_precision(probs, labels):
    """Computes MAP for ActivityNet evaluation"""
    if not isinstance(probs, torch.Tensor):
        probs = torch.Tensor(probs).cuda()
    
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels).long().cuda()

    gt = torch.zeros_like(probs).int()
    acc_meter = meter.ClassErrorMeter(topk=[1, 3], accuracy=True)
    gt[torch.LongTensor(range(gt.size(0))), labels] = 1
    acc_meter.add(probs, labels)
    acc = acc_meter.value()

    probs = torch.nn.functional.softmax(probs, dim=1)
    
    map_meter = meter.mAPMeter()
    map_meter.add(probs, gt)
    ap = map_meter.value()
    ap = float(ap) * 100
    return [torch.tensor(acc[0]).cuda(), torch.tensor(ap).cuda()]


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    params = np.sum([p.numel() for p in model.parameters()]).item()
    tunable_params = np.sum([p.numel() for p in model.parameters() if p.requires_grad]).item()
    return params, tunable_params


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def get_flops(model, config, use_train_input=True):
    """
    Compute the number of FLOPs.
    Args:
        model (model): model to compute the FLOPs.
        config (CfgNode): configs.
        use_train_input (bool): if True, use the training input size.
    
    Returns:
        flops (float): number of FLOPs.
    """
    model_mode = model.training
    model.eval()
    input_size = config.data.input_size
    if not use_train_input:
        input_size = config.data.input_size
    if not isinstance(input_size, list):
        input_size = [input_size, input_size]
    input_size = [config.data.num_segments, 3] + input_size
    input_data = torch.randn(*input_size)
    flops = FlopCountAnalysis(model, input_data)
    model.train(model_mode)
    return flops
    

def log_model_info(model, config, use_train_input=True):
    """
    Log info, includes number of parameters, gpu usage, gflops and activation count.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
        config (CfgNode): configs.
        use_train_input (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    def _human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.3f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    logger.info("Model:\n{}".format(model))
    flops = get_flops(model, config, use_train_input)
    logger.info("Flops: {}".format(_human_format(flops.total())))
    params, tunable_params = params_count(model)
    logger.info("Params: {}, tunable Params: {}".format(_human_format(params), _human_format(tunable_params)))
    return flops, params, tunable_params


if __name__=='__main__':
    probs = torch.load('ANet_similarity_336.pth')        # similarity logits
    labels = torch.load('ANet_labels_336.pth')       # class ids

    mAP = mean_average_precision(probs, labels)
    print(mAP)

