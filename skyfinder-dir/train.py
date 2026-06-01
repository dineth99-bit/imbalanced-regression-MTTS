import os
import time
import argparse
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from scipy.stats import gmean
from torch.utils.data import DataLoader
from tensorboard_logger import Logger
from tqdm import tqdm

from datasets import SkyFinder
from loss import *
from resnet import resnet50
from utils import *

os.environ["KMP_WARNINGS"] = "FALSE"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lds', action='store_true', default=False)
parser.add_argument('--lds_kernel', type=str, default='gaussian', choices=['gaussian', 'triang', 'laplace'])
parser.add_argument('--lds_ks', type=int, default=5)
parser.add_argument('--lds_sigma', type=float, default=2)
parser.add_argument('--fds', action='store_true', default=False)
parser.add_argument('--fds_kernel', type=str, default='gaussian', choices=['gaussian', 'triang', 'laplace'])
parser.add_argument('--fds_ks', type=int, default=5)
parser.add_argument('--fds_sigma', type=float, default=2)
parser.add_argument('--start_update', type=int, default=0)
parser.add_argument('--start_smooth', type=int, default=1)
parser.add_argument('--bucket_num', type=int, default=81)
parser.add_argument('--bucket_start', type=int, default=0)
parser.add_argument('--fds_mmt', type=float, default=0.9)
parser.add_argument('--label_min', type=int, default=-30)
parser.add_argument('--label_max', type=int, default=50)
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'])
parser.add_argument('--retrain_fc', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='skyfinder')
parser.add_argument('--meta_dir', type=str, default='./data', help='directory containing skyfinder.csv')
parser.add_argument('--data_dir', type=str, default='./data/images', help='directory containing camera image folders')
parser.add_argument('--mask_dir', type=str, default='./data', help='root containing skyfinder_masks/')
parser.add_argument('--crop_mode', type=str, default='full', choices=['full', 'sky_bbox', 'sky_masked'])
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--store_root', type=str, default='checkpoint')
parser.add_argument('--store_name', type=str, default='')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=90)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--print_freq', type=int, default=20)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--pretrained', type=str, default='')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--many_shot_thr', type=int, default=50)
parser.add_argument('--low_shot_thr', type=int, default=10)
parser.set_defaults(augment=True)
args, _ = parser.parse_known_args()

args.start_epoch, args.best_loss = 0, 1e5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
if args.retrain_fc:
    args.store_name += f'_retrain_fc'
args.store_name = f"{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"

prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")
print(f"Device: {device}")

tb_logger = Logger(logdir=os.path.join(args.store_root, args.store_name), flush_secs=2) if device.type == 'cuda' else None


def main():
    print('=====> Preparing data...')
    csv_path = os.path.join(args.meta_dir, f"{args.dataset}.csv")
    df = pd.read_csv(csv_path)
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']
    train_labels = df_train['temperature'].round().astype(int)

    common = dict(
        data_dir=args.data_dir, img_size=args.img_size, label_min=args.label_min, label_max=args.label_max,
        crop_mode=args.crop_mode, mask_dir=args.mask_dir,
    )
    train_dataset = SkyFinder(df=df_train, split='train', reweight=args.reweight,
                              lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma, **common)
    val_dataset = SkyFinder(df=df_val, split='val', **common)
    test_dataset = SkyFinder(df=df_test, split='test', **common)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == 'cuda', drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == 'cuda', drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=device.type == 'cuda', drop_last=False)
    print(f"Training: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    print('=====> Building model...')
    model = resnet50(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                     start_update=args.start_update, start_smooth=args.start_smooth,
                     kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)
    if device.type == 'cuda':
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    if args.evaluate:
        assert args.resume
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        validate(test_loader, model, train_labels=train_labels, prefix='Test')
        return

    if args.retrain_fc:
        assert args.reweight != 'none' and args.pretrained
        for name, param in model.named_parameters():
            if 'fc' not in name and 'linear' not in name:
                param.requires_grad = False

    if not args.retrain_fc:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        args.start_epoch = checkpoint['epoch']
        args.best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if device.type == 'cuda':
        cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train_epoch(train_loader, model, optimizer, epoch)
        val_loss_mse, val_loss_l1, val_loss_gmean = validate(val_loader, model, train_labels=train_labels)

        loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print(f"Epoch #{epoch}: Train [{train_loss:.4f}] Val L1 [{val_loss_l1:.4f}] G-Mean [{val_loss_gmean:.4f}]")
        if tb_logger:
            tb_logger.log_value('train_loss', train_loss, epoch)
            tb_logger.log_value('val_loss_l1', val_loss_l1, epoch)

    print("Test best model...")
    checkpoint = torch.load(f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    validate(test_loader, model, train_labels=train_labels, prefix='Test')


def _to_device(batch):
    inputs, targets, weights = batch
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True), weights.to(device, non_blocking=True)


def _fds_targets(targets):
    """Map °C targets to FDS bucket indices."""
    buckets = (targets - args.label_min).round().long().squeeze(1)
    return buckets.clamp(0, args.bucket_num - 1).float().unsqueeze(1)


def train_epoch(train_loader, model, optimizer, epoch):
    model.train()
    losses = AverageMeter('Loss', ':.3f')
    for idx, batch in enumerate(train_loader):
        inputs, targets, weights = _to_device(batch)
        if args.fds:
            outputs, _ = model(inputs, _fds_targets(targets), epoch)
        else:
            outputs = model(inputs, targets, epoch)
        loss = globals()[f"weighted_{args.loss}_loss"](outputs, targets, weights)
        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.fds and epoch >= args.start_update:
        encodings, labels = [], []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc='FDS features'):
                inputs, targets, _ = _to_device(batch)
                _, feature = model(inputs, _fds_targets(targets), epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())
        encodings = torch.from_numpy(np.vstack(encodings)).to(device)
        labels = torch.from_numpy(np.hstack(labels)).to(device)
        fds_module = model.module.FDS if hasattr(model, 'module') else model.FDS
        fds_module.update_last_epoch_stats(epoch)
        fds_module.update_running_stats(encodings, labels, epoch)
    return losses.avg


def validate(val_loader, model, train_labels=None, prefix='Val'):
    model.eval()
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')
    preds, labels_list, losses_all = [], [], []
    losses_mse = AverageMeter('MSE', ':.3f')
    losses_l1 = AverageMeter('L1', ':.3f')

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets, _ = _to_device(batch)
            outputs = model(inputs)
            preds.extend(outputs.data.cpu().numpy())
            labels_list.extend(targets.data.cpu().numpy())
            losses_mse.update(criterion_mse(outputs, targets).item(), inputs.size(0))
            losses_l1.update(criterion_l1(outputs, targets).item(), inputs.size(0))
            losses_all.extend(criterion_gmean(outputs, targets).cpu().numpy())

    preds = np.hstack(preds)
    labels_arr = np.hstack(labels_list)
    shot_dict = shot_metrics(preds, labels_arr, train_labels,
                             many_shot_thr=args.many_shot_thr, low_shot_thr=args.low_shot_thr)
    loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
    print(f" * {prefix} Overall: MSE {losses_mse.avg:.3f} L1 {losses_l1.avg:.3f} G-Mean {loss_gmean:.3f}")
    for name, key in [('Many', 'many'), ('Median', 'median'), ('Low', 'low')]:
        print(f" * {prefix} {name}: L1 {shot_dict[key]['l1']:.3f} G-Mean {shot_dict[key]['gmean']:.3f}")
    return losses_mse.avg, losses_l1.avg, loss_gmean


def shot_metrics(preds, labels, train_labels, many_shot_thr=50, low_shot_thr=10):
    train_labels = np.array(train_labels).astype(int)
    preds = preds.reshape(-1)
    labels = labels.reshape(-1).astype(int)

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    for key, mse_l, l1_l, g_l, cnt in [
        ('many', many_shot_mse, many_shot_l1, many_shot_gmean, many_shot_cnt),
        ('median', median_shot_mse, median_shot_l1, median_shot_gmean, median_shot_cnt),
        ('low', low_shot_mse, low_shot_l1, low_shot_gmean, low_shot_cnt),
    ]:
        if sum(cnt) == 0:
            shot_dict[key] = {'mse': float('nan'), 'l1': float('nan'), 'gmean': float('nan')}
        else:
            shot_dict[key]['mse'] = np.sum(mse_l) / np.sum(cnt)
            shot_dict[key]['l1'] = np.sum(l1_l) / np.sum(cnt)
            shot_dict[key]['gmean'] = gmean(np.hstack(g_l), axis=None).astype(float) if g_l else float('nan')
    return shot_dict


if __name__ == '__main__':
    main()
