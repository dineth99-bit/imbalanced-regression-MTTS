"""Train PRIME (+ optional PRW) on AgeDB-DIR. Hyperparams follow Lim et al. (ICML 2025) Table 11."""
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from scipy.stats import gmean
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import AgeDB
from loss import weighted_l1_loss
from prime import ProxyBank, alignment_loss, proxy_loss
from resnet import resnet50
from utils import (
    AverageMeter,
    ProgressMeter,
    adjust_learning_rate,
    prepare_folders,
    save_checkpoint,
)

os.environ["KMP_WARNINGS"] = "FALSE"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--store_root", type=str, default="checkpoint")
parser.add_argument("--store_name", type=str, default="md_prime_prw")
parser.add_argument("--reweight", type=str, default="sqrt_inv", choices=["none", "sqrt_inv", "inverse"])
parser.add_argument("--num_proxies", type=int, default=20)
parser.add_argument("--lambda_p", type=float, default=10.0)
parser.add_argument("--lambda_a", type=float, default=50.0)
parser.add_argument("--tau_f", type=float, default=10.0)
parser.add_argument("--tau_t", type=float, default=2.0)
parser.add_argument("--alpha", type=float, default=0.0005)
parser.add_argument("--prw", action="store_true", default=True)
parser.add_argument("--prw_delta", type=float, default=0.05)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--epoch", type=int, default=90)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--schedule", type=int, nargs="*", default=[60, 80])
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--print_freq", type=int, default=50)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--evaluate", action="store_true")
args = parser.parse_args()

args.start_epoch, args.best_loss = 0, 1e5
args.store_name = (
    f"agedb_resnet50_{args.store_name}_{args.reweight}_prime"
    f"_c{args.num_proxies}_adam_l1_{args.lr}_{args.batch_size}"
)
prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, "training.log")),
        logging.StreamHandler(),
    ],
)
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")


def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    from collections import defaultdict

    train_labels = np.array(train_labels).astype(int)
    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    buckets = {"many": [], "median": [], "low": []}
    cnts = {"many": [], "median": [], "low": []}
    gm = {"many": [], "median": [], "low": []}
    for i in range(len(train_class_count)):
        key = "many" if train_class_count[i] > many_shot_thr else "low" if train_class_count[i] < low_shot_thr else "median"
        buckets[key].append(l1_per_class[i])
        cnts[key].append(test_class_count[i])
        gm[key] += list(l1_all_per_class[i])

    out = defaultdict(dict)
    for key in ("many", "median", "low"):
        out[key]["l1"] = np.sum(buckets[key]) / np.sum(cnts[key])
        out[key]["mse"] = out[key]["l1"]  # unused
        out[key]["gmean"] = gmean(np.hstack(gm[key]), axis=None).astype(float)
    return out


def validate(val_loader, model, proxies, train_labels, prefix="Val"):
    model.eval()
    proxies.eval()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction="none")
    losses_l1 = AverageMeter("Loss (L1)", ":.3f")
    losses_all = []
    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())
            losses_l1.update(criterion_l1(outputs, targets).item(), inputs.size(0))
            losses_all.extend(criterion_gmean(outputs, targets).cpu().numpy())
    shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
    loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
    print(f" * Overall: MSE {losses_l1.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
    for name in ("many", "median", "low"):
        title = {"many": "Many", "median": "Median", "low": "Low"}[name]
        print(
            f" * {title}: MSE {shot_dict[name]['l1']:.3f}\t"
            f"L1 {shot_dict[name]['l1']:.3f}\tG-Mean {shot_dict[name]['gmean']:.3f}"
        )
    return losses_l1.avg, loss_gmean


def train_epoch(train_loader, model, proxies, optimizer, epoch):
    model.train()
    proxies.train()
    losses = AverageMeter("Loss", ":.3f")
    for inputs, targets, weights in tqdm(train_loader, desc=f"Epoch {epoch}"):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).float()
        weights = weights.cuda(non_blocking=True)
        outputs, enc = model(inputs, return_encoding=True)
        z_p, y_p = proxies()

        l_reg = weighted_l1_loss(outputs, targets, weights)
        l_p = proxy_loss(z_p, y_p, args.tau_t, args.tau_f, args.alpha)
        l_a = alignment_loss(
            targets, enc, z_p, y_p, args.tau_t, args.tau_f,
            prw=args.prw, delta_min=args.prw_delta,
        )
        loss = l_reg + args.lambda_p * l_p + args.lambda_a * l_a
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
    return losses.avg


def main():
    df = pd.read_csv(os.path.join(args.data_dir, "agedb.csv"))
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]
    train_labels = df_train["age"]
    y_min, y_max = float(train_labels.min()), float(train_labels.max())

    train_loader = DataLoader(
        AgeDB(df_train, args.data_dir, 224, split="train", reweight=args.reweight),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        AgeDB(df_val, args.data_dir, 224, split="val"),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        AgeDB(df_test, args.data_dir, 224, split="test"),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    model = nn.DataParallel(
        resnet50(fds=False, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel="gaussian", ks=5, sigma=2, momentum=0.9)
    ).cuda()
    proxies = ProxyBank(args.num_proxies, 2048, y_min, y_max).cuda()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(proxies.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    if args.evaluate:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        proxies.load_state_dict(ckpt["proxies"])
        print(f"Loaded {args.resume}")
        print("Test best model...")
        validate(test_loader, model, proxies, train_labels, prefix="Test")
        return

    cudnn.benchmark = True
    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train_epoch(train_loader, model, proxies, optimizer, epoch)
        val_l1, val_gm = validate(val_loader, model, proxies, train_labels)
        is_best = val_l1 < args.best_loss
        args.best_loss = min(val_l1, args.best_loss)
        save_checkpoint(args, {
            "epoch": epoch + 1,
            "best_loss": args.best_loss,
            "state_dict": model.state_dict(),
            "proxies": proxies.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, is_best)
        print(f"Epoch {epoch}: train={train_loss:.4f} val_l1={val_l1:.4f}")

    print("Test best model...")
    ckpt = torch.load(os.path.join(args.store_root, args.store_name, "ckpt.best.pth.tar"))
    model.load_state_dict(ckpt["state_dict"])
    proxies.load_state_dict(ckpt["proxies"])
    validate(test_loader, model, proxies, train_labels, prefix="Test")
    print("DONE: PRIME")


if __name__ == "__main__":
    main()
