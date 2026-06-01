"""
Train AgeDB-DIR with MTTS (Meta-Learned Target Topology Smoothing).
"""
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import AgeDB
from loss import weighted_focal_l1_loss, weighted_l1_loss, weighted_mse_loss
from mtts import MTTSKernel, bin_balanced_meta_loss
from resnet import resnet50
from train import AverageMeter, ProgressMeter, adjust_learning_rate, validate
from utils import save_checkpoint

os.environ["KMP_WARNINGS"] = "FALSE"

LOSS_FN = {
    "l1": weighted_l1_loss,
    "mse": weighted_mse_loss,
    "focal_l1": weighted_focal_l1_loss,
}


def split_train_meta(df_train, meta_ratio=0.1, seed=666):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df_train))
    rng.shuffle(idx)
    n_meta = max(1, int(len(idx) * meta_ratio))
    return idx[n_meta:], idx[:n_meta]


def unwrap_model(model):
    """Return inner module for meta-learning (stateless forward cannot use DataParallel)."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def functional_forward_resnet(backbone, params, buffers, inputs, targets, meta_epoch=0):
    from torch.nn.utils.stateless import functional_call

    combined = {**params, **buffers}
    out = functional_call(
        backbone,
        combined,
        (inputs,),
        kwargs={"targets": targets, "epoch": meta_epoch},
    )
    return out[0] if isinstance(out, tuple) else out


def inner_step_update(model, mtts, inputs, targets, inner_lr, loss_name="l1"):
    weights = mtts.sample_weights(targets)
    outputs = model(inputs, targets, 0)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss_fn = LOSS_FN.get(loss_name, weighted_l1_loss)
    loss = loss_fn(outputs, targets, weights)

    backbone = unwrap_model(model)
    all_params = dict(backbone.named_parameters())
    trainable = [(n, p) for n, p in all_params.items() if p.requires_grad]
    if not trainable:
        return None, loss.item()

    grads = torch.autograd.grad(
        loss, [p for _, p in trainable], create_graph=True, allow_unused=True
    )
    fast_params = dict(all_params)
    for (name, p), g in zip(trainable, grads):
        fast_params[name] = p if g is None else p - inner_lr * g
    return fast_params, loss.item()


def _is_fc_only_trainable(backbone):
    trainable = [(n, p) for n, p in backbone.named_parameters() if p.requires_grad]
    return bool(trainable) and all("fc" in n or "linear" in n for n, _ in trainable)


def meta_step_kernel_only(mtts, model, meta_loader, lam_local, lam_smooth, lam_sym, device, meta_batch_size=32):
    """Meta-update K without MAML inner step (fits full ResNet in VRAM)."""
    mtts.train()
    torch.cuda.empty_cache()
    was_training = model.training
    model.eval()
    per_bin_losses, per_bin_counts = {}, {}

    with torch.no_grad():
        for inputs, targets, _ in meta_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            n = inputs.size(0)
            for start in range(0, n, meta_batch_size):
                end = min(start + meta_batch_size, n)
                x = inputs[start:end]
                y = targets[start:end]
                outputs = model(x, y, 0)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                per_sample = F.l1_loss(outputs, y, reduction="none").view(-1)
                for i in range(y.size(0)):
                    b = int(y[i].item())
                    val = per_sample[i].detach()
                    per_bin_losses.setdefault(b, []).append(val)
                    per_bin_counts[b] = per_bin_counts.get(b, 0) + 1

    if was_training:
        model.train()

    per_bin_mean = {b: torch.stack(v).mean() for b, v in per_bin_losses.items()}
    K = mtts.kernel_matrix()
    meta_loss = bin_balanced_meta_loss(per_bin_mean, per_bin_counts, K, tau=mtts.meta_tau)
    reg = mtts.kernel_regularization(lam_local, lam_smooth, lam_sym)
    return meta_loss + reg, meta_loss.item()


def meta_step(
    model,
    mtts,
    train_loader,
    meta_loader,
    inner_lr,
    loss_name,
    lam_local,
    lam_smooth,
    lam_sym,
    device,
    meta_batch_size=32,
):
    backbone = unwrap_model(model)
    if not _is_fc_only_trainable(backbone):
        return meta_step_kernel_only(mtts, model, meta_loader, lam_local, lam_smooth, lam_sym, device, meta_batch_size)

    mtts.train()
    train_iter = iter(train_loader)
    fast_params = None
    buffers = {k: v for k, v in backbone.named_buffers()}

    try:
        inputs, targets, _ = next(train_iter)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        fast_params, _ = inner_step_update(model, mtts, inputs, targets, inner_lr, loss_name)
    except StopIteration:
        return torch.tensor(0.0, device=device), 0.0

    if fast_params is None:
        return torch.tensor(0.0, device=device), 0.0

    torch.cuda.empty_cache()
    was_training = backbone.training
    backbone.eval()

    per_bin_losses, per_bin_counts = {}, {}
    try:
        for inputs, targets, _ in meta_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            n = inputs.size(0)
            for start in range(0, n, meta_batch_size):
                end = min(start + meta_batch_size, n)
                x = inputs[start:end]
                y = targets[start:end]
                outputs = functional_forward_resnet(
                    backbone, fast_params, buffers, x, y, meta_epoch=0
                )
                per_sample = F.l1_loss(outputs, y, reduction="none").view(-1)
                for i in range(y.size(0)):
                    b = int(y[i].item())
                    per_bin_losses.setdefault(b, []).append(per_sample[i])
                    per_bin_counts[b] = per_bin_counts.get(b, 0) + 1
    finally:
        if was_training:
            backbone.train()

    per_bin_mean = {b: torch.stack(v).mean() for b, v in per_bin_losses.items()}
    meta_loss = bin_balanced_meta_loss(per_bin_mean, per_bin_counts, mtts.kernel_matrix(), tau=mtts.meta_tau)
    reg = mtts.kernel_regularization(lam_local, lam_smooth, lam_sym)
    return meta_loss + reg, meta_loss.item()


def train_epoch(train_loader, model, mtts, optimizer, epoch, args):
    losses = AverageMeter(f"Loss ({args.loss.upper()})", ":.3f")
    progress = ProgressMeter(len(train_loader), [losses], prefix=f"Epoch: [{epoch}]")
    model.train()
    mtts.train()
    loss_fn = LOSS_FN.get(args.loss, weighted_l1_loss)

    for idx, (inputs, targets, _) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        weights = mtts.sample_weights(targets)
        outputs = model(inputs, targets, epoch)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = loss_fn(outputs, targets, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update:
        encodings, labels = [], []
        with torch.no_grad():
            for inputs, targets, _ in tqdm(train_loader, desc="FDS features"):
                inputs = inputs.cuda(non_blocking=True)
                outputs, feature = model(inputs, targets, epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())
        encodings = torch.from_numpy(np.vstack(encodings)).cuda()
        labels = torch.from_numpy(np.hstack(labels)).cuda()
        unwrap_model(model).FDS.update_last_epoch_stats(epoch)
        unwrap_model(model).FDS.update_running_stats(encodings, labels, epoch)

    return losses.avg


def build_store_name(args):
    """Match train.py checkpoint folder naming so logs and ckpts align."""
    sn = f"_{args.store_name}" if args.store_name else "_mtts"
    if args.reweight != "none":
        sn += f"_{args.reweight}"
    if args.fds:
        sn += f"_fds_{args.fds_kernel[:3]}_{args.fds_ks}"
        if args.fds_kernel in ("gaussian", "laplace"):
            sn += f"_{args.fds_sigma}"
        sn += f"_{args.start_update}_{args.start_smooth}_{args.fds_mmt}"
    if args.retrain_fc:
        sn += "_retrain_fc"
    return f"{args.dataset}_resnet50{sn}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="agedb")
    parser.add_argument("--store_root", type=str, default="checkpoint")
    parser.add_argument("--store_name", type=str, default="", help="checkpoint folder name")
    parser.add_argument("--store_suffix", type=str, default="")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--inner_lr", type=float, default=1e-4)
    parser.add_argument("--meta_lr", type=float, default=1e-2)
    parser.add_argument("--meta_ratio", type=float, default=0.1)
    parser.add_argument("--meta_freq", type=int, default=1)
    parser.add_argument("--meta_batch_size", type=int, default=32, help="meta forward micro-batch (lower saves VRAM)")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "mse", "focal_l1"])
    parser.add_argument("--schedule", type=int, nargs="*", default=[60, 80])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--num_bins", type=int, default=121)
    parser.add_argument("--reweight", type=str, default="sqrt_inv", choices=["sqrt_inv", "inverse"])
    parser.add_argument("--lds_sigma", type=float, default=2.0)
    parser.add_argument("--lds_ks", type=int, default=5)
    parser.add_argument("--mtts_beta", type=float, default=1.0)
    parser.add_argument("--lambda_d", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--lam_local", type=float, default=0.01)
    parser.add_argument("--lam_smooth", type=float, default=0.01)
    parser.add_argument("--lam_sym", type=float, default=0.01)
    parser.add_argument("--unconstrained_kernel", action="store_true")
    parser.add_argument("--fds", action="store_true")
    parser.add_argument("--fds_kernel", type=str, default="gaussian")
    parser.add_argument("--fds_ks", type=int, default=5)
    parser.add_argument("--fds_sigma", type=float, default=2.0)
    parser.add_argument("--start_update", type=int, default=0)
    parser.add_argument("--start_smooth", type=int, default=1)
    parser.add_argument("--bucket_num", type=int, default=100)
    parser.add_argument("--bucket_start", type=int, default=3)
    parser.add_argument("--fds_mmt", type=float, default=0.9)
    parser.add_argument("--retrain_fc", action="store_true")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.unconstrained_kernel:
        args.lam_local = args.lam_smooth = args.lam_sym = 0.0

    args.store_name = build_store_name(args)
    log_dir = os.path.join(args.store_root, args.store_name)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, "training.log")), logging.StreamHandler()],
    )
    log = logging.info
    log(f"Args: {args}")

    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val, df_test = df[df["split"] == "val"], df[df["split"] == "test"]
    train_labels = df_train["age"].values

    train_idx, meta_idx = split_train_meta(df_train, meta_ratio=args.meta_ratio)
    df_tr = df_train.iloc[train_idx].reset_index(drop=True)
    df_meta = df_train.iloc[meta_idx].reset_index(drop=True)

    train_set = AgeDB(df_tr, args.data_dir, args.img_size, split="train", reweight="none")
    meta_set = AgeDB(df_meta, args.data_dir, args.img_size, split="val")
    val_set = AgeDB(df_val, args.data_dir, args.img_size, split="val")
    test_set = AgeDB(df_test, args.data_dir, args.img_size, split="test")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    meta_loader = DataLoader(meta_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = resnet50(
        fds=args.fds,
        bucket_num=args.bucket_num,
        bucket_start=args.bucket_start,
        start_update=args.start_update,
        start_smooth=args.start_smooth,
        kernel=args.fds_kernel,
        ks=args.fds_ks,
        sigma=args.fds_sigma,
        momentum=args.fds_mmt,
    ).cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if args.retrain_fc:
        for name, param in model.named_parameters():
            if "fc" not in name and "linear" not in name:
                param.requires_grad = False

    mtts = MTTSKernel(
        num_bins=args.num_bins,
        embed_dim=args.embed_dim,
        lambda_d=args.lambda_d,
        lds_sigma=args.lds_sigma,
        lds_ks=args.lds_ks,
        beta=args.mtts_beta,
        unconstrained=args.unconstrained_kernel,
    ).cuda()
    mtts.set_raw_counts(train_labels, reweight=args.reweight)

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)
    mtts_optimizer = torch.optim.Adam(mtts.parameters(), lr=args.meta_lr)

    if args.pretrained and os.path.isfile(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        new_state = {k: v for k, v in state.items() if "linear" not in k and "fc" not in k}
        model.load_state_dict(new_state, strict=False)
        log(f"Loaded pretrained backbone from {args.pretrained}")

    if args.evaluate:
        ckpt = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ckpt["state_dict"])
        validate(test_loader, model, train_labels=train_labels, prefix="Test")
        return

    best_loss, start_epoch = 1e5, 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ckpt["state_dict"])
        if "mtts" in ckpt:
            mtts.load_state_dict(ckpt["mtts"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("best_loss", best_loss)

    cudnn.benchmark = True
    device = torch.device("cuda")

    for epoch in range(start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train_epoch(train_loader, model, mtts, optimizer, epoch, args)

        if args.meta_freq > 0 and (epoch + 1) % args.meta_freq == 0:
            mtts_optimizer.zero_grad()
            total_kernel, meta_val = meta_step(
                model, mtts, train_loader, meta_loader, args.inner_lr, args.loss,
                args.lam_local, args.lam_smooth, args.lam_sym, device,
                meta_batch_size=args.meta_batch_size,
            )
            if isinstance(total_kernel, torch.Tensor) and total_kernel.requires_grad and total_kernel.item() != 0:
                total_kernel.backward()
                mtts_optimizer.step()
            log(f"Kernel update: L_meta={meta_val:.4f}")

        val_mse, val_l1, val_gm = validate(val_loader, model, train_labels=train_labels)
        metric = val_l1 if "l1" in args.loss else val_mse
        is_best = metric < best_loss
        best_loss = min(metric, best_loss)
        save_checkpoint(
            argparse.Namespace(store_root=args.store_root, store_name=args.store_name),
            {"epoch": epoch + 1, "best_loss": best_loss, "state_dict": model.state_dict(), "mtts": mtts.state_dict(), "optimizer": optimizer.state_dict()},
            is_best,
        )
        log(f"Epoch {epoch}: train={train_loss:.4f} val_l1={val_l1:.4f}")

    ckpt = torch.load(os.path.join(log_dir, "ckpt.best.pth.tar"))
    model.load_state_dict(ckpt["state_dict"])
    log("Test best model...")
    validate(test_loader, model, train_labels=train_labels, prefix="Test")


if __name__ == "__main__":
    main()
