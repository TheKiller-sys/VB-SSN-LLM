"""Bucle de entrenamiento con resume, grad-accum, AMP, eval y checkpoints."""
from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext

import torch

from .checkpoint import save_checkpoint
from .config import resolve_device


def _lr_lambda(step: int, warmup: int, total: int, min_ratio: float = 0.0) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


@torch.no_grad()
def evaluate(model, val_dl, device, max_batches: int = 50) -> dict:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(val_dl):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    if was_training:
        model.train()
    if n == 0:
        return {"loss": float("nan"), "ppl": float("nan")}
    avg = total_loss / n
    return {"loss": avg, "ppl": math.exp(min(avg, 20.0))}


def train(model, train_dl, val_dl, config, logger=None, start_step: int = 0,
          optimizer: torch.optim.Optimizer | None = None,
          optimizer_state=None, scheduler_state=None):
    device = resolve_device(config.device)
    model = model.to(device)

    if config.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            if logger:
                logger.info("torch.compile activado")
        except Exception as e:
            if logger:
                logger.warning(f"torch.compile falló: {e}")

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    min_ratio = config.min_lr / config.lr if config.lr > 0 else 0.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: _lr_lambda(s, config.warmup_steps, config.max_steps, min_ratio),
    )
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    elif start_step > 0:
        scheduler.step(start_step)

    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    )

    accum = max(1, config.grad_accum_steps)
    step = start_step
    micro_step = 0
    t0 = time.time()
    model.train()

    if logger:
        logger.info(
            f"Entrenando: device={device} amp={use_amp} "
            f"accum={accum} start_step={start_step} max_steps={config.max_steps}"
        )

    optimizer.zero_grad(set_to_none=True)

    while step < config.max_steps:
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast_ctx:
                _, loss = model(x, y)
                loss_scaled = loss / accum

            scaler.scale(loss_scaled).backward()
            micro_step += 1

            if micro_step % accum != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if step % config.log_every == 0 and logger:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"step={step} loss={loss.item():.4f} "
                    f"ppl={math.exp(min(loss.item(), 20)):.2f} "
                    f"lr={lr:.2e} t={time.time() - t0:.0f}s"
                )

            if (
                config.eval_every > 0
                and step > 0
                and step % config.eval_every == 0
            ):
                metrics = evaluate(model, val_dl, device, config.eval_batches)
                if logger:
                    logger.info(
                        f"[eval] step={step} loss={metrics['loss']:.4f} "
                        f"ppl={metrics['ppl']:.2f}"
                    )
                model.train()

            if (
                config.ckpt_every > 0
                and step > 0
                and step % config.ckpt_every == 0
            ):
                save_checkpoint(
                    model, optimizer, scheduler, config, step,
                    os.path.join(config.out_dir, f"step_{step}.pt"),
                )
                save_checkpoint(
                    model, optimizer, scheduler, config, step,
                    os.path.join(config.out_dir, "latest.pt"),
                )
                if logger:
                    logger.info(f"[ckpt] step={step} guardado")

            step += 1
            if step >= config.max_steps:
                break

    save_checkpoint(
        model, optimizer, scheduler, config, step,
        os.path.join(config.out_dir, "final.pt"),
    )
    if logger:
        logger.info(f"Entrenamiento completo: {step} pasos")
    return model
