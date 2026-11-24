import math
import os
import time

import torch
import torch.nn.functional as F

from .checkpoint import save_checkpoint


def _lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def evaluate(model, val_dl, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_dl):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, loss = model(x, y)
            total_loss += loss.item()
            n += 1
    model.train()
    if n == 0:
        return {"loss": float("nan"), "ppl": float("nan")}
    avg = total_loss / n
    return {"loss": avg, "ppl": math.exp(avg)}


def train(model, train_dl, val_dl, config, logger=None):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: _lr_lambda(s, config.warmup_steps, config.max_steps),
    )
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    t0 = time.time()
    model.train()

    while step < config.max_steps:
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % config.log_every == 0 and logger:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"step={step} loss={loss.item():.4f} "
                    f"ppl={math.exp(loss.item()):.2f} lr={lr:.2e} "
                    f"t={time.time()-t0:.0f}s"
                )

            if config.eval_every > 0 and step > 0 and step % config.eval_every == 0:
                metrics = evaluate(model, val_dl, device)
                if logger:
                    logger.info(
                        f"[eval] step={step} loss={metrics['loss']:.4f} "
                        f"ppl={metrics['ppl']:.2f}"
                    )

            if config.ckpt_every > 0 and step > 0 and step % config.ckpt_every == 0:
                save_checkpoint(model, optimizer, config, step,
                                os.path.join(config.out_dir, f"step_{step}.pt"))
                save_checkpoint(model, optimizer, config, step,
                                os.path.join(config.out_dir, "latest.pt"))
                if logger:
                    logger.info(f"[ckpt] step={step} guardado")

            step += 1
            if step >= config.max_steps:
                break

    save_checkpoint(model, optimizer, config, step,
                    os.path.join(config.out_dir, "final.pt"))
    if logger:
        logger.info(f"Entrenamiento completo: {step} pasos")
    return model
