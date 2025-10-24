from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import typing
import json
import torch
import time
import torch.nn as nn
import math
import wandb
import numpy as np
import numpy.typing as npt
import argparse
from cs336_basics.layers import Transformer
from cs336_basics.optimizer import AdamW, cross_entropy, gradient_clipping, calc_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
import os, io
from array import array
from tqdm import tqdm, trange


def get_batch(token_ids: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    """
    Load data from a numpy array and return sampled inputs and targets tensor with size (batch_size, context_length)
    """
    n = len(token_ids)
    start_indices = np.random.choice(n - context_length, batch_size, replace=False)
    sample_inputs = torch.stack([torch.from_numpy(token_ids[i:i+context_length].copy())
                                 for i in start_indices]).to(torch.long).to(device)
    sample_outputs = torch.stack([torch.from_numpy(token_ids[i+1:i+1+context_length].copy()).to(torch.long)
                                  for i in start_indices]).to(torch.long).to(device)
    return sample_inputs, sample_outputs


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int ,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    """
    Save model stats into out
    """
    obj = {
        "model_stats": model.state_dict(),
        "optimizer_stats": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(obj, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    """
    Load stats into model and optimizer
    Return: the iteration number
    """
    obj = torch.load(src)
    model.load_state_dict(obj["model_stats"])
    optimizer.load_state_dict(obj["optimizer_stats"])
    return obj["iteration"]


import os, io
import numpy as np
from array import array
from tqdm import tqdm


def tokenize_data_fast(tokenizer, input_path, output_path,
                       flush_tokens=1_000_000,
                       dtype=np.uint16,
                       show_progress=True):
    """Tokenize a large text file and stream token IDs to disk with tqdm progress."""
    if np.iinfo(dtype).max < len(tokenizer.vocab) - 1:
        raise ValueError("dtype too small for vocab; use np.uint32.")

    if os.path.exists(output_path):
        os.remove(output_path)

    total_bytes = os.path.getsize(input_path)
    processed_bytes = 0
    total_tokens = 0

    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, disable=not show_progress,
                desc="Tokenizing")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "wb", buffering=io.DEFAULT_BUFFER_SIZE) as fout:

        buf = array('H') if dtype == np.uint16 else array('I')

        for line in fin:  # iterate manually over lines
            processed_bytes += len(line.encode("utf-8"))
            token_ids = tokenizer.encode(line)
            for tok in token_ids:
                if isinstance(tok, int):
                    buf.append(tok)
                else:
                    buf.extend(tok)

                total_tokens += 1

                if len(buf) >= flush_tokens:
                    buf.tofile(fout)
                    buf = array('H') if dtype == np.uint16 else array('I')

            # update tqdm every few lines
            if processed_bytes - pbar.n > 64 * 1024:  # every 64 KB
                pbar.n = processed_bytes
                pbar.refresh()

        if len(buf):
            buf.tofile(fout)

    pbar.n = total_bytes
    pbar.close()

    print(f"\n✅ Done: {total_tokens:,} tokens written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset and tokenizer
    parser.add_argument("--train_path", type=str, default="../data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid_path", type=str, default="../data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_path", type=str, default="../preprocess/TinyStoriesV2-GPT4-train-vocab.json")
    parser.add_argument("--merge_path", type=str, default="../preprocess/TinyStoriesV2-GPT4-train-merges.txt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--context_length", type=int, default=256)
    # model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps")
    # optimizer
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta_0", type=float, default=0.9)
    parser.add_argument("--beta_1", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_l2_norm", type=float, default=10.0)
    # save model stats
    parser.add_argument("--checkpoint", type=str, default="../checkpoint")
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--save_wandb", action="store_true")
    args = parser.parse_args()
    if args.save_wandb:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="uoft-db",
            # Set the wandb project where this run will be logged.
            project="cs336-hw1",
            # Track hyperparameters and run metadata.
            config=vars(args),
        )

    # tokenize input datasets
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merge_path, special_tokens=["<|endoftext|>"])
    train_filename = Path(args.train_path).stem
    valid_filename = Path(args.valid_path).stem
    train_token_path = f"../preprocess/{train_filename}.bin"
    valid_token_path = f"../preprocess/{valid_filename}.bin"
    if args.vocab_size <= np.iinfo(np.uint16).max:
        id_type = np.uint16
    else:
        id_type = np.uint32

    if not os.path.exists(valid_token_path):
        tokenize_data_fast(tokenizer, args.valid_path, valid_token_path, dtype=id_type)
    if not os.path.exists(train_token_path):
        tokenize_data_fast(tokenizer, args.train_path, train_token_path, dtype=id_type)

    train_token_ids = np.memmap(train_token_path, dtype=id_type, mode="r")
    valid_token_ids = np.memmap(valid_token_path, dtype=id_type, mode="r")

    # setup model and optimizer
    model = Transformer(vocab_size=args.vocab_size,
                        context_length=args.context_length,
                        d_model=args.d_model,
                        num_layers=args.num_layers,
                        num_heads=args.num_heads,
                        d_ff=args.d_ff,
                        rope_theta=args.rope_theta,
                        device=args.device)
    opt = AdamW(params=model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(args.beta_0, args.beta_1),
                eps=args.eps)

    model.train()

    # Exponential moving average for training loss
    ema_beta = 0.9
    train_loss_ema = None  # will init on first step
    ema_bias_correction = 1.0

    avg_val_loss = float('nan')
    progress_bar = trange(1, args.total_steps + 1, desc="Training", leave=True)
    start_time = time.time()

    # Optionally, predefine a deterministic set of validation batches:
    rng_val = np.random.RandomState(42)
    fixed_val_indices = [rng_val.randint(0, len(valid_token_ids) - args.context_length) for _ in range(100)]

    for step in progress_bar:
        inputs, targets = get_batch(train_token_ids,
                                    batch_size=args.batch_size,
                                    context_length=args.context_length,
                                    device=args.device)

        logits = model(inputs)
        train_loss_batch_t = cross_entropy(logits, targets)  # tensor
        train_loss_batch = float(train_loss_batch_t.item())  # scalar

        opt.zero_grad()

        lr_t = calc_lr_cosine_schedule(
            t=step,
            lr_max=args.lr,
            lr_min=args.lr_min,
            t_w=args.warmup_steps,
            t_c=args.total_steps,
        )
        for group in opt.param_groups:
            group["lr"] = lr_t

        train_loss_batch_t.backward()
        gradient_clipping(model.parameters(), args.max_l2_norm)
        opt.step()

        # Update EMA with bias correction: m_t = beta*m_{t-1} + (1-beta)*x_t; m_hat = m_t / (1 - beta^t)
        if train_loss_ema is None:
            train_loss_ema = train_loss_batch
            ema_bias_correction = 1.0 - ema_beta  # beta^1
        else:
            train_loss_ema = ema_beta * train_loss_ema + (1.0 - ema_beta) * train_loss_batch
            ema_bias_correction = 1.0 - (ema_beta ** step)

        train_loss_ema_corrected = train_loss_ema / max(ema_bias_correction, 1e-12)

        if step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                # Use fixed deterministic subset for comparability:
                for idx in fixed_val_indices:
                    x = torch.from_numpy(valid_token_ids[idx:idx + args.context_length]).to(args.device).unsqueeze(
                        0).repeat(args.batch_size, 1)
                    y = torch.from_numpy(valid_token_ids[idx + 1:idx + 1 + args.context_length]).to(
                        args.device).unsqueeze(0).repeat(args.batch_size, 1)
                    val_logits = model(x)
                    val_loss_t = cross_entropy(val_logits, y)
                    val_losses.append(float(val_loss_t.item()))
            avg_val_loss = float(np.mean(val_losses))

            if args.save_wandb:
                time_elapsed_min = (time.time() - start_time) / 60.0
                run.log({
                    "step": step,
                    "training_time_min": time_elapsed_min,
                    "lr": lr_t,
                    # log both:
                    "train_loss_batch": train_loss_batch,
                    "train_loss_ema": train_loss_ema_corrected,
                    "valid_loss": avg_val_loss,
                })

            output_path = Path(args.checkpoint) / f"lm_{train_filename}_{step}.pt"
            os.makedirs(output_path.parent, exist_ok=True)
            save_checkpoint(model, opt, step, output_path)
            model.train()

        progress_bar.set_postfix({
            "train_loss(batch)": f"{train_loss_batch:.3f}",
            "train_loss(ema)": f"{train_loss_ema_corrected:.3f}",
            "val_loss": f"{avg_val_loss:.3f}",
            "lr": f"{lr_t:.2e}",
        })

    # model.train()
    # train_loss = 0.0
    # avg_val_loss = float('nan')  # initialize
    # progress_bar = trange(1, args.total_steps + 1, desc="Training", leave=True)
    # start_time = time.time()
    # for step in progress_bar:
    #     inputs, targets = get_batch(train_token_ids,
    #                                 batch_size=args.batch_size,
    #                                 context_length=args.context_length,
    #                                 device=args.device)
    #     logits = model(inputs)
    #     loss = cross_entropy(logits, targets)
    #     opt.zero_grad()
    #     lr_t = calc_lr_cosine_schedule(
    #         t=step,
    #         lr_max=args.lr,
    #         lr_min=args.lr_min,
    #         t_w=args.warmup_steps,
    #         t_c=args.total_steps,
    #     )
    #     for group in opt.param_groups:
    #         group["lr"] = lr_t
    #
    #     loss.backward()
    #     train_loss = 0.9 * train_loss + 0.1 * loss.item()  # weighting average
    #     gradient_clipping(model.parameters(), args.max_l2_norm)
    #     opt.step()
    #     if step % args.eval_interval == 0:
    #         model.eval()
    #         val_losses = []
    #         with torch.no_grad():
    #             for _ in range(100):  # could be all, or just e.g. 100 batches
    #                 x, y = get_batch(valid_token_ids, args.batch_size, args.context_length, args.device)
    #                 logits = model(x)
    #                 loss = cross_entropy(logits, y)
    #                 val_losses.append(loss.item())
    #         avg_val_loss = np.mean(val_losses)
    #         if args.save_wandb:
    #             time_elapsed = (time.time() - start_time) / 60
    #             run.log({
    #                 "step": step,
    #                 "training_time(min)": time_elapsed,
    #                 "train_loss": loss,
    #                 "valid_loss": avg_val_loss
    #             })
    #
    #         output_path = Path(args.checkpoint) / f"lm_{train_filename}_{step}.pt"
    #         if not os.path.exists(os.path.dirname(output_path)):
    #             os.makedirs(os.path.dirname(output_path))
    #         save_checkpoint(model, opt, step, output_path)
    #         model.train()
    #
    #     progress_bar.set_postfix({
    #         "train_loss": f"{train_loss:.3f}",
    #         "val_loss": f"{avg_val_loss:.3f}",
    #         "lr": f"{lr_t:.2e}"
    #     })

    output_path = Path(args.checkpoint) / f"lm_{train_filename}_final.pt"
    save_checkpoint(model, opt, args.total_steps+1, output_path)
    print(f"Final model saved to {output_path}")
    if args.save_wandb:
        run.finish()
