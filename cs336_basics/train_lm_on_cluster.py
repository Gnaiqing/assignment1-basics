from torch import Tensor
import typing
import torch
import time
import wandb
import numpy.typing as npt
import argparse
from cs336_basics.layers import Transformer
from cs336_basics.optimizer import AdamW, cross_entropy, gradient_clipping, calc_lr_cosine_schedule
from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
from tqdm import tqdm, trange
import os, io, argparse, time, typing
from pathlib import Path
import yaml
import os, io
import numpy as np
from array import array
from tqdm import tqdm
from contextlib import nullcontext


def autocast_ctx(device_str: str, dtype: torch.dtype):
    if device_str == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    # CPU autocast supports bf16; keep it for consistency
    if device_str == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16 if dtype==torch.bfloat16 else torch.float32)
    # mps (Apple) and other backends: disable autocast
    return nullcontext()


def _expand_env(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))


def load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # expand $VARS in string values
    def rec(x):
        if isinstance(x, str): return _expand_env(x)
        if isinstance(x, dict): return {k: rec(v) for k, v in x.items()}
        if isinstance(x, list): return [rec(v) for v in x]
        return x
    return rec(cfg)


def merge_args_with_yaml(parser: argparse.ArgumentParser, config: dict) -> argparse.Namespace:
    # YAML sets defaults, CLI overrides
    for k, v in config.items():
        if k in {a.dest for a in parser._actions}:
            parser.set_defaults(**{k: v})
    return parser.parse_args()


def resolve_device(dev_str: str) -> str:
    if dev_str == "auto":
        return "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    return dev_str


def get_batch(
    token_ids: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    seed: int | None = None
) -> tuple[Tensor, Tensor]:
    """
    Load data from a numpy array and return sampled inputs and targets tensor
    with size (batch_size, context_length). If a seed is given, the sampling is deterministic.
    """
    n = len(token_ids)

    # Use fixed RNG seed for deterministic sampling (validation)
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    start_indices = np.random.choice(n - context_length, batch_size, replace=False)

    if seed is not None:
        np.random.set_state(rng_state)

    sample_inputs = torch.stack([
        torch.from_numpy(token_ids[i:i + context_length].copy()) for i in start_indices
    ]).to(torch.long).to(device)

    sample_outputs = torch.stack([
        torch.from_numpy(token_ids[i + 1:i + 1 + context_length].copy()) for i in start_indices
    ]).to(torch.long).to(device)

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
    parser.add_argument("--config", type=str, default=None)
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="bf16")
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
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="resume from latest checkpoint in ckpt dir if present")
    parser.add_argument("--num_val_batches", type=int, default=100)
    parser.add_argument("--runtime_data_dir", type=str, default=None)
    parser.add_argument("--archive_dir", type=str, default=None)
    parser.add_argument("--save_wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="uoft-db")
    parser.add_argument("--wandb_project", type=str, default="cs336-hw1")

    # load YAML then merge with CLI
    cfg = load_config(os.environ.get("CONFIG_PATH", None) or None)  # allows exporting CONFIG_PATH
    args_pre = parser.parse_args()
    cfg2 = load_config(args_pre.config)
    cfg.update(cfg2)
    args = merge_args_with_yaml(parser, cfg)
    from datetime import datetime

    # Get the current date and time
    now = datetime.now()
    timestamp_string = now.strftime("%Y-%m-%d %H:%M:%S")

    # expand env vars and resolve device
    args.train_path = _expand_env(args.train_path)
    args.valid_path = _expand_env(args.valid_path)
    args.vocab_path = _expand_env(args.vocab_path)
    args.merge_path = _expand_env(args.merge_path)
    args.checkpoint = _expand_env(args.checkpoint)
    if args.runtime_data_dir:
        args.runtime_data_dir = _expand_env(args.runtime_data_dir)
    if args.archive_dir:
        args.archive_dir = _expand_env(args.archive_dir)
    args.device = resolve_device(args.device)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    # tokenize input datasets
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merge_path, special_tokens=["<|endoftext|>"])

    # Prefer staged copies if runtime_data_dir exists (set by SLURM script)
    data_root = args.runtime_data_dir if (args.runtime_data_dir and os.path.isdir(args.runtime_data_dir)) else None

    train_filename = Path(args.train_path).stem
    valid_filename = Path(args.valid_path).stem

    if data_root:
        train_token_path = f"{data_root}/{train_filename}.bin"
        valid_token_path = f"{data_root}/{valid_filename}.bin"
    else:
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

    if args.dtype == "bf16":
        data_type = torch.bfloat16
    elif args.dtype == "fp16":
        data_type = torch.float16
    elif args.dtype == "fp32":
        data_type = torch.float32
    else:
        raise ValueError(f"dtype {args.dtype} not supported")

    if args.save_wandb:
        # os.environ.setdefault("WANDB_MODE", "offline")
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=vars(args))

    torch.set_float32_matmul_precision("high")  # optional
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

    if args.resume:
        ckpt_dir = Path(args.checkpoint)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted(ckpt_dir.glob("lm_*_*.pt"))
        if candidates:
            last = candidates[-1]
            print(f"Resuming from {last}")
            _iter = load_checkpoint(last, model, opt)

    model.train()

    # Exponential moving average for training loss
    ema_beta = 0.9
    train_loss_ema = None  # will init on first step
    ema_bias_correction = 1.0

    avg_val_loss = float('nan')
    progress_bar = trange(1, args.total_steps + 1, desc="Training", leave=True)
    start_time = time.time()

    for step in progress_bar:
        inputs, targets = get_batch(train_token_ids,
                                    batch_size=args.batch_size,
                                    context_length=args.context_length,
                                    device=args.device)

        with autocast_ctx(args.device, data_type):
            logits = model(inputs)
            train_loss_batch_t = cross_entropy(logits, targets)  # tensor
            train_loss_batch = float(train_loss_batch_t.item())  # scalar

        opt.zero_grad(set_to_none=True)

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

        if train_loss_ema is None:
            train_loss_ema = train_loss_batch
            ema_bias_correction = 1.0 - ema_beta  # beta^1
        else:
            train_loss_ema = ema_beta * train_loss_ema + (1.0 - ema_beta) * train_loss_batch
            ema_bias_correction = 1.0 - (ema_beta ** step)

        train_loss_ema_corrected = train_loss_ema / max(ema_bias_correction, 1e-12)
        if args.save_wandb:
            wandb.log({
                "step": step,
                "train/loss": train_loss_batch,
                "train/loss_ema": train_loss_ema,
                "train/lr": lr_t,
            }, step=step)

        if step % args.eval_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad(), autocast_ctx(args.device, data_type):
                for i in range(args.num_val_batches):
                    x, y = get_batch(valid_token_ids, args.batch_size, args.context_length, args.device, seed=i)
                    val_logits = model(x)
                    val_loss_t = cross_entropy(val_logits, y)
                    val_losses.append(float(val_loss_t.item()))

            avg_val_loss = float(np.mean(val_losses))
            model.train()

            if args.save_wandb:
                wandb.log({
                    "valid_loss": avg_val_loss,
                }, step=step)

            output_path = Path(args.checkpoint) / f"lm_{train_filename}_{timestamp_string}_{step}.pt"
            os.makedirs(output_path.parent, exist_ok=True)
            save_checkpoint(model, opt, step, output_path)


        progress_bar.set_postfix({
            "train_loss(batch)": f"{train_loss_batch:.3f}",
            "train_loss(ema)": f"{train_loss_ema_corrected:.3f}",
            "val_loss": f"{avg_val_loss:.3f}",
            "lr": f"{lr_t:.2e}",
        })

    output_path = Path(args.checkpoint) / f"lm_{train_filename}_{timestamp_string}_final.pt"
    save_checkpoint(model, opt, args.total_steps+1, output_path)
    print(f"Final model saved to {output_path}")
    if args.save_wandb:
        run.finish()
