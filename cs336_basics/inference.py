from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import typing
import json
import torch
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
from tqdm import tqdm

def decode(model, input):



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,default="../checkpoint/lm_TinyStoriesV2-GPT4-train_final.pt")
    parser.add_argument("--vocab_path", type=str, default="../preprocess/TinyStoriesV2-GPT4-train-vocab.json")
    parser.add_argument("--merge_path", type=str, default="../preprocess/TinyStoriesV2-GPT4-train-merges.txt")
    parser.add_argument("--input_text", type=str)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    # transformer parameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"File {args.model_path} not exist.")

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merge_path, special_tokens=["<|endoftext|>"])

    model = Transformer(vocab_size=args.vocab_size,
                        context_length=args.context_length,
                        d_model=args.d_model,
                        num_layers=args.num_layers,
                        num_heads=args.num_heads,
                        d_ff=args.d_ff,
                        rope_theta=args.rope_theta,
                        device=args.device)
    obj = torch.load(args.model_path)
    model.load_state_dict(obj["model_stats"])
    model.eval()



