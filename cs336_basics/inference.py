import argparse
from cs336_basics.layers import Transformer
from cs336_basics.tokenizer import Tokenizer
import os, io
from layers import softmax
import torch
from torch import nn


@torch.no_grad()
def decode(
    model: nn.Module,
    tokenizer,
    input_tokens: torch.Tensor,                 # shape: (seq_len,), dtype=torch.long, on correct device
    *,
    context_length: int = 256,                  # your model's window; will crop prompt if too long
    eos_tokens: list[int] | None = None,        # e.g. [tokenizer.lookup[b"<|endoftext|>"]]
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    return_tokens: bool = False,
):
    """
    Autoregressive decoding for a Transformer without KV cache.
    Returns text by default; set return_tokens=True to also get (full_tokens, new_tokens).
    """
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    device = input_tokens.device
    was_training = model.training
    model.eval()

    # Crop prompt to fit context window
    if input_tokens.numel() > context_length:
        input_tokens = input_tokens[-context_length:]

    new_tokens: list[int] = []

    for _ in range(max_new_tokens):
        # Forward the entire (possibly trimmed) context because there is no KV cache.
        logits = model(input_tokens.unsqueeze(0))  # (1, seq, vocab)
        next_logits = logits[0, -1, :]             # (vocab,)

        if temperature == 0.0:
            # Greedy
            next_id = int(torch.argmax(next_logits))
        else:
            # Stable softmax with temperature
            next_logits = next_logits / temperature
            probs = softmax(next_logits, dim=-1)

            if top_p < 1.0:
                # Nucleus sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                # first index where cumsum > top_p
                cutoff = torch.searchsorted(cumsum, torch.tensor(top_p, device=device))
                cutoff = int(torch.clamp(cutoff + 1, max=sorted_probs.size(0)))
                keep_idx = sorted_idx[:cutoff]
                keep_probs = sorted_probs[:cutoff]
                keep_probs = keep_probs / keep_probs.sum()
                # sample inside the truncated set
                sample_pos = torch.multinomial(keep_probs, 1)
                next_id = int(keep_idx[sample_pos])
            else:
                next_id = int(torch.multinomial(probs, 1))

        new_tokens.append(next_id)

        # Stop on EOS if provided
        if eos_tokens is not None and next_id in eos_tokens:
            break

        # Append and trim to context window
        input_tokens = torch.cat(
            (input_tokens, torch.tensor([next_id], device=device, dtype=torch.long)),
            dim=0
        )
        if input_tokens.size(0) > context_length:
            input_tokens = input_tokens[-context_length:]

    text = tokenizer.decode(new_tokens) if new_tokens else ""

    if was_training:
        model.train()

    if return_tokens:
        return input_tokens, torch.tensor(new_tokens, device=device), text
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,default="../checkpoint/lm_TinyStoriesV2-GPT4-train_10.pt")
    parser.add_argument("--vocab_path", type=str, default="../preprocess/TinyStoriesV2-GPT4-train-vocab.json")
    parser.add_argument("--merge_path", type=str, default="../preprocess/TinyStoriesV2-GPT4-train-merges.txt")
    parser.add_argument("--input", type=str, default=" ")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--eos_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    # transformer parameters
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps")

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
    input_tokens = tokenizer.encode(args.input)
    input_tokens = torch.tensor(input_tokens[-args.context_length:], dtype=torch.long, device=args.device)
    eos_tokens = []
    for eos_token in args.eos_tokens:
        if eos_token.encode("utf-8") in tokenizer.lookup:
            eos_tokens.append(tokenizer.lookup[eos_token.encode("utf-8")])

    output = decode(model=model,
                    tokenizer=tokenizer,
                    input_tokens=input_tokens,
                    context_length=args.context_length,
                    max_new_tokens=args.max_new_tokens,
                    eos_tokens=eos_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p)

    print(output)



