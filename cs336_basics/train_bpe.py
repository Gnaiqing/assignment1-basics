from pathlib import Path
from cs336_basics.pretokenization import pre_tokenize_file
import sys
import psutil
from collections import defaultdict
import subprocess
import json
import os
import pathlib
import time
import argparse
import cProfile
import pstats

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent


# def visualize_byte_token(b):
#     # Decode safely
#     s = b.decode("utf-8", errors="replace")
#     # Replace leading spaces with Ġ (for readability)
#     s = s.replace(" ", "Ġ")
#     return s

def bytes_to_unicode():
    # Visible set: keep printable latin plus extended
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

# Build forward & inverse maps
_B2U = bytes_to_unicode()
_U2B = {v: k for k, v in _B2U.items()}


def visualize_bytes(b: bytes) -> str:
    """Render a byte sequence as printable unicode using GPT-2 mapping."""
    return "".join(_B2U[x] for x in b)

def unvisualize_text(s: str) -> bytes:
    """Recover the original bytes from a visualized string."""
    return bytes(_U2B[ch] for ch in s)

def write_merges_gpt_style(merges: list[tuple[bytes, bytes]], path: str) -> None:
    """Write merges in human-readable GPT-style (Ġ for space, Ċ for newline, ĉ for tab, etc.)."""
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{visualize_bytes(a)} {visualize_bytes(b)}\n")


def update_bytes_tuple(bytes_tuple, merged_bytes_pair):
    """
    Return an updated bytes tuple with a new bytes pair to be merged
    """
    new_token = merged_bytes_pair[0] + merged_bytes_pair[1]
    new_bytes_tuple = []
    for i in range(len(bytes_tuple)):
        if i > 0 and (new_bytes_tuple[-1], bytes_tuple[i]) == merged_bytes_pair:
            new_bytes_tuple[-1] = new_token
        else:
            new_bytes_tuple.append(bytes_tuple[i])

    new_bytes_tuple = tuple(new_bytes_tuple)
    return new_bytes_tuple

def update_bytes_pair_count(count_dict, bytes_tuple, delta=1):
    """
    Update the bytes pair counts when we add delta bytes_tuple into the corpus
    """
    for b1, b2 in zip(bytes_tuple[:-1], bytes_tuple[1:]):
        count_dict[(b1, b2)] += delta


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str]
              ) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
    """
    Train a BPE Tokenizer.
    Input:
        input_path: path to the training data
        vocab_size: maximum final vocabulary size
        special_tokens: A list of strings to add to the vocabulary
    Return:
        vocab: the tokenizer vocabulary, mapping from ID to bytes
        merges: A list of BPE merges produced from training
    """
    # Step 1: pre-tokenize the data
    input_filename = Path(input_path).stem
    pre_token_path = f"{ROOT_PATH}/temp/{input_filename}_pre_tokens.json"
    pre_token_dir = os.path.dirname(pre_token_path)
    os.makedirs(pre_token_dir, exist_ok=True)
    print("Start pretokenization...")
    word_count = pre_tokenize_file(input_path, n_process=4, special_tokens=["<|endoftext|>"])
    with open(pre_token_path, "w") as f:
        json.dump(word_count, f)

    bytes_count = defaultdict(int)
    for word in word_count:
        word_bytes = word.encode("utf-8")
        tuple_of_bytes = tuple(word_bytes[i:i + 1] for i in range(len(word_bytes)))
        bytes_count[tuple_of_bytes] = word_count[word]

    # initialize vocabulary
    vocab = dict()
    merges = list()
    for i in range(256):
        val = bytes([i])
        vocab[i] = val

    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    bytes_pair_count = defaultdict(int)
    bytes_pair_sources = defaultdict(set)  # track which tuples the bytes pairs come from
    for bytes_tuple, count in bytes_count.items():
        for b1, b2 in zip(bytes_tuple[:-1], bytes_tuple[1:]):
            bytes_pair_count[(b1, b2)] += count
            bytes_pair_sources[(b1, b2)].add(bytes_tuple)

    # count byte pairs
    while len(vocab) < vocab_size:
        max_count = 0
        max_bytes_pairs = []
        for bytes_pair, count in bytes_pair_count.items():
            if count > max_count:
                max_bytes_pairs.clear()
                max_bytes_pairs.append(bytes_pair)
                max_count = count
            elif count == max_count:
                max_bytes_pairs.append(bytes_pair)

        # add new token to merges and vocab
        merged_bytes_pair = max(max_bytes_pairs)
        merges.append(merged_bytes_pair)
        new_token = merged_bytes_pair[0] + merged_bytes_pair[1]

        vocab[next_id] = new_token
        next_id += 1

        # update info
        updates = []
        for original_bytes_tuple in bytes_pair_sources[merged_bytes_pair]:
            tuple_count = bytes_count[original_bytes_tuple]
            updated_bytes_tuple = update_bytes_tuple(original_bytes_tuple, merged_bytes_pair)
            updates.append((original_bytes_tuple, updated_bytes_tuple, tuple_count))

        for original_bytes_tuple, updated_bytes_tuple, tuple_count in updates:
            # update bytes pair count and sources
            for b1, b2 in zip(original_bytes_tuple[:-1], original_bytes_tuple[1:]):
                bytes_pair_count[(b1, b2)] -= tuple_count
                bytes_pair_sources[(b1, b2)].discard(original_bytes_tuple)

            for b1, b2 in zip(updated_bytes_tuple[:-1], updated_bytes_tuple[1:]):
                bytes_pair_count[(b1, b2)] += tuple_count
                bytes_pair_sources[(b1, b2)].add(updated_bytes_tuple)

            bytes_count[original_bytes_tuple] = 0
            bytes_count[updated_bytes_tuple] += tuple_count

    return vocab, merges


def main(args):
    input_path = args.input_path
    input_name = Path(input_path).stem
    special_tokens = ["<|endoftext|>"]
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1e9  # in GB

    vocab, merges = train_bpe(input_path=input_path,
                              vocab_size=args.vocab_size,
                              special_tokens=special_tokens)
    end_time = time.time()
    end_mem = process.memory_info().rss / 1e9
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Training took {elapsed_minutes:.2f} minutes")
    print(f"Memory used: {end_mem - start_mem:.2f} GB")

    # Longest token by string length
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {repr(longest_token)}, length = {len(longest_token)}")

    os.makedirs(args.output_dir, exist_ok=True)
    vocab_serial = {k: visualize_bytes(v) for k, v in vocab.items()}
    with open(f"{args.output_dir}/{input_name}-vocab.json", "w") as file:
        json.dump(vocab_serial, file, indent=4)  # indent for pretty-printing

    with open(f"{args.output_dir}/{input_name}-merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            # Decode each byte sequence to string
            f.write(f"{visualize_bytes(a)} {visualize_bytes(b)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="../data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="../preprocess")
    args = parser.parse_args()
    # profiler = cProfile.Profile()
    # profiler.enable()
    main(args)
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # # Sort the statistics by 'tottime' (total time spent in the function itself)
    # stats.sort_stats(pstats.SortKey.CUMULATIVE)
    # # Print the top 10 results
    # stats.print_stats(10)