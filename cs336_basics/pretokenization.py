import os
from typing import BinaryIO
from collections import defaultdict
import multiprocessing
import psutil
import time
import codecs
import io
import json
import regex as re
import argparse


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize(content: str, special_tokens: list[str]) -> dict[str, int]:
    """
    Pre-tokenize a chunk of content
    """
    split_pattern = "|".join(re.escape(token) for token in special_tokens)
    docs = re.split(split_pattern, content)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_count = defaultdict(int)
    for doc in docs:
        for match in re.finditer(PAT, doc):
            token_count[match.group()] += 1

    return token_count


def pre_tokenize_file_part(
    input_file: str,
    start: int,
    end: int,
    special_tokens: list[str],
    *,
    encoding: str = "utf-8",
    chunk_bytes: int = 1 << 20,  # 1 MiB
) -> dict[str, int]:
    """
    Stream [start, end) and pre-tokenize lazily.
    We only cut work units at SPECIAL token boundaries, so normal tokens never get split.
    """

    # Special tokens: longest-first to prefer longer matches
    special_alt = (
        "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        if special_tokens else r"(?!x)x"  # never matches
    )
    special_re = re.compile(special_alt, re.UNICODE | re.MULTILINE)

    # Your original PAT (regex module supports \p{L}/\p{N})
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat_re = re.compile(PAT, re.UNICODE | re.MULTILINE)

    counts = defaultdict(int)

    # Rolling decoded text buffer
    buf = ""
    # Incremental decoder so we never break UTF-8 codepoints across chunk boundaries
    inc = codecs.getincrementaldecoder(encoding)(errors="strict")

    def count_range(txt: str):
        """Count PAT tokens in txt (no special tokens inside by construction)."""
        for m in pat_re.finditer(txt):
            counts[m.group()] += 1

    remaining = max(0, end - start)

    with open(input_file, "rb") as f:
        f.seek(start)
        reader = io.BufferedReader(f, buffer_size=min(chunk_bytes, remaining))

        eof = False
        while True:
            # Try to find the next SPECIAL in the current buffer
            m = special_re.search(buf)
            if m:
                # Process everything BEFORE the special token
                head = buf[:m.start()]
                if head:
                    count_range(head)
                # Drop head + the special token (special acts as a boundary; we don't count it)
                buf = buf[m.end():]
                # Loop again; there might be another SPECIAL already in buffer
                continue

            # No SPECIAL in the buffer: need more data or finish
            if not eof and remaining > 0:
                to_read = min(chunk_bytes, remaining)
                data = reader.read(to_read)
                if not data:
                    eof = True
                else:
                    remaining -= len(data)
                    buf += inc.decode(data, final=False)
                continue

            # EOF for our [start, end) range: process whatever remains and stop
            # Flush any decoder tail (e.g., final codepoint completion)
            buf += inc.decode(b"", final=True)
            if buf:
                count_range(buf)
            break

    return counts


def pre_tokenize_file(input_file: str, n_process: int, special_tokens: list[str]) -> dict[str, int]:
    """
    Pre-tokenize a file into a dictionary that map pre-tokens to ints
    """
    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, n_process, b"<|endoftext|>")
        arg_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            arg_list.append([input_file, start, end, special_tokens])

        with multiprocessing.Pool(len(arg_list)) as pool:
            results = pool.starmap(pre_tokenize_file_part, arg_list)

        total_count = defaultdict(int)
        for result in results:
            for k in result:
                total_count[k] += result[k]

    return total_count

# def pre_tokenize_file(input_file: str, n_process: int, special_tokens: list[str]) -> dict[str, int]:
#     """
#     Pre-tokenize a file into a dictionary that map pre-tokens to ints
#     """
#     with open(input_file, "rb") as f:
#         boundaries = find_chunk_boundaries(f, n_process, b"<|endoftext|>")
#         arg_list = []
#         for start, end in zip(boundaries[:-1], boundaries[1:]):
#             f.seek(start)
#             chunk = f.read(end - start).decode("utf-8", errors="ignore")
#             arg_list.append([chunk, special_tokens])
#
#         with multiprocessing.Pool(len(arg_list)) as pool:
#             results = pool.starmap(pre_tokenize, arg_list)
#
#         total_count = defaultdict(int)
#         for result in results:
#             for k in result:
#                 total_count[k] += result[k]
#
#     return total_count


## Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-process", type=int, default=4)
    parser.add_argument("--input-file", type=str, default="../data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--output-file", type=str, default="../temp/TinyStoriesV2-GPT4-valid_pre_tokens.json")
    args = parser.parse_args()
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1e9  # in GB
    start_time = time.time()
    with open(args.input_file, "rb") as f:
        num_processes = args.n_process
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        arg_list = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            arg_list.append([chunk, ["<|endoftext|>"]])

        with multiprocessing.Pool(len(arg_list)) as pool:
            results = pool.starmap(pre_tokenize, arg_list)

        total_count = defaultdict(int)
        for result in results:
            for k in result:
                total_count[k] += result[k]

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(total_count, f)

    end_time = time.time()
    end_mem = process.memory_info().rss / 1e9
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Pretokenization took {elapsed_minutes:.2f} minutes")
    print(f"Pretokenization Memory used: {end_mem - start_mem:.2f} GB")

