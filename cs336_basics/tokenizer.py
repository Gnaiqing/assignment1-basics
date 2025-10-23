import json
import regex as re
from typing import Iterable, Iterator
import argparse
from cs336_basics.train_bpe import unvisualize_text
from collections import OrderedDict
from typing import MutableMapping, Iterator, Tuple, Any, Optional


class LRUDict(MutableMapping[str, tuple[int, ...]]):
    """A tiny LRU cache with dict-like API.
       Stores values as tuples (slightly smaller than lists)."""
    def __init__(self, maxsize: int = 100_000):
        self.maxsize = maxsize
        self._d: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    # --- core mapping protocol ---
    def __getitem__(self, key: str) -> tuple[int, ...]:
        val = self._d.pop(key)          # move to end (most-recently used)
        self._d[key] = val
        return val

    def __setitem__(self, key: str, value: Any) -> None:
        # normalize to tuple to save a bit of memory
        if not isinstance(value, tuple):
            value = tuple(value)
        if key in self._d:
            self._d.pop(key)
        self._d[key] = value
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)  # evict least-recently used

    def __delitem__(self, key: str) -> None:
        del self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    # --- dict conveniences ---
    def get(self, key: str, default: Optional[tuple[int, ...]] = None):
        # mark-as-used on hit
        if key in self._d:
            return self[key]
        return default

    def clear(self) -> None:
        self._d.clear()


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # bytes -> id
        self.lookup = {tok: idx for idx, tok in vocab.items()}

        # --- SPEEDUP: pair -> rank (O(1) membership & ordering)
        # merges is an ordered list; earlier = lower rank
        self.rank = {pair: i for i, pair in enumerate(self.merges)}

        # --- SPEEDUP: precompile regexes once
        # Special token splitter
        if self.special_tokens:
            # longest-first avoids partial matches eating longer ones
            st_sorted = sorted(self.special_tokens, key=len, reverse=True)
            st_pat = "(" + "|".join(re.escape(tok) for tok in st_sorted) + ")"
            self._special_split_re = re.compile(st_pat)
        else:
            self._special_split_re = None

        # GPT-2-like pretokenizer (requires 'regex' library for \p{...})
        self._pretoken_re = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Optional: limit pretoken cache with an LRU-like dict if needed
        self.pretoken_encode = LRUDict(maxsize=100_000)

    def encode_pretoken(self, pretoken: str) -> list[int]:
        # bytes of the pretoken
        b = pretoken.encode("utf-8")

        # start as a list of single-byte tokens (bytes objects)
        # note: iterating bytes gives ints; slice to keep bytes objects
        tokens = [b[i:i + 1] for i in range(len(b))]

        if len(tokens) <= 1:
            return [self.lookup[tokens[0]]] if tokens else []

        rank = self.rank  # local bind for speed

        # Greedy BPE using rank dict:
        # Repeatedly find the adjacent pair with the best (lowest) rank and merge it.
        while True:
            best_i = -1
            best_rank = None

            # scan adjacent pairs once to find the best ranked pair
            for i in range(len(tokens) - 1):
                r = rank.get((tokens[i], tokens[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i

            if best_i < 0:
                break  # no mergeable pair remains

            # merge tokens[best_i] and tokens[best_i+1]
            tokens[best_i] = tokens[best_i] + tokens[best_i + 1]
            del tokens[best_i + 1]

        # map bytes to ids
        lookup = self.lookup  # local bind
        return [lookup[tok] for tok in tokens]

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
            vocab = {int(k): unvisualize_text(v) for k, v in vocab.items()}

        with open(merges_filepath, "r") as f:
            merges = []
            for line in f:
                if len(line.strip()) == 0:
                    continue
                t1, t2 = line.split()
                t1 = unvisualize_text(t1)
                t2 = unvisualize_text(t2)
                merges.append((t1, t2))

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        token_ids = []

        # split by special tokens if configured
        if self._special_split_re is not None:
            parts = [p for p in self._special_split_re.split(text) if p]
        else:
            parts = [text]

        pretoken_re = self._pretoken_re
        st_set = set(self.special_tokens)  # local, for O(1) checks
        cache = self.pretoken_encode
        lookup = self.lookup

        for part in parts:
            if part in st_set:
                # special token must exist in vocab as its UTF-8 bytes
                token_ids.append(lookup[part.encode("utf-8")])
                continue

            # normal text -> pretokens
            for m in pretoken_re.finditer(part):
                pre = m.group()
                enc = cache.get(pre)
                if enc is None:
                    enc = self.encode_pretoken(pre)
                    cache[pre] = enc
                token_ids.extend(enc)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            # yield from is fine here; keeps API the same
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # join once; avoid quadratic +=
        bytes_string = b"".join(self.vocab[idx] for idx in ids)
        return bytes_string.decode("utf-8", errors="replace")


class TokenizerOld:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens

        self.lookup = dict() # map token to ids
        for idx, token in vocab.items():
            self.lookup[token] = idx

        self.pretoken_encode = dict()

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
            vocab = {int(k): unvisualize_text(v) for k, v in vocab.items()}

        with open(merges_filepath, "r") as f:
            merges = []
            for line in f:
                if len(line.strip()) == 0:
                    continue
                t1, t2 = line.split()
                t1 = unvisualize_text(t1)
                t2 = unvisualize_text(t2)
                merges.append((t1, t2))

        return Tokenizer(vocab, merges, special_tokens)

    def encode_pretoken(self, pretoken: str) -> list[int]:
        """
        Encode a single pretoken into token ids
        """
        pretoken_bytes = pretoken.encode("utf-8")
        tokens = [] # list of bytes
        for i in range(len(pretoken_bytes)):
            tokens.append(pretoken_bytes[i:i + 1])

        while True:
            candidate_merge_idx = -1
            for i in range(len(tokens) - 1):
                if (tokens[i], tokens[i+1]) in self.merges:
                    merge_idx = self.merges.index((tokens[i], tokens[i+1]))
                    if candidate_merge_idx < 0 or merge_idx < candidate_merge_idx:
                        candidate_merge_idx = merge_idx

            if candidate_merge_idx < 0:
                break

            new_tokens = []
            for token in tokens:
                if len(new_tokens)> 0 and (new_tokens[-1], token) == self.merges[candidate_merge_idx]:
                    new_tokens[-1] = new_tokens[-1] + token
                else:
                    new_tokens.append(token)

            tokens = new_tokens

        token_ids = [self.lookup[token] for token in tokens]
        return token_ids

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token ids
        """
        token_ids = []
        if len(self.special_tokens) > 0:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # Escape for regex safety
            pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
            # Split and filter out empty strings
            docs = [p for p in re.split(pattern, text) if p]
        else:
            docs = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for doc in docs:
            if doc in self.special_tokens:
                # current split is a special token
                token_ids.append(self.lookup[doc.encode("utf-8")])
            else:
                # current split is a document
                for match in re.finditer(PAT, doc):
                    pretoken = match.group()
                    if pretoken not in self.pretoken_encode:
                        self.pretoken_encode[pretoken] = self.encode_pretoken(pretoken)

                    token_ids.extend(self.pretoken_encode[pretoken])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        return a generator that lazily yields token IDs.
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token ids to text
        """
        bytes_string = b""
        for idx in ids:
            bytes_string += self.vocab[idx]

        text = bytes.decode(bytes_string, encoding="utf-8", errors="replace")
        return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-path", type=str, default="../output/TinyStoriesV2-GPT4-valid-vocab.json")
    parser.add_argument("--merges-path", type=str, default="../output/TinyStoriesV2-GPT4-valid-merges.txt")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=None)
    test_string = "hello world"
    encoded_ids = tokenizer.encode(test_string)
    print("IDs:", encoded_ids)
    decoded_string = tokenizer.decode(encoded_ids)
    print("Decoded text:", decoded_string)











