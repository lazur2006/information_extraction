# loader.py
# Dataloader supports:
# 1) CRF word-level vocab encoding (neural CRF baseline)
# 2) Transformer inputs (subword) + CRF tags/mask (word-level) in one sample ("tfm_crf" mode)

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


# Utilities: tags + vocab

SPECIAL_TOKENS = ["PAD", "UNK"]


def compute_num_tags(df, fine_labels_enbl: bool) -> int:
    # +2 convention (max label id + 1) + 1 extra
    if fine_labels_enbl:
        max_label = max(df.ner_tags_index.explode().tolist())
    else:
        max_label = max(df.coarse_ner_tag_idx.explode().tolist())
    return int(max_label) + 2


def build_token2id(df) -> Dict[str, int]:
    tokens = sorted(list(set(df.tokens.explode().tolist())))
    token2id = {t: i for i, t in enumerate(SPECIAL_TOKENS + tokens)}
    return token2id


def get_labels(row, fine_labels_enbl: bool) -> List[int]:
    return row.ner_tags_index if fine_labels_enbl else row.coarse_ner_tag_idx


# Encoders

@dataclass(frozen=True)
class VocabEncoder:
    token2id: Dict[str, int]
    pad_token: str = "PAD"
    unk_token: str = "UNK"

    @property
    def pad_id(self) -> int:
        return self.token2id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token2id[self.unk_token]

    def encode_tokens(self, tokens: List[str], max_len: int) -> Tuple[List[int], List[bool]]:
        # Pad/truncate at word-level
        t_len = len(tokens)
        if t_len < max_len:
            tok_list = tokens + [self.pad_token] * (max_len - t_len)
        else:
            tok_list = tokens[:max_len]

        ids = [self.token2id.get(t, self.unk_id) for t in tok_list]
        mask = [False if i == self.pad_id else True for i in ids]
        return ids, mask


@dataclass(frozen=True)
class TransformerEncoder:
    tokenizer: Any  # e.g. transformers.AutoTokenizer (fast)

    def encode(
        self,
        tokens: List[str],
        max_len: int,
        add_special_tokens: bool = True,
        return_word_ids: bool = False,
    ) -> Dict[str, Any]:

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_attention_mask=True,
            add_special_tokens=add_special_tokens,
            return_tensors=None,
        )

        out: Dict[str, Any] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

        word_ids = None
        if return_word_ids:
            word_ids = enc.word_ids() if hasattr(enc, "word_ids") else None
            if word_ids is None:
                raise ValueError(
                    "Tokenizer output has no word_ids(). Ensure you use a fast tokenizer "
                    "(e.g. AutoTokenizer(..., use_fast=True))."
                )
            out["word_ids"] = word_ids

        return out


# Dataset

class MultiCoNERDataset(Dataset):
    def __init__(
        self,
        dataframe,
        max_seq_len: int,
        fine_labels_enbl: bool,
        pad_tag: int,
        encoder: Union[VocabEncoder, TransformerEncoder],
        output_format: str = "crf",  # "crf" or "tfm_crf"
        max_word_len: Optional[int] = None,  # used in "tfm_crf": word-level length for tags/mask
    ):
        super().__init__()
        if output_format not in ("crf", "tfm_crf"):
            raise ValueError("output_format must be 'crf' or 'tfm_crf'")

        self.df = dataframe
        self.max_seq_len = int(max_seq_len)  # word-len for VocabEncoder; subword-len for TransformerEncoder
        self.max_word_len = int(max_word_len) if max_word_len is not None else None
        self.fine_labels_enbl = bool(fine_labels_enbl)
        self.pad_tag = int(pad_tag)
        self.encoder = encoder
        self.output_format = output_format

        if self.output_format == "crf" and not isinstance(self.encoder, VocabEncoder):
            raise ValueError("output_format='crf' requires encoder=VocabEncoder")
        if self.output_format == "tfm_crf":
            if not isinstance(self.encoder, TransformerEncoder):
                raise ValueError("output_format='tfm_crf' requires encoder=TransformerEncoder")
            if self.max_word_len is None:
                raise ValueError("output_format='tfm_crf' requires max_word_len (e.g. MAX_WORDS)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        words: List[str] = list(row.tokens)
        labels: List[int] = list(get_labels(row, self.fine_labels_enbl))

        # VocabEncoder (word-level CRF baseline)
        if isinstance(self.encoder, VocabEncoder):
            ids, mask = self.encoder.encode_tokens(words, self.max_seq_len)

            # pad/truncate labels at word-level
            if len(labels) < self.max_seq_len:
                padded_labels = labels + [self.pad_tag] * (self.max_seq_len - len(labels))
            else:
                padded_labels = labels[:self.max_seq_len]

            seq_t = torch.tensor(ids, dtype=torch.long)
            tag_t = torch.tensor(padded_labels, dtype=torch.long)
            mask_t = torch.tensor(mask, dtype=torch.bool)

            return seq_t, tag_t, mask_t, words

        # TransformerEncoder
        if isinstance(self.encoder, TransformerEncoder):
            # Transformer inputs (subwords) + CRF tags/mask (words)
            if self.output_format == "tfm_crf":
                max_words = self.max_word_len

                # Cap at word-level for CRF supervision
                words_w  = words[:max_words]
                labels_w = labels[:max_words]

                # Tokenize capped words to subwords
                enc = self.encoder.encode(
                    tokens=words_w,
                    max_len=self.max_seq_len,   # max_subwords
                    add_special_tokens=True,
                    return_word_ids=True,
                )

                wid_raw = enc["word_ids"]
                wid = [w if w is not None else -1 for w in wid_raw]
                wid_t = torch.tensor(wid, dtype=torch.long)

                # Determine how many words survived subword truncation
                keep = [w for w in wid if w >= 0]
                survived = (max(keep) + 1) if keep else 0
                L = min(max_words, survived)

                # Word-level mask/tags for CRF (length=max_words)
                word_mask_t = torch.tensor([True] * L + [False] * (max_words - L), dtype=torch.bool)
                word_tags_t = torch.tensor(labels_w[:L] + [self.pad_tag] * (max_words - L), dtype=torch.long)

                return (
                    torch.tensor(enc["input_ids"], dtype=torch.long),
                    torch.tensor(enc["attention_mask"], dtype=torch.long),
                    wid_t,
                    word_tags_t,
                    word_mask_t,
                    words_w,
                )

        raise TypeError("encoder must be VocabEncoder or TransformerEncoder")


# Collate

def collate_multiconer(batch):
    first = batch[0]

    if isinstance(first, tuple):
        if len(first) == 4:
            seqs, tags, masks, words = zip(*batch)
            return torch.stack(seqs), torch.stack(tags), torch.stack(masks), list(words)

        if len(first) == 6:
            input_ids, attn, wids, tags, mask, words = zip(*batch)
            return (
                torch.stack(input_ids),
                torch.stack(attn),
                torch.stack(wids),
                torch.stack(tags),
                torch.stack(mask),
                list(words),
            )

        raise TypeError("Unsupported tuple size.")

    raise TypeError("Unsupported batch item type.")
