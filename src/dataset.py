"""
Dataset class and collation function for feeding transliteration
pairs into the neural network training pipeline.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.config import PAD_IDX


class TransliterationDataset(Dataset):
    """
    PyTorch Dataset for transliteration pairs.

    Each sample consists of:
        - Source (Roman): BPE-tokenized integer sequence
        - Target (Native): Character-level integer sequence with SOS/EOS

    Args:
        df: DataFrame with 'roman' and 'native' columns.
        pipeline: PreModelPipeline instance (provides tokenizer and normalizer).
        target_vocab: NativeVocab instance for encoding target strings.
    """
    def __init__(self, df, pipeline, target_vocab):
        self.df = df[["roman", "native"]].dropna().reset_index(drop=True)
        self.pipeline = pipeline
        self.trg_vocab = target_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_roman = str(self.df.iloc[idx]["roman"])
        native_word = str(self.df.iloc[idx]["native"])

        # Normalize exactly as inference does
        norm_roman = self.pipeline.normalize_roman(raw_roman)

        # Encode Source (Roman) using BPE Tokenizer
        encoded_src = self.pipeline.tokenizer.encode(norm_roman).ids
        src_tensor = torch.tensor(encoded_src, dtype=torch.long)

        # Encode Target (Devanagari) using Character Vocab (with SOS/EOS)
        encoded_trg = self.trg_vocab.encode(native_word)
        trg_tensor = torch.tensor(encoded_trg, dtype=torch.long)

        return src_tensor, trg_tensor


def collate_fn(batch):
    """
    Dynamic padding collation function for DataLoader.

    Pads all sequences in a batch to the length of the longest
    sequence using PAD_IDX.

    Args:
        batch: List of (src_tensor, trg_tensor) tuples.

    Returns:
        Tuple of (src_padded, trg_padded) tensors with shape [batch, max_seq_len].
    """
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        src_batch.append(src_item)
        trg_batch.append(trg_item)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)

    return src_padded, trg_padded
