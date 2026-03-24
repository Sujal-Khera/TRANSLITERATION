"""
Hybrid Inference Decoder
========================
Combines dictionary lookup with neural network inference
for production-quality transliteration.

Routing Logic:
    1. Dictionary Backoff (O(1) lookup) → for known common words
    2. Neural Greedy Decode → for out-of-vocabulary words
"""

import torch

from src.config import SOS_IDX, EOS_IDX, MAX_DECODE_LEN


class HybridDecoder:
    """
    Production-ready hybrid decoder that routes words through
    the fastest available path:
        - Common words → instant dictionary lookup
        - Rare/unknown words → neural network autoregressive decoding

    Args:
        model: Trained TransliterationEngine model (in eval mode).
        pipeline: PreModelPipeline instance with dictionary and tokenizer.
        trg_vocab: NativeVocab instance for decoding output indices.
        device: torch.device (cuda or cpu).
        max_word_len: Maximum number of characters to generate per word.
    """

    def __init__(self, model, pipeline, trg_vocab, device, max_word_len=MAX_DECODE_LEN):
        self.model = model
        self.pipeline = pipeline
        self.trg_vocab = trg_vocab
        self.device = device
        self.max_word_len = max_word_len

    def _decode_single_word_greedy(self, roman_word):
        """
        Neural Fallback: Greedy search (argmax) for OOV words.

        Generates one character at a time, always picking the
        highest-probability character until EOS is produced.

        Args:
            roman_word: Normalized Roman word string.

        Returns:
            Decoded Devanagari string.
        """
        encoded_src = self.pipeline.tokenizer.encode(roman_word).ids
        src_tensor = torch.tensor([encoded_src], dtype=torch.long).to(self.device)

        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)
            mask = self.model.create_mask(src_tensor)

            seq = [SOS_IDX]
            for _ in range(self.max_word_len):
                input_char = torch.tensor([seq[-1]]).to(self.device)
                output, hidden = self.model.decoder(
                    input_char, hidden, encoder_outputs, mask
                )

                top_pred = output.argmax(1).item()
                seq.append(top_pred)

                if top_pred == EOS_IDX:
                    break

            return self.trg_vocab.decode(seq)

    def transliterate(self, roman_text):
        """
        Transliterate a full Roman sentence into Devanagari.

        Routes each word through the optimal path:
            1. Dictionary lookup for known words
            2. Neural decoding for unknown words

        Args:
            roman_text: Input Roman text (e.g., "namaste doston").

        Returns:
            Devanagari string (e.g., "नमस्ते दोस्तों").
        """
        words = str(roman_text).lower().split()
        native_words = []

        for w in words:
            # ROUTE 1: O(1) Dictionary Lookup
            if hasattr(self.pipeline, "fast_lookup") and w in self.pipeline.fast_lookup:
                native_words.append(self.pipeline.fast_lookup[w])
                continue

            # ROUTE 2: Neural Network Fallback
            norm_w = self.pipeline.normalize_roman(w)
            native_pred = self._decode_single_word_greedy(norm_w)
            native_words.append(native_pred)

        return " ".join(native_words)
