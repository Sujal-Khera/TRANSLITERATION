"""
Phase 5: Evaluation & Benchmarking
====================================
Computes Character Error Rate (CER), Word Error Rate (WER),
and Top-k Accuracy on the test set.

Usage:
    python scripts/evaluate.py
"""

import os
import sys

import torch
import pandas as pd
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PAD_IDX, DEVICE, DATA_DIR, TOKENIZER_PATH, MASTER_CORPUS_DIR
from src.preprocessing import PreModelPipeline
from src.vocab import NativeVocab
from src.model import Encoder, Decoder, TransliterationEngine
from src.decoder import HybridDecoder


def character_error_rate(predicted, reference):
    """
    Compute Character Error Rate using edit distance.
    CER = edit_distance(pred, ref) / len(ref)
    """
    pred_chars = list(predicted)
    ref_chars = list(reference)

    # Dynamic programming for Levenshtein distance
    m, n = len(pred_chars), len(ref_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_chars[i - 1] == ref_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    if n == 0:
        return 0.0 if m == 0 else 1.0
    return dp[m][n] / n


def word_error_rate(predicted, reference):
    """
    Compute Word Error Rate.
    WER = edit_distance(pred_words, ref_words) / len(ref_words)
    """
    pred_words = predicted.split()
    ref_words = reference.split()

    m, n = len(pred_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == ref_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    if n == 0:
        return 0.0 if m == 0 else 1.0
    return dp[m][n] / n

def main():
    print(f"Phase 5: Evaluation — Device: {DEVICE}\n")

    # Load pipeline
    corpus_paths = [os.path.join(MASTER_CORPUS_DIR, "words_train.csv")]
    existing = [p for p in corpus_paths if os.path.exists(p)]
    pipeline = PreModelPipeline(corpus_paths=existing, vocab_size=5000)
    pipeline.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    INPUT_VOCAB_SIZE = pipeline.tokenizer.get_vocab_size()

    # Load vocabulary
    df_aksh = pd.read_csv(os.path.join(DATA_DIR, "aksharantar_train.csv"))
    trg_vocab = NativeVocab()
    trg_vocab.build_vocab(df_aksh["native"])
    OUTPUT_VOCAB_SIZE = trg_vocab.vocab_size

    # Load model
    enc = Encoder(INPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    dec = Decoder(OUTPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    model = TransliterationEngine(enc, dec, PAD_IDX, DEVICE).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(DATA_DIR, "stage2_best_model.pt"),
                    map_location=DEVICE, weights_only=True)
    )
    model.eval()

    decoder = HybridDecoder(model, pipeline, trg_vocab, DEVICE)

    # Load test data
    test_path = os.path.join(DATA_DIR, "aksharantar_test.csv")
    df_test = pd.read_csv(test_path)
    df_test = df_test[["roman", "native"]].dropna().reset_index(drop=True)

    # Evaluate on a sample (full test set can be very large)
    sample_size = min(5000, len(df_test))
    df_sample = df_test.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"Evaluating on {sample_size} samples...\n")

    total_cer = 0
    total_wer = 0
    exact_matches = 0

    for idx in range(len(df_sample)):
        roman = str(df_sample.iloc[idx]["roman"])
        reference = str(df_sample.iloc[idx]["native"])

        predicted = decoder.transliterate(roman)

        cer = character_error_rate(predicted, reference)
        wer = word_error_rate(predicted, reference)

        total_cer += cer
        total_wer += wer
        if predicted.strip() == reference.strip():
            exact_matches += 1

        if idx < 10:
            print(f"  [{idx+1}] {roman}")
            print(f"       Pred: {predicted}")
            print(f"       Ref:  {reference}")
            print(f"       CER: {cer:.4f}")
            print()

    avg_cer = total_cer / sample_size
    avg_wer = total_wer / sample_size
    accuracy = exact_matches / sample_size

    print("=" * 50)
    print("         EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Samples Evaluated:  {sample_size}")
    print(f"  Average CER:        {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"  Average WER:        {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"  Exact Match Acc:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
