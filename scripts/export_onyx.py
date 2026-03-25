"""
Phase 6: ONNX Export
=====================
Exports the trained Encoder and Decoder as separate ONNX graphs
for cross-platform deployment (mobile, C++, etc).

Usage:
    python scripts/export_onnx.py
"""

import os
import sys

import torch
import pandas as pd
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PAD_IDX, DEVICE, DATA_DIR, TOKENIZER_PATH, EMB_DIM, ENC_HID_DIM
from src.vocab import NativeVocab
from src.model import Encoder, Decoder, TransliterationEngine


def main():
    print("Phase 6: ONNX Export\n")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    INPUT_VOCAB_SIZE = tokenizer.get_vocab_size()

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

    # ==========================================
    # EXPORT ENCODER
    # ==========================================
    print("Exporting Encoder to ONNX...")
    dummy_src = torch.randint(1, INPUT_VOCAB_SIZE, (1, 10)).to(DEVICE)

    encoder_path = os.path.join(DATA_DIR, "encoder.onnx")
    torch.onnx.export(
        model.encoder,
        dummy_src,
        encoder_path,
        input_names=["source"],
        output_names=["encoder_outputs", "hidden"],
        dynamic_axes={
            "source": {0: "batch", 1: "src_len"},
            "encoder_outputs": {0: "batch", 1: "src_len"},
            "hidden": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,  # Use legacy TorchScript exporter (dynamo can't trace GRU)
    )
    encoder_size = os.path.getsize(encoder_path) / (1024 * 1024)
    print(f"-> Encoder saved: {encoder_path} ({encoder_size:.2f} MB)")

    # ==========================================
    # EXPORT DECODER (single step)
    # ==========================================
    print("\nExporting Decoder to ONNX...")
    dummy_input_char = torch.tensor([2]).to(DEVICE)   # SOS token
    dummy_hidden = torch.randn(1, ENC_HID_DIM).to(DEVICE)
    dummy_enc_out = torch.randn(1, 10, ENC_HID_DIM * 2).to(DEVICE)
    dummy_mask = torch.ones(1, 10, dtype=torch.bool).to(DEVICE)

    decoder_path = os.path.join(DATA_DIR, "decoder.onnx")
    torch.onnx.export(
        model.decoder,
        (dummy_input_char, dummy_hidden, dummy_enc_out, dummy_mask),
        decoder_path,
        input_names=["input_char", "hidden", "encoder_outputs", "mask"],
        output_names=["prediction", "new_hidden"],
        dynamic_axes={
            "encoder_outputs": {0: "batch", 1: "src_len"},
            "mask": {0: "batch", 1: "src_len"},
        },
        opset_version=17,
        dynamo=False,  # Use legacy TorchScript exporter
    )
    decoder_size = os.path.getsize(decoder_path) / (1024 * 1024)
    print(f"-> Decoder saved: {decoder_path} ({decoder_size:.2f} MB)")

    total_size = encoder_size + decoder_size
    print(f"\nTotal ONNX size: {total_size:.2f} MB")
    print("✓ ONNX Export Complete!")


if __name__ == "__main__":
    main()
