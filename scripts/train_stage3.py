"""
Stage 3: Sentence-Level Context Training
==========================================
Fine-tunes the Stage 2 model on sentence pairs with vocabulary
expansion surgery to handle spaces and sentence-level characters.

Usage:
    python scripts/train_stage3.py
"""

import os
import sys
import math
import json

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PAD_IDX, DEVICE, BATCH_SIZE_SENTENCES, GRAD_CLIP,
    STAGE3_CONFIG, DATA_DIR, SENTENCES_DIR, TOKENIZER_PATH,
)
from src.preprocessing import PreModelPipeline
from src.vocab import NativeVocab
from src.dataset import TransliterationDataset, collate_fn
from src.model import Encoder, Decoder, TransliterationEngine

from tokenizers import Tokenizer


def train_epoch(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):
    """Run a single training epoch."""
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(iterator, leave=False, desc="Training")

    for src, trg in progress_bar:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(iterator)


def evaluate_epoch(model, iterator, criterion):
    """Run a single evaluation epoch."""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def sample_predictions(model, pipeline, trg_vocab, test_sentences, device):
    """Generate sample predictions for test sentences."""
    from src.decoder import HybridDecoder
    decoder = HybridDecoder(model, pipeline, trg_vocab, device)
    model.eval()
    results = []
    for sent in test_sentences:
        pred = decoder.transliterate(sent)
        results.append({"input": sent, "prediction": pred})
    return results


def main():
    print(f"Stage 3 Training — Device: {DEVICE}\n")

    # Load tokenizer
    pipeline = PreModelPipeline(corpus_paths=None, vocab_size=5000)
    pipeline.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    INPUT_VOCAB_SIZE = pipeline.tokenizer.get_vocab_size()

    # Sequential vocabulary building (preserves Stage 1/2 indices)
    df_aksh_train = pd.read_csv(os.path.join(DATA_DIR, "aksharantar_train.csv"))
    df_sent_train = pd.read_csv(os.path.join(SENTENCES_DIR, "sentences_train.csv"))

    trg_vocab = NativeVocab()
    trg_vocab.build_vocab(df_aksh_train["native"])   # Lock Stage 1/2 chars (IDs 4-73)
    trg_vocab.build_vocab(df_sent_train["native"])    # Append sentence chars (IDs 74+)
    NEW_OUTPUT_VOCAB_SIZE = trg_vocab.vocab_size

    # Save the expanded vocabulary
    torch.save(trg_vocab, os.path.join(DATA_DIR, "trg_vocab.pt"))
    print(f"-> Expanded vocab saved ({NEW_OUTPUT_VOCAB_SIZE} chars)")

    # Create dataloaders
    df_sent_val = pd.read_csv(os.path.join(SENTENCES_DIR, "sentences_val.csv"))

    train_dataset = TransliterationDataset(df_sent_train, pipeline, trg_vocab)
    val_dataset = TransliterationDataset(df_sent_val, pipeline, trg_vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_SENTENCES, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_SENTENCES, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # Load Stage 2 model with vocabulary expansion surgery
    print("\nLoading Stage 2 Weights with Vocab Expansion Surgery...")
    stage2_path = os.path.join(DATA_DIR, "stage2_best_model.pt")
    old_state_dict = torch.load(stage2_path, map_location=DEVICE, weights_only=True)
    OLD_OUTPUT_VOCAB_SIZE = old_state_dict["decoder.embedding.weight"].shape[0]

    enc = Encoder(INPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    dec_new = Decoder(NEW_OUTPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    model = TransliterationEngine(enc, dec_new, PAD_IDX, DEVICE).to(DEVICE)

    # Weight surgery: copy old weights, initialize new vocab entries randomly
    new_state_dict = model.state_dict()
    for name, param in old_state_dict.items():
        if name in new_state_dict:
            if new_state_dict[name].shape == param.shape:
                new_state_dict[name] = param
            elif name in ["decoder.embedding.weight", "decoder.fc_out.weight", "decoder.fc_out.bias"]:
                new_state_dict[name][:param.shape[0]] = param

    model.load_state_dict(new_state_dict)
    print(f"-> Surgery complete: {OLD_OUTPUT_VOCAB_SIZE} → {NEW_OUTPUT_VOCAB_SIZE} output vocab")

    optimizer = optim.Adam(model.parameters(), lr=STAGE3_CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Training loop with history logging
    config = STAGE3_CONFIG
    best_valid_loss = float("inf")
    save_path = os.path.join(DATA_DIR, config["save_name"])
    history_path = os.path.join(DATA_DIR, "stage3_history.json")

    sample_sentences = [
        "mera naam sujal hai",
        "bharat ek sundar desh hai",
        "aaj mausam bahut accha hai",
        "namaste dosto",
    ]

    history = {
        "train_loss": [], "val_loss": [],
        "train_ppl": [], "val_ppl": [],
        "tf_ratio": [], "predictions": [],
    }

    print(f"\n--- Starting Stage 3 Training ({config['epochs']} Epochs) ---\n")

    try:
        for epoch in range(config["epochs"]):
            tf_ratio = max(config["tf_floor"], config["tf_start"] - epoch * config["tf_decay"])

            train_loss = train_epoch(model, train_loader, optimizer, criterion, GRAD_CLIP, tf_ratio)
            valid_loss = evaluate_epoch(model, val_loader, criterion)

            train_ppl = math.exp(train_loss)
            valid_ppl = math.exp(valid_loss)

            print(f"Epoch {epoch+1:02} | TF: {tf_ratio:.2f} | Train: {train_loss:.3f} (PPL {train_ppl:.1f}) | "
                  f"Val: {valid_loss:.3f} (PPL {valid_ppl:.1f})")

            # Log metrics
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(valid_loss, 4))
            history["train_ppl"].append(round(train_ppl, 2))
            history["val_ppl"].append(round(valid_ppl, 2))
            history["tf_ratio"].append(round(tf_ratio, 4))

            # Sample predictions every epoch (only 5 epochs total)
            preds = sample_predictions(model, pipeline, trg_vocab, sample_sentences, DEVICE)
            history["predictions"].append({"epoch": epoch + 1, "samples": preds})
            print(f"  Sample: {preds[0]['input']} → {preds[0]['prediction']}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), save_path)
                print(f"  [*] Best model saved")

            # Save history after every epoch
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Checkpoints remain saved.")

    print(f"\n✓ Stage 3 Training Complete!")
    print(f"  History saved to {history_path}")
    print(f"  Run: python scripts/visualize_training.py stage3")


if __name__ == "__main__":
    main()
