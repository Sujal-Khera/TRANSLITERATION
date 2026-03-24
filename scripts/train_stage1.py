"""
Stage 1: Character/Subword Phonetics Training
Trains the model on Aksharantar data only to learn core
phonetic mappings (k→क, m→म, etc).
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
    PAD_IDX, DEVICE, BATCH_SIZE, GRAD_CLIP,
    STAGE1_CONFIG, DATA_DIR, TOKENIZER_PATH,
)
from src.preprocessing import PreModelPipeline
from src.vocab import NativeVocab
from src.dataset import TransliterationDataset, collate_fn
from src.model import Encoder, Decoder, TransliterationEngine, count_parameters

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


def sample_predictions(model, pipeline, trg_vocab, test_words, device):
    """Generate sample predictions for a list of test words."""
    from src.decoder import HybridDecoder
    decoder = HybridDecoder(model, pipeline, trg_vocab, device)
    model.eval()
    results = []
    for word in test_words:
        pred = decoder.transliterate(word)
        results.append({"input": word, "prediction": pred})
    return results


def main():
    print(f"Stage 1 Training — Device: {DEVICE}\n")

    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True

    # Load pipeline and tokenizer
    corpus_paths = [
        os.path.join(DATA_DIR, "aksharantar_train.csv"),
        os.path.join(DATA_DIR, "dakshina_train.csv"),
    ]
    pipeline = PreModelPipeline(corpus_paths=corpus_paths, vocab_size=5000)

    if os.path.exists(TOKENIZER_PATH):
        pipeline.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # Build vocabulary
    df_train = pd.read_csv(os.path.join(DATA_DIR, "aksharantar_train.csv"))
    trg_vocab = NativeVocab()
    trg_vocab.build_vocab(df_train["native"])

    # Create dataloaders
    train_dataset = TransliterationDataset(df_train, pipeline, trg_vocab)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    df_val = pd.read_csv(os.path.join(DATA_DIR, "aksharantar_val.csv"))
    val_dataset = TransliterationDataset(df_val, pipeline, trg_vocab)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Initialize model
    INPUT_VOCAB_SIZE = pipeline.tokenizer.get_vocab_size()
    OUTPUT_VOCAB_SIZE = trg_vocab.vocab_size

    enc = Encoder(INPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    dec = Decoder(OUTPUT_VOCAB_SIZE, pad_idx=PAD_IDX)
    model = TransliterationEngine(enc, dec, PAD_IDX, DEVICE).to(DEVICE)

    print(f"Source Vocab: {INPUT_VOCAB_SIZE} | Target Vocab: {OUTPUT_VOCAB_SIZE}")
    print(f"Parameters: {count_parameters(model):,}\n")

    optimizer = optim.Adam(model.parameters(), lr=STAGE1_CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Training loop with history logging
    config = STAGE1_CONFIG
    best_valid_loss = float("inf")
    save_path = os.path.join(DATA_DIR, config["save_name"])
    history_path = os.path.join(DATA_DIR, "stage1_history.json")

    # Words to test at each epoch
    sample_words = ["namaste", "bharat", "kolkata", "kisan", "sundar", "desh", "vidyalaya", "pariksha"]

    history = {
        "train_loss": [], "val_loss": [],
        "train_ppl": [], "val_ppl": [],
        "tf_ratio": [], "predictions": [],
    }

    print(f"--- Starting Stage 1 Training ({config['epochs']} Epochs) ---\n")

    try:
        for epoch in range(config["epochs"]):
            tf_ratio = max(config["tf_floor"], config["tf_start"] - epoch * config["tf_decay"])

            print(f"Epoch {epoch+1:02} | TF Ratio: {tf_ratio:.2f}")

            train_loss = train_epoch(model, train_loader, optimizer, criterion, GRAD_CLIP, tf_ratio)
            valid_loss = evaluate_epoch(model, val_loader, criterion)

            train_ppl = math.exp(train_loss)
            valid_ppl = math.exp(valid_loss)

            print(f"  Train Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}")
            print(f"  Val. Loss:  {valid_loss:.3f} | Val. PPL:  {valid_ppl:7.3f}")

            # Log metrics
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(valid_loss, 4))
            history["train_ppl"].append(round(train_ppl, 2))
            history["val_ppl"].append(round(valid_ppl, 2))
            history["tf_ratio"].append(round(tf_ratio, 4))

            # Sample predictions every 5 epochs (and epoch 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                preds = sample_predictions(model, pipeline, trg_vocab, sample_words, DEVICE)
                history["predictions"].append({"epoch": epoch + 1, "samples": preds})
                print(f"  Sample: {preds[0]['input']} → {preds[0]['prediction']}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), save_path)
                print(f"  [*] Best model saved to {save_path}")

            # Save history after every epoch (survives interrupts)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Checkpoints remain saved.")

    print(f"\n✓ Stage 1 Training Complete!")
    print(f"  History saved to {history_path}")
    print(f"  Run: python scripts/visualize_training.py stage1")


if __name__ == "__main__":
    main()
