#!/usr/bin/env python3
"""Example: Using SemiMarkovCRFHead with PyTorch Lightning.

This example demonstrates how to integrate torch-semimarkov with PyTorch Lightning
for distributed training of sequence segmentation models.

Key points:
1. SemiMarkovCRFHead wraps Triton kernels in a simple nn.Module
2. DDP works automatically - no special handling needed
3. Must use precision=32 for numerical stability at T > 100K

Usage:
    # Single GPU
    python lightning_integration.py --devices 1

    # Multi-GPU DDP
    python lightning_integration.py --devices 4 --strategy ddp

    # CPU (for testing)
    python lightning_integration.py --accelerator cpu
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Check for Lightning availability
try:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import LearningRateMonitor

    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    print("PyTorch Lightning not installed. Install with: pip install pytorch-lightning")

from torch_semimarkov import SemiMarkovCRFHead


class SimpleEncoder(nn.Module):
    """Simple encoder for demonstration purposes.

    In practice, replace this with your actual encoder (Mamba, Transformer, CNN, etc.)
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input token IDs of shape (batch, T)

        Returns:
            Hidden states of shape (batch, T, hidden_dim)
        """
        h = self.embedding(x)
        return self.layers(h)


class DummyDataset(Dataset):
    """Dummy dataset for demonstration.

    Generates random sequences with random segment labels.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 100,
        vocab_size: int = 5,
        num_classes: int = 10,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random sequence
        inputs = torch.randint(0, self.vocab_size, (self.seq_length,))

        # Random segment labels (create segments of varying length)
        labels = torch.zeros(self.seq_length, dtype=torch.long)
        pos = 0
        while pos < self.seq_length:
            # Random segment length (1 to 20)
            seg_len = torch.randint(1, 21, (1,)).item()
            seg_len = min(seg_len, self.seq_length - pos)
            # Random label
            label = torch.randint(0, self.num_classes, (1,)).item()
            labels[pos : pos + seg_len] = label
            pos += seg_len

        return {
            "inputs": inputs,
            "labels": labels,
            "lengths": torch.tensor(self.seq_length),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "inputs": torch.stack([item["inputs"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "lengths": torch.stack([item["lengths"] for item in batch]),
    }


if HAS_LIGHTNING:

    class SegmentationModel(L.LightningModule):
        """PyTorch Lightning module for sequence segmentation.

        This demonstrates how to use SemiMarkovCRFHead in a LightningModule
        for distributed training.
        """

        def __init__(
            self,
            vocab_size: int = 5,
            hidden_dim: int = 64,
            num_classes: int = 10,
            max_duration: int = 50,
            learning_rate: float = 1e-3,
        ):
            super().__init__()
            self.save_hyperparameters()

            # Encoder
            self.encoder = SimpleEncoder(vocab_size, hidden_dim)

            # CRF head (this wraps Triton streaming kernels)
            self.crf = SemiMarkovCRFHead(
                num_classes=num_classes,
                max_duration=max_duration,
                hidden_dim=hidden_dim,
            )

        def forward(self, inputs, lengths):
            """Forward pass returning partition function."""
            h = self.encoder(inputs)
            return self.crf(h, lengths)

        def training_step(self, batch, batch_idx):
            """Training step with NLL loss."""
            h = self.encoder(batch["inputs"])
            loss = self.crf.compute_loss(h, batch["lengths"], batch["labels"])

            # Log metrics (sync_dist=True for DDP)
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)

            return loss

        def validation_step(self, batch, batch_idx):
            """Validation step."""
            h = self.encoder(batch["inputs"])
            loss = self.crf.compute_loss(h, batch["lengths"], batch["labels"])

            self.log("val/loss", loss, prog_bar=True, sync_dist=True)

            return loss

        def configure_optimizers(self):
            """Configure optimizer with separate parameter groups.

            CRF parameters (transition, duration_bias) often benefit from
            a lower learning rate than the encoder.
            """
            encoder_params = list(self.encoder.parameters())
            crf_params = list(self.crf.parameters())

            optimizer = torch.optim.AdamW(
                [
                    {"params": encoder_params, "lr": self.hparams.learning_rate},
                    {
                        "params": crf_params,
                        "lr": self.hparams.learning_rate * 0.1,
                    },  # Lower LR for CRF
                ],
                weight_decay=0.01,
            )

            return optimizer


def main():
    parser = argparse.ArgumentParser(description="Lightning + SemiMarkovCRFHead Example")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy (ddp, etc)")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator (gpu, cpu)")
    parser.add_argument("--max_epochs", type=int, default=5, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--max_duration", type=int, default=50, help="Max segment duration")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    args = parser.parse_args()

    if not HAS_LIGHTNING:
        print("PyTorch Lightning is required for this example.")
        print("Install with: pip install pytorch-lightning")
        return

    # Create datasets
    train_dataset = DummyDataset(
        num_samples=1000,
        seq_length=args.seq_length,
        num_classes=args.num_classes,
    )
    val_dataset = DummyDataset(
        num_samples=100,
        seq_length=args.seq_length,
        num_classes=args.num_classes,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Create model
    model = SegmentationModel(
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        max_duration=args.max_duration,
    )

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    ]

    # Trainer
    # CRITICAL: precision=32 is required for numerical stability at T > 100K
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=32,  # REQUIRED for Semi-Markov CRF at scale
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    print(f"\nTraining with {args.devices} device(s), strategy={args.strategy}")
    print(f"Model: {args.num_classes} classes, max_duration={args.max_duration}")
    print(f"Sequence length: {args.seq_length}, batch size: {args.batch_size}\n")

    trainer.fit(model, train_loader, val_loader)

    print("\nTraining complete!")
    print(f"Final train loss: {trainer.callback_metrics.get('train/loss', 'N/A')}")
    print(f"Final val loss: {trainer.callback_metrics.get('val/loss', 'N/A')}")


if __name__ == "__main__":
    main()
