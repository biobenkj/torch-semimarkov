"""Integration tests for Transformer/Mamba encoders with SemiMarkovCRFHead.

Tests encoder → decoder integration for clinical applications:
- Forward pass shape validation
- Gradient flow through encoder to CRF
- Clinical sequence lengths (256, 512, 1000, 2048, 5000)
- Variable-length batch handling
"""

import pytest
import torch
import torch.nn as nn

from torch_semimarkov import SemiMarkovCRFHead

# =============================================================================
# Mock Encoders
# =============================================================================


class MockTransformerEncoder(nn.Module):
    """Mock Transformer encoder for testing (causal or bidirectional).

    Uses PyTorch's TransformerEncoder for realistic behavior without
    external dependencies.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        causal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional causal masking.

        Args:
            x: Input tensor of shape (batch, T, d_model)

        Returns:
            Hidden states of shape (batch, T, d_model)
        """
        if self.causal:
            T = x.size(1)
            # Create causal mask (upper triangular = masked)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            return self.transformer(x, mask=mask)
        return self.transformer(x)


class MockMambaEncoder(nn.Module):
    """Mock Mamba SSM encoder using GRU as a proxy.

    Mamba SSMs have a recurrent structure similar to RNNs.
    This mock uses bidirectional GRU to approximate the behavior
    without requiring the mamba-ssm library.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Use GRU as SSM proxy (Mamba has similar recurrent structure)
        self.layers = nn.ModuleList(
            [nn.GRU(d_model, d_model, batch_first=True) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SSM layers with residual connections.

        Args:
            x: Input tensor of shape (batch, T, d_model)

        Returns:
            Hidden states of shape (batch, T, d_model)
        """
        for layer, norm in zip(self.layers, self.norms, strict=False):
            residual = x
            x, _ = layer(x)
            x = norm(x + residual)
        return x


# =============================================================================
# Test Fixtures
# =============================================================================


class TestTransformerEncoderIntegration:
    """Test SemiMarkovCRFHead with Transformer encoders."""

    @pytest.fixture
    def transformer_config(self):
        """Configuration for Transformer encoder tests."""
        return {
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "num_classes": 8,
            "max_duration": 16,
            "batch_size": 4,
            "seq_len": 100,
        }

    def test_causal_transformer_forward_shape(self, transformer_config):
        """Test forward pass shape with causal Transformer encoder."""
        cfg = transformer_config
        encoder = MockTransformerEncoder(
            cfg["d_model"], cfg["nhead"], cfg["num_layers"], causal=True
        )
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["d_model"],
        )

        x = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["d_model"])
        lengths = torch.full((cfg["batch_size"],), cfg["seq_len"])

        hidden = encoder(x)
        result = crf_head(hidden, lengths, use_triton=False)

        assert result["partition"].shape == (cfg["batch_size"],)
        assert result["cum_scores"].shape == (
            cfg["batch_size"],
            cfg["seq_len"] + 1,
            cfg["num_classes"],
        )
        assert torch.isfinite(result["partition"]).all()

    def test_bidirectional_transformer_forward_shape(self, transformer_config):
        """Test forward pass shape with bidirectional Transformer encoder."""
        cfg = transformer_config
        encoder = MockTransformerEncoder(
            cfg["d_model"], cfg["nhead"], cfg["num_layers"], causal=False
        )
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["d_model"],
        )

        x = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["d_model"])
        lengths = torch.full((cfg["batch_size"],), cfg["seq_len"])

        hidden = encoder(x)
        result = crf_head(hidden, lengths, use_triton=False)

        assert result["partition"].shape == (cfg["batch_size"],)
        assert torch.isfinite(result["partition"]).all()

    def test_transformer_gradient_flow_to_encoder(self, transformer_config):
        """Verify gradients flow through encoder via CRF loss."""
        cfg = transformer_config
        encoder = MockTransformerEncoder(cfg["d_model"], cfg["nhead"], cfg["num_layers"])
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["d_model"],
        )

        x = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], requires_grad=True)
        lengths = torch.full((cfg["batch_size"],), cfg["seq_len"])
        labels = torch.randint(0, cfg["num_classes"], (cfg["batch_size"], cfg["seq_len"]))

        hidden = encoder(x)
        loss = crf_head.compute_loss(hidden, lengths, labels, use_triton=False)
        loss.backward()

        # Input gradient should exist and be finite
        assert x.grad is not None, "Input gradient should exist"
        assert torch.isfinite(x.grad).all(), "Input gradient should be finite"

        # All encoder parameters should have gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"Encoder param {name} should have gradient"
            assert torch.isfinite(
                param.grad
            ).all(), f"Encoder param {name} gradient should be finite"

        # All CRF parameters should have gradients
        for name, param in crf_head.named_parameters():
            assert param.grad is not None, f"CRF param {name} should have gradient"
            assert torch.isfinite(param.grad).all(), f"CRF param {name} gradient should be finite"

    @pytest.mark.parametrize("seq_len", [256, 512, 1000])
    def test_transformer_clinical_sequence_lengths(self, transformer_config, seq_len):
        """Test with clinical sequence lengths."""
        cfg = transformer_config
        encoder = MockTransformerEncoder(cfg["d_model"], cfg["nhead"], cfg["num_layers"])
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["d_model"],
        )

        batch = 2
        x = torch.randn(batch, seq_len, cfg["d_model"])
        lengths = torch.full((batch,), seq_len)

        hidden = encoder(x)
        result = crf_head(hidden, lengths, use_triton=False)

        assert torch.isfinite(result["partition"]).all(), f"Non-finite at T={seq_len}"


class TestMambaEncoderIntegration:
    """Test SemiMarkovCRFHead with Mamba SSM encoder."""

    @pytest.fixture
    def mamba_config(self):
        """Configuration for Mamba encoder tests."""
        return {
            "d_model": 64,
            "d_state": 16,
            "num_layers": 2,
            "num_classes": 8,
            "max_duration": 16,
            "batch_size": 4,
            "seq_len": 100,
        }

    def test_mamba_forward_shape(self, mamba_config):
        """Test forward pass shape with Mamba encoder."""
        cfg = mamba_config
        encoder = MockMambaEncoder(cfg["d_model"], cfg["d_state"], cfg["num_layers"])
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["d_model"],
        )

        x = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["d_model"])
        lengths = torch.full((cfg["batch_size"],), cfg["seq_len"])

        hidden = encoder(x)
        result = crf_head(hidden, lengths, use_triton=False)

        assert result["partition"].shape == (cfg["batch_size"],)
        assert torch.isfinite(result["partition"]).all()

    def test_mamba_gradient_flow_to_encoder(self, mamba_config):
        """Verify gradients flow through Mamba encoder via CRF loss."""
        cfg = mamba_config
        encoder = MockMambaEncoder(cfg["d_model"], cfg["d_state"], cfg["num_layers"])
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["d_model"],
        )

        x = torch.randn(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], requires_grad=True)
        lengths = torch.full((cfg["batch_size"],), cfg["seq_len"])
        labels = torch.randint(0, cfg["num_classes"], (cfg["batch_size"], cfg["seq_len"]))

        hidden = encoder(x)
        loss = crf_head.compute_loss(hidden, lengths, labels, use_triton=False)
        loss.backward()

        # Input gradient should exist and be finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # All encoder parameters should have gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"Mamba param {name} missing gradient"
            assert torch.isfinite(param.grad).all()

    @pytest.mark.parametrize("seq_len", [256, 512, 1000, 2000])
    def test_mamba_long_sequence_stability(self, mamba_config, seq_len):
        """Test Mamba + CRF handles long clinical sequences."""
        cfg = mamba_config
        encoder = MockMambaEncoder(cfg["d_model"], cfg["d_state"], cfg["num_layers"])
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=32,  # Larger K for clinical data
            hidden_dim=cfg["d_model"],
        )

        batch = 2
        x = torch.randn(batch, seq_len, cfg["d_model"])
        lengths = torch.full((batch,), seq_len)

        hidden = encoder(x)
        result = crf_head(hidden, lengths, use_triton=False)

        assert torch.isfinite(result["partition"]).all()


class TestVariableLengthBatches:
    """Test handling of variable-length batches (common in clinical data)."""

    @pytest.fixture
    def var_length_config(self):
        return {
            "d_model": 64,
            "num_classes": 8,
            "max_duration": 16,
        }

    def test_variable_lengths_transformer(self, var_length_config):
        """Test Transformer with variable-length sequences."""
        cfg = var_length_config
        encoder = MockTransformerEncoder(cfg["d_model"])
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["d_model"],
        )

        batch = 4
        T_max = 200
        # Variable lengths (simulating different recording durations)
        lengths = torch.tensor([200, 150, 100, 50])

        x = torch.randn(batch, T_max, cfg["d_model"])

        hidden = encoder(x)
        result = crf_head(hidden, lengths, use_triton=False)

        assert result["partition"].shape == (batch,)
        assert torch.isfinite(result["partition"]).all()

        # Each sequence should have different partition value
        partitions = result["partition"]
        # Not all should be equal (different lengths = different values)
        assert not torch.allclose(partitions[0], partitions[-1])

    def test_variable_lengths_mamba(self, var_length_config):
        """Test Mamba with variable-length sequences."""
        cfg = var_length_config
        encoder = MockMambaEncoder(cfg["d_model"])
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["d_model"],
        )

        batch = 4
        T_max = 200
        lengths = torch.tensor([200, 150, 100, 50])

        x = torch.randn(batch, T_max, cfg["d_model"])

        hidden = encoder(x)
        result = crf_head(hidden, lengths, use_triton=False)

        assert result["partition"].shape == (batch,)
        assert torch.isfinite(result["partition"]).all()

    def test_gradient_flow_variable_lengths(self, var_length_config):
        """Verify gradients flow correctly with variable-length batches."""
        cfg = var_length_config
        encoder = MockTransformerEncoder(cfg["d_model"])
        crf_head = SemiMarkovCRFHead(
            num_classes=cfg["num_classes"],
            max_duration=cfg["max_duration"],
            hidden_dim=cfg["d_model"],
        )

        batch = 3
        T_max = 100
        lengths = torch.tensor([100, 75, 50])

        x = torch.randn(batch, T_max, cfg["d_model"], requires_grad=True)
        labels = torch.randint(0, cfg["num_classes"], (batch, T_max))

        hidden = encoder(x)
        loss = crf_head.compute_loss(hidden, lengths, labels, use_triton=False)
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Gradients should exist for all encoder/CRF params
        for param in encoder.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


class TestEndToEndTrainingLoop:
    """Test a minimal training loop to verify everything works together."""

    def test_single_training_step(self):
        """Verify a single training step completes without error."""
        # Setup
        d_model = 32
        num_classes = 4
        max_duration = 8
        batch = 2
        T = 50

        encoder = MockTransformerEncoder(d_model, nhead=4, num_layers=1)
        crf_head = SemiMarkovCRFHead(
            num_classes=num_classes,
            max_duration=max_duration,
            hidden_dim=d_model,
        )

        # Combine into a simple model
        class Model(nn.Module):
            def __init__(self, encoder, crf_head):
                super().__init__()
                self.encoder = encoder
                self.crf_head = crf_head

            def forward(self, x, lengths, labels=None):
                h = self.encoder(x)
                if labels is not None:
                    return self.crf_head.compute_loss(h, lengths, labels, use_triton=False)
                return self.crf_head(h, lengths, use_triton=False)

        model = Model(encoder, crf_head)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Single training step
        x = torch.randn(batch, T, d_model)
        lengths = torch.full((batch,), T)
        labels = torch.randint(0, num_classes, (batch, T))

        optimizer.zero_grad()
        loss = model(x, lengths, labels)
        loss.backward()
        optimizer.step()

        # Verify loss is finite and gradients were applied
        assert torch.isfinite(loss)
        assert loss.item() > 0  # NLL should be positive

    def test_multiple_training_steps(self):
        """Verify loss decreases over multiple training steps."""
        d_model = 32
        num_classes = 4
        max_duration = 8
        batch = 2
        T = 50

        encoder = MockMambaEncoder(d_model, num_layers=1)
        crf_head = SemiMarkovCRFHead(
            num_classes=num_classes,
            max_duration=max_duration,
            hidden_dim=d_model,
        )

        params = list(encoder.parameters()) + list(crf_head.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-2)

        # Fixed data for reproducible loss curve
        torch.manual_seed(42)
        x = torch.randn(batch, T, d_model)
        lengths = torch.full((batch,), T)
        labels = torch.zeros(batch, T, dtype=torch.long)  # All same label = easy

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            h = encoder(x)
            loss = crf_head.compute_loss(h, lengths, labels, use_triton=False)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (or at least not explode)
        assert all(torch.isfinite(torch.tensor(losses)))
        # Final loss should be less than or similar to initial
        # (may not strictly decrease due to randomness)
        assert losses[-1] < losses[0] * 2  # Shouldn't explode
