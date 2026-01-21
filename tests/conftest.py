"""
Pytest configuration for torch-semimarkov tests.

IMPORTANT: CPU-ONLY TESTING
---------------------------
This test suite is designed to run on CPU only. GPU support is not enabled
for CI due to cost constraints. The Triton kernels will automatically fall
back to CPU implementations when CUDA is not available.

All tests should:
1. Use CPU tensors (the default)
2. Not require CUDA to pass
3. Work correctly with the CPU fallback implementations
"""

import pytest
import torch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_cuda: mark test as requiring CUDA (will be skipped if not available)",
    )


@pytest.fixture(autouse=True)
def ensure_cpu_default():
    """
    Fixture that runs before each test to ensure we're using CPU.

    This is a documentation/verification fixture - it doesn't force CPU
    but warns if CUDA is being used unexpectedly.
    """
    # Just verify torch is available - tests should create CPU tensors by default
    assert torch.tensor([1.0]).device.type == "cpu", "Default device should be CPU"
    yield


@pytest.fixture
def cpu_device():
    """Fixture providing CPU device for explicit device specification."""
    return torch.device("cpu")


@pytest.fixture
def skip_if_no_cuda():
    """Fixture to skip tests that require CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# =============================================================================
# Clinical Domain Fixtures
# =============================================================================


@pytest.fixture
def clinical_ecg_config():
    """Configuration for ECG-style clinical data.

    ECG arrhythmia detection: classify heartbeat types (N, V, S, F, Q)
    with boundaries around QRS complexes.

    Typical sampling: 250-360 Hz
    Typical segment lengths: 10s-60s recordings
    """
    return {
        "num_classes": 5,  # N, V, S, F, Q beat types (MIT-BIH convention)
        "max_duration": 100,  # ~400ms at 250Hz (covers longest QRS + ST segment)
        "sample_rate": 250,
        "typical_lengths": [2500, 7500, 15000],  # 10s, 30s, 60s
        "d_model": 64,
        "description": "ECG arrhythmia detection with beat-level segmentation",
    }


@pytest.fixture
def clinical_eeg_config():
    """Configuration for EEG sleep staging data.

    Sleep staging: classify 30-second epochs into W, N1, N2, N3, REM.
    Features are typically extracted at lower rate (e.g., 10 Hz).

    Typical recordings: 4-12 hours
    """
    return {
        "num_classes": 5,  # Wake, N1, N2, N3, REM
        "max_duration": 300,  # 30s epochs at 10Hz feature rate
        "sample_rate": 10,  # Feature rate, not raw EEG
        "typical_lengths": [2400, 4800, 7200],  # 4h, 8h, 12h of features
        "d_model": 128,
        "description": "EEG sleep staging with epoch-level segmentation",
    }


@pytest.fixture
def clinical_genomics_config():
    """Configuration for genomic segmentation data.

    Gene annotation: segment DNA sequences into coding/non-coding regions.
    Very long sequences (chromosomes can be 100M+ bp).

    Typical segments: genes (1K-100K bp), exons (100-1000 bp)
    """
    return {
        "num_classes": 24,  # Multiple gene structure categories
        "max_duration": 3000,  # Max segment length in tokens
        "sample_rate": 1,  # 1 token per base pair (or per window)
        "typical_lengths": [10000, 50000, 100000, 400000],  # 10K-400K bp windows
        "d_model": 256,
        "description": "Genomic segmentation for gene structure annotation",
    }


@pytest.fixture
def create_synthetic_clinical_data():
    """Factory for creating synthetic clinical sequences with known boundaries.

    Returns a function that generates:
    - Features with segment-specific patterns
    - Labels with known boundaries
    - Segment length information for verification
    """

    def _create(num_classes, T, num_segments, noise_level=0.1, seed=None):
        """Create synthetic data with known segment boundaries.

        Args:
            num_classes: Number of label classes
            T: Total sequence length
            num_segments: Number of segments to generate
            noise_level: Standard deviation of noise (default 0.1)
            seed: Random seed for reproducibility

        Returns:
            features: (1, T, num_classes) tensor with segment-specific patterns
            labels: (1, T) tensor with per-position labels
            segment_info: List of (start, end, label, duration) tuples
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Create segment boundaries
        min_seg_len = max(5, T // (num_segments * 2))
        segment_lengths = torch.randint(
            min_seg_len, T // num_segments + min_seg_len, (num_segments,)
        )
        # Adjust to sum to T
        segment_lengths = (segment_lengths / segment_lengths.sum() * T).long()
        segment_lengths[-1] = T - segment_lengths[:-1].sum()  # Ensure sum = T

        # Create labels
        labels = torch.zeros(1, T, dtype=torch.long)
        features = torch.zeros(1, T, num_classes)

        segment_info = []
        pos = 0
        for i, length in enumerate(segment_lengths):
            length = length.item()
            if length <= 0:
                continue
            label = i % num_classes
            end = min(pos + length, T)

            labels[0, pos:end] = label
            # Features: high score for correct label, noise for others
            features[0, pos:end, label] = 1.0
            features[0, pos:end] += torch.randn(end - pos, num_classes) * noise_level

            segment_info.append((pos, end - 1, label, end - pos))
            pos = end

        return features, labels, segment_info

    return _create
