import torch

from torch_semimarkov.triton_scan import semi_crf_forward_pytorch, semi_crf_triton_forward


def test_triton_forward_cpu_fallback_matches_pytorch():
    torch.manual_seed(0)
    batch, T, K, C = 2, 6, 4, 3
    edge = torch.randn(batch, T - 1, K, C, C, requires_grad=True)
    lengths = torch.full((batch,), T, dtype=torch.long)

    out = semi_crf_triton_forward(edge, lengths, use_triton=True)
    ref = semi_crf_forward_pytorch(edge.detach(), lengths)

    assert torch.allclose(out, ref, atol=1e-6)

    out.sum().backward()
    assert edge.grad is not None
    assert torch.isfinite(edge.grad).all()


def test_triton_validate_mode_matches_double():
    torch.manual_seed(1)
    batch, T, K, C = 1, 5, 3, 2
    edge = torch.randn(batch, T - 1, K, C, C)
    lengths = torch.full((batch,), T, dtype=torch.long)

    out = semi_crf_triton_forward(edge, lengths, validate=True)
    ref = semi_crf_forward_pytorch(edge.double(), lengths).to(edge.dtype)

    assert torch.allclose(out, ref, atol=1e-6)
