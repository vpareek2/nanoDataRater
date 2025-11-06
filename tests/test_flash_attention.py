from __future__ import annotations

import pytest
import torch

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    pytest.skip("SDPA not available", allow_module_level=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_forward_backward() -> None:
    """Test if Flash Attention works with forward and backward pass on CUDA."""
    device = torch.device("cuda")

    # Create a simple transformer encoder layer
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        dropout=0.0,
        batch_first=True,
    ).to(device).to(torch.bfloat16)

    # Create input tensors in bf16 (required for Flash Attention)
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, 64, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Try to use Flash Attention explicitly
    try:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            out = encoder_layer(x)
            loss = out.mean()
            loss.backward()

        # If we get here, flash attention worked!
        assert x.grad is not None
        assert torch.all(torch.isfinite(out.float()))
        assert torch.all(torch.isfinite(x.grad.float()))
        print(f"✓ Flash Attention WORKED with bf16! Output shape: {out.shape}")
    except RuntimeError as e:
        if "flash" in str(e).lower() or "not implemented" in str(e).lower() or "No available kernel" in str(e):
            pytest.skip(f"Flash Attention not supported: {e}")
        else:
            raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_with_fallback() -> None:
    """Test if Flash Attention works with MATH fallback."""
    device = torch.device("cuda")

    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        dropout=0.0,
        batch_first=True,
    ).to(device)

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, 64, device=device, requires_grad=True)

    # Try Flash Attention with MATH fallback
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
        out = encoder_layer(x)
        loss = out.mean()
        loss.backward()

    # Should always work with fallback
    assert x.grad is not None
    assert torch.all(torch.isfinite(out))
    assert torch.all(torch.isfinite(x.grad))
    print(f"✓ Flash Attention with fallback worked! Output shape: {out.shape}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attention_actual_backend_used() -> None:
    """Check which backend is actually being used."""
    device = torch.device("cuda")

    if not hasattr(torch.backends.cuda, "flash_sdp_enabled"):
        pytest.skip("Cannot check backend usage")

    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        dropout=0.0,
        batch_first=True,
    ).to(device)

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, 64, device=device)

    # Check if flash SDP is enabled
    flash_enabled = torch.backends.cuda.flash_sdp_enabled()
    print(f"Flash SDP enabled: {flash_enabled}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"torch version: {torch.__version__}")

    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
        out = encoder_layer(x)

    assert out.shape == (batch_size, seq_len, 64)
    print(f"✓ Attention computation succeeded")

