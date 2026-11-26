"""Unit tests for the CTN pacemaker-detection model.

These tests are deterministic, CPU-only and use no patient data. They cover the
minimum properties a training pipeline depends on: construction, forward-pass
shape, loss computation, gradient flow and an optimisation step.
"""

import pytest
import torch
import torch.nn as nn

from src.model import CTN, PositionalEncoding, Transformer

# Deliberately small configuration: keeps the suite fast on CPU while exercising
# every layer type used by the published configuration.
D_MODEL = 64
N_HEAD = 4
D_FF = 128
NUM_LAYERS = 2
DROPOUT = 0.1
DEEPFEAT_SZ = 32
NB_FEATS = 10
NB_DEMO = 2
CLASSES = ["no_pacemaker", "pacemaker"]

BATCH = 2
LEADS = 12
SAMPLES = 2500  # 5 s at 500 Hz; comfortably larger than the conv receptive field


def build_model():
    """Return a small CTN instance in training mode."""
    return CTN(
        d_model=D_MODEL,
        nhead=N_HEAD,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT,
        deepfeat_sz=DEEPFEAT_SZ,
        nb_feats=NB_FEATS,
        nb_demo=NB_DEMO,
        classes=CLASSES,
    )


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return build_model()


@pytest.fixture()
def batch():
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(BATCH, LEADS, SAMPLES, generator=generator)
    y = torch.tensor([0, 1], dtype=torch.long)
    return x, y


def test_model_initialises_with_expected_output_dimension():
    model = build_model()

    assert isinstance(model, nn.Module)
    assert model.fc2.out_features == len(CLASSES)

    parameters = list(model.parameters())
    assert parameters, "model must expose trainable parameters"
    assert all(p.requires_grad for p in parameters)
    assert sum(p.numel() for p in parameters) > 0


def test_positional_encoding_preserves_shape_and_is_bounded(model):
    pe = PositionalEncoding(d_model=D_MODEL, dropout=0.0)
    x = torch.zeros(1, 8, D_MODEL)

    out = pe(x)

    assert out.shape == (1, 8, D_MODEL)
    # Sine/cosine encodings are bounded by 1.
    assert torch.all(out.abs() <= 1.0)


def test_transformer_returns_one_vector_per_sequence():
    torch.manual_seed(0)
    transformer = Transformer(D_MODEL, N_HEAD, D_FF, NUM_LAYERS, dropout=0.1)
    transformer.eval()

    # CTN passes (batch, channels, length) and Transformer permutes internally.
    out = transformer(torch.randn(BATCH, D_MODEL, 16))

    assert out.shape == (BATCH, D_MODEL)


def test_forward_pass_produces_finite_logits(model, batch):
    x, _ = batch
    model.eval()

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (BATCH, len(CLASSES))
    assert torch.isfinite(logits).all()


def test_forward_pass_is_deterministic_in_eval_mode(model, batch):
    x, _ = batch
    model.eval()

    with torch.no_grad():
        first = model(x)
        second = model(x)

    assert torch.allclose(first, second)


def test_cross_entropy_loss_is_a_finite_scalar(model, batch):
    x, y = batch
    criterion = nn.CrossEntropyLoss()

    loss = criterion(model(x), y)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_backward_pass_populates_gradients(model, batch):
    x, y = batch
    criterion = nn.CrossEntropyLoss()

    loss = criterion(model(x), y)
    loss.backward()

    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert encoder_grads, "encoder must receive gradients"
    assert any(g.abs().sum() > 0 for g in encoder_grads)

    assert model.fc2.weight.grad is not None
    assert torch.isfinite(model.fc2.weight.grad).all()


def test_one_optimiser_step_reduces_loss(model, batch):
    x, y = batch
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

    before = criterion(model(x), y).item()

    optimiser.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimiser.step()

    model.eval()
    with torch.no_grad():
        after = criterion(model(x), y).item()

    assert after < before, "a gradient step must reduce the training loss"
