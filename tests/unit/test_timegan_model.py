"""Unit tests for TimeGAN model components."""
import torch
import pytest

from src.layer1_timegan.timegan_model import (
    Embedder, Recovery, Generator, Supervisor, Discriminator,
    build_timegan, count_parameters,
)


class TestEmbedder:
    def test_output_shape(self):
        model = Embedder(n_features=11, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 24, 11)
        out = model(x)
        assert out.shape == (4, 24, 32)


class TestRecovery:
    def test_output_shape(self):
        model = Recovery(hidden_dim=32, n_features=11, num_layers=2)
        h = torch.randn(4, 24, 32)
        out = model(h)
        assert out.shape == (4, 24, 11)


class TestGenerator:
    def test_output_shape(self):
        model = Generator(noise_dim=32, hidden_dim=32, num_layers=2)
        z = torch.randn(4, 24, 32)
        out = model(z)
        assert out.shape == (4, 24, 32)


class TestSupervisor:
    def test_output_shape(self):
        model = Supervisor(hidden_dim=32, num_layers=2)
        h = torch.randn(4, 24, 32)
        out = model(h)
        assert out.shape == (4, 24, 32)


class TestDiscriminator:
    def test_output_shape(self):
        model = Discriminator(hidden_dim=32, num_layers=2)
        h = torch.randn(4, 24, 32)
        out = model(h)
        assert out.shape == (4, 1)

    def test_output_range(self):
        model = Discriminator(hidden_dim=32, num_layers=2)
        h = torch.randn(4, 24, 32)
        out = model(h)
        assert (out >= 0).all() and (out <= 1).all()


class TestBuildTimeGAN:
    def test_returns_all_components(self, sample_config):
        components = build_timegan(sample_config)
        assert set(components.keys()) == {"embedder", "recovery", "generator", "supervisor", "discriminator"}

    def test_autoencoder_roundtrip(self, sample_config):
        components = build_timegan(sample_config)
        x = torch.randn(2, 24, 11)
        h = components["embedder"](x)
        x_hat = components["recovery"](h)
        assert x_hat.shape == x.shape


class TestCountParameters:
    def test_count(self):
        model = Embedder(n_features=11, hidden_dim=32, num_layers=2)
        n = count_parameters(model)
        assert n > 0
