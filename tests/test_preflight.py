"""Hermetic tests for lightweight model preflight metadata and estimates."""

import json

from transformers import AutoConfig

from src.memory import preflight


def _write_model_fixture(tmp_path, weight_bytes=1 * (1024**2), include_weights=True):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 151936,
    }))
    if include_weights:
        (model_dir / "model.safetensors").write_bytes(b"0" * weight_bytes)
    return model_dir


def test_local_snapshot_derives_parameter_count_without_loading_weights(tmp_path):
    weight_bytes = 1 * (1024**2)
    model_dir = _write_model_fixture(tmp_path, weight_bytes=weight_bytes)

    metadata = preflight.get_model_metadata(str(model_dir))
    report = preflight.validate_model_preflight(str(model_dir), device="cpu")

    assert metadata.parameter_count == weight_bytes // 2
    assert report.parameter_count == weight_bytes // 2
    assert report.download_size_bytes == weight_bytes
    kv_cache_bytes = 2 * 24 * 2 * 64 * 2 * 2048
    cpu_overhead = 400 * (1024**2)
    assert report.runtime_memory_needed == weight_bytes + kv_cache_bytes + cpu_overhead
    assert report.fits_memory
    assert report.fits_disk
    assert report.can_run


def test_missing_local_weight_metadata_fails_closed(tmp_path):
    model_dir = _write_model_fixture(tmp_path, include_weights=False)

    metadata = preflight.get_model_metadata(str(model_dir))
    report = preflight.validate_model_preflight(str(model_dir), device="cpu")

    assert metadata.parameter_count is None
    assert metadata.weight_bytes == 0
    assert report.parameter_count is None
    assert report.runtime_memory_needed == 0
    assert report.download_size_bytes == 0
    assert not report.fits_memory
    assert not report.fits_disk
    assert not report.can_run


def test_unavailable_hub_metadata_fails_closed_offline(monkeypatch):
    monkeypatch.setattr(preflight, "HfApi", None)

    def unavailable_download(*args, **kwargs):
        raise OSError("offline test fixture")

    def unavailable_config(*args, **kwargs):
        raise OSError("offline test fixture")

    monkeypatch.setattr(preflight, "hf_hub_download", unavailable_download)
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        staticmethod(unavailable_config),
    )

    report = preflight.validate_model_preflight("offline/test-model", device="cpu")

    assert report.parameter_count is None
    assert report.download_size_bytes == 0
    assert report.runtime_memory_needed == 0
    assert not report.fits_memory
    assert not report.fits_disk
    assert not report.can_run


def test_list_installed_models_with_fixtures(tmp_path):
    """Hermetic test verifying local cache scanning distinguishes READY vs METADATA ONLY models."""
    cache_dir = tmp_path / "hub"
    cache_dir.mkdir()

    # 1. Complete model with safetensors
    repo1 = cache_dir / "models--test-org--model-ready"
    snap1 = repo1 / "snapshots" / "commit123"
    snap1.mkdir(parents=True)
    (snap1 / "config.json").write_text(json.dumps({"model_type": "qwen2", "hidden_size": 896}))
    (snap1 / "model.safetensors").write_bytes(b"0" * 1024)

    # 2. Metadata-only model
    repo2 = cache_dir / "models--test-org--model-meta"
    snap2 = repo2 / "snapshots" / "commit456"
    snap2.mkdir(parents=True)
    (snap2 / "config.json").write_text(json.dumps({"model_type": "bert", "hidden_size": 768}))

    models = preflight.list_installed_models(cache_dir=str(cache_dir))

    assert len(models) == 2
    # READY models sort first
    m_ready = models[0]
    m_meta = models[1]

    assert m_ready.repo_id == "test-org/model-ready"
    assert m_ready.status == "READY"
    assert m_ready.has_weights is True
    assert m_ready.model_type == "qwen2"
    assert "model.safetensors" in m_ready.weight_files

    assert m_meta.repo_id == "test-org/model-meta"
    assert m_meta.status == "METADATA ONLY"
    assert m_meta.has_weights is False
    assert m_meta.model_type == "bert"
    assert len(m_meta.weight_files) == 0


def test_print_installed_models_report(tmp_path, capsys):
    """Verifies report printer produces readable output and summaries."""
    cache_dir = tmp_path / "hub"
    cache_dir.mkdir()

    # Empty cache case
    models_empty = preflight.list_installed_models(cache_dir=str(cache_dir))
    preflight.print_installed_models_report(models_empty, cache_dir=str(cache_dir))
    captured = capsys.readouterr()
    assert "No locally installed or cached models found" in captured.out

    # Non-empty cache case
    repo = cache_dir / "models--org--my-model"
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (snap / "model.safetensors").write_bytes(b"0" * 2048)

    models = preflight.list_installed_models(cache_dir=str(cache_dir))
    preflight.print_installed_models_report(models, cache_dir=str(cache_dir), verbose=True)
    captured = capsys.readouterr()
    assert "LOCALLY INSTALLED & CACHED MODELS REPORT" in captured.out
    assert "org/my-model" in captured.out
    assert "READY" in captured.out


def test_cli_list_models_flag_parsing():
    """Verifies that -l, --list-models, and --models flags are parsed correctly."""
    from cli import build_parser

    parser = build_parser()

    args1 = parser.parse_args(["--list-models"])
    assert getattr(args1, "list_models", False) is True

    args2 = parser.parse_args(["-l"])
    assert getattr(args2, "list_models", False) is True

    args3 = parser.parse_args(["--models"])
    assert getattr(args3, "list_models", False) is True

    args4 = parser.parse_args(["check", "--list-models"])
    assert getattr(args4, "list_models", False) is True

    args5 = parser.parse_args(["list-models"])
    assert args5.subcommand == "list-models"
