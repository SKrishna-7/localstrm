"""
Unit and Integration Tests for CLI Error Hygiene & Reporting (Phase 8.1 Scope D)
---------------------------------------------------------------------------------
Validates:
  1. Safe step duration formatting when start_step is None.
  2. Distinct [SUCCESS] and [FAILED] request output cards.
  3. Proper exit code 1 when any request reaches terminal failure.
  4. Clean admission MemoryError handling without unhandled tracebacks.
  5. Clean exit code 0 when all requests succeed.
"""

import sys
import io
import pytest
from unittest.mock import MagicMock, patch
from src.engine.llm_engine import RequestStatus, EngineRequest
from cli import cmd_engine


class DummyArgs:
    def __init__(self, **kwargs):
        self.model = "Qwen/Qwen2.5-0.5B"
        self.device = "cpu"
        self.quant = "none"
        self.force = True
        self.prompt = ["Test prompt"]
        self.max_tokens = 8
        self.batch_size = 2
        self.total_blocks = 16
        self.block_size = 16
        self.metrics = False
        self.export_prometheus = None
        self.verbose = False
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_cli_safe_duration_calculation_with_none_start_step(capsys):
    """Ensure requests with start_step=None do not cause TypeError in cli.py."""
    req_success = EngineRequest(
        req_id=1,
        prompt="Hello world",
        max_tokens=8,
        arrival_step=0,
        status=RequestStatus.FINISHED,
        start_step=0,
        finish_step=3,
        output_token_ids=[10, 20, 30]
    )
    req_pre_admission_failed = EngineRequest(
        req_id=2,
        prompt="Impossible prompt",
        max_tokens=1000,
        arrival_step=0,
        status=RequestStatus.FAILED,
        stop_reason="kv_cache_full",
        error="request requires 100 blocks, pool has 16",
        start_step=None,
        finish_step=0
    )

    mock_engine = MagicMock()
    mock_engine.run.return_value = [req_success, req_pre_admission_failed]
    mock_engine.tokenizer.decode.return_value = "Generated text"
    mock_engine.last_run_time = 0.5

    args = DummyArgs()
    with patch("src.memory.preflight.enforce_preflight_check"), \
         patch("src.engine.llm_engine.MiniVLLMEngine", return_value=mock_engine):
        with pytest.raises(SystemExit) as excinfo:
            cmd_engine(args)
        assert excinfo.value.code == 1

    captured = capsys.readouterr().out
    assert "[Request #1]" in captured
    assert "[SUCCESS]" in captured
    assert "Duration: 4 steps" in captured
    assert "[Request #2]" in captured
    assert "[FAILED]" in captured
    assert "Failure Reason:" in captured
    assert "kv_cache_full" in captured
    assert "Details:" in captured
    assert "request requires 100 blocks, pool has 16" in captured
    assert "Duration: N/A" in captured
    assert "1 succeeded, 1 failed" in captured


def test_cli_admission_memory_error_handling(capsys):
    """Ensure engine admission MemoryError prints user hint and exits with code 1."""
    mock_engine = MagicMock()
    mock_engine.run.side_effect = MemoryError("request requires 50 KV blocks, but the pool has only 16")

    args = DummyArgs()
    with patch("src.memory.preflight.enforce_preflight_check"), \
         patch("src.engine.llm_engine.MiniVLLMEngine", return_value=mock_engine):
        with pytest.raises(SystemExit) as excinfo:
            cmd_engine(args)
        assert excinfo.value.code == 1

    captured = capsys.readouterr().out
    assert "Admission rejected due to insufficient memory" in captured
    assert "Hint: Increase --total-blocks or reduce --max-tokens / prompt length." in captured


def test_cli_all_successful_requests_exit_0(capsys):
    """Ensure all-successful execution exits with code 0."""
    req_success = EngineRequest(
        req_id=1,
        prompt="Valid prompt",
        max_tokens=8,
        arrival_step=0,
        status=RequestStatus.FINISHED,
        start_step=1,
        finish_step=4,
        output_token_ids=[101, 102]
    )
    mock_engine = MagicMock()
    mock_engine.run.return_value = [req_success]
    mock_engine.tokenizer.decode.return_value = "Healthy output"
    mock_engine.last_run_time = 0.2

    args = DummyArgs()
    with patch("src.memory.preflight.enforce_preflight_check"), \
         patch("src.engine.llm_engine.MiniVLLMEngine", return_value=mock_engine):
        # Should not raise SystemExit
        cmd_engine(args)

    captured = capsys.readouterr().out
    assert "[Request #1] \033[92m[SUCCESS]\033[0m" in captured
    assert "1 succeeded, 0 failed" in captured
    assert "Duration: 4 steps" in captured
