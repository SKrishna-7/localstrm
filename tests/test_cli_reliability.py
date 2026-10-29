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


def test_cli_trust_remote_code_flag_parsing():
    """Verify that --trust-remote-code is parsed and defaults to False across configs and CLI."""
    from cli import build_parser
    from src.config import EngineConfig, ModelConfig

    assert ModelConfig().trust_remote_code is False

    parser = build_parser()
    for subcmd in ("chat", "serve", "engine", "benchmark", "inspect"):
        args_default = parser.parse_args([subcmd])
        assert args_default.trust_remote_code is False

        args_enabled = parser.parse_args([subcmd, "--trust-remote-code"])
        assert args_enabled.trust_remote_code is True

    cfg_default = EngineConfig.from_args(parser.parse_args(["engine"]))
    assert cfg_default.trust_remote_code is False
    assert cfg_default.model.trust_remote_code is False

    cfg_enabled = EngineConfig.from_args(parser.parse_args(["engine", "--trust-remote-code"]))
    assert cfg_enabled.trust_remote_code is True
    assert cfg_enabled.model.trust_remote_code is True


def test_cmd_chat_stopping_on_im_end(capsys):
    """
    Regression test: Verify that cmd_chat halts token generation immediately when encountering
    <|im_end|> (151645) or generation_config list of EOS tokens, preventing infinite generation loops.
    """
    from cli import cmd_chat
    import torch

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 151643
    mock_tokenizer.unk_token_id = 0
    mock_tokenizer.chat_template = None
    mock_tokenizer.convert_tokens_to_ids.side_effect = lambda tok: 151645 if tok == "<|im_end|>" else 0
    mock_tokenizer.encode.return_value = torch.tensor([[10, 20]])
    mock_tokenizer.decode.side_effect = lambda ids, **kw: "Hello" if 100 in ids else ("" if 151645 in ids else " tok")

    mock_model = MagicMock()
    mock_model.generation_config = MagicMock()
    mock_model.generation_config.eos_token_id = [151645, 151643]

    # Step 0 (prefill): output token 100 ("Hello")
    out_prefill = MagicMock()
    out_prefill.logits = torch.zeros((1, 1, 200000))
    out_prefill.logits[0, 0, 100] = 10.0
    out_prefill.past_key_values = MagicMock()

    # Step 1: output token 151645 (<|im_end|>)
    out_step1 = MagicMock()
    out_step1.logits = torch.zeros((1, 1, 200000))
    out_step1.logits[0, 0, 151645] = 10.0
    out_step1.past_key_values = MagicMock()

    # Step 2: if loop didn't stop, would emit token 999
    out_step2 = MagicMock()
    out_step2.logits = torch.zeros((1, 1, 200000))
    out_step2.logits[0, 0, 999] = 10.0
    out_step2.past_key_values = MagicMock()

    mock_model.side_effect = [out_prefill, out_step1, out_step2]

    chat_args = DummyArgs(
        model="Qwen/Qwen2.5-Coder-1.5B",
        max_tokens=10,
        temperature=0.0,
        system=None,
        quant="none",
        trust_remote_code=False
    )

    # Simulate user sending one message "hi" then exiting via "/exit"
    inputs = iter(["hi", "/exit"])
    with patch("builtins.input", side_effect=inputs), \
         patch("src.memory.preflight.enforce_preflight_check"), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("cli.load_and_quantize_model", return_value=(mock_model, 1000, 1000)):
        cmd_chat(chat_args)

    # Ensure model was called only twice: prefill + step 1 (stopping at 151645)
    assert mock_model.call_count == 2
    captured = capsys.readouterr().out
    assert "Tokens:    2" in captured


def test_cmd_chat_prefill_immediate_eos(capsys):
    """
    Regression test: Verify that if the prefill token is an EOS token, cmd_chat halts immediately
    without entering the decode loop.
    """
    from cli import cmd_chat
    import torch

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 151643
    mock_tokenizer.unk_token_id = 0
    mock_tokenizer.chat_template = None
    mock_tokenizer.convert_tokens_to_ids.side_effect = lambda tok: 151645 if tok == "<|im_end|>" else 0
    mock_tokenizer.encode.return_value = torch.tensor([[10, 20]])
    mock_tokenizer.decode.return_value = ""

    mock_model = MagicMock()
    mock_model.generation_config = MagicMock()
    mock_model.generation_config.eos_token_id = 151645

    # Prefill outputs token 151645 (<|im_end|>)
    out_prefill = MagicMock()
    out_prefill.logits = torch.zeros((1, 1, 200000))
    out_prefill.logits[0, 0, 151645] = 10.0
    out_prefill.past_key_values = MagicMock()

    mock_model.return_value = out_prefill

    chat_args = DummyArgs(
        model="Qwen/Qwen2.5-Coder-1.5B",
        max_tokens=10,
        temperature=0.0,
        system=None,
        quant="none",
        trust_remote_code=False
    )

    inputs = iter(["hi", "/exit"])
    with patch("builtins.input", side_effect=inputs), \
         patch("src.memory.preflight.enforce_preflight_check"), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("cli.load_and_quantize_model", return_value=(mock_model, 1000, 1000)):
        cmd_chat(chat_args)

    # Ensure model was only called once for prefill
    assert mock_model.call_count == 1


def test_stopping_criteria_does_not_alias_unk_or_pad():
    """
    Regression test: Verify that candidate end tokens (<|im_end|>, <|eot_id|>) cannot
    accidentally alias unk_token_id, pad_token_id, or unmapped token IDs in other models (e.g. BERT/GPT-2).
    """
    from cli import get_stopping_criteria_token_ids

    # Case 1: Model with unk_token_id=100 where unsupported token maps to 100
    mock_tok_bert = MagicMock()
    mock_tok_bert.eos_token_id = 102
    mock_tok_bert.unk_token_id = 100
    mock_tok_bert.pad_token_id = 0
    mock_tok_bert.get_vocab.return_value = {"[CLS]": 101, "[SEP]": 102, "[UNK]": 100, "[PAD]": 0}
    mock_tok_bert.convert_tokens_to_ids.side_effect = lambda tok: 100

    stops = get_stopping_criteria_token_ids(mock_tok_bert)
    assert 100 not in stops, "unk_token_id was incorrectly added as a stop token!"
    assert 0 not in stops, "pad_token_id was incorrectly added as a stop token!"
    assert stops == {102}

    # Case 2: Tokenizer without get_vocab where convert_tokens_to_ids returns 99, but decode returns '<unk>'
    mock_tok_legacy = MagicMock(spec=["eos_token_id", "unk_token_id", "pad_token_id", "convert_tokens_to_ids", "decode"])
    mock_tok_legacy.eos_token_id = 50256
    mock_tok_legacy.unk_token_id = None
    mock_tok_legacy.pad_token_id = None
    mock_tok_legacy.convert_tokens_to_ids.return_value = 99
    mock_tok_legacy.decode.return_value = "<unk>"

    stops_legacy = get_stopping_criteria_token_ids(mock_tok_legacy)
    assert 99 not in stops_legacy, "Unmapped token ID was incorrectly added without vocabulary verification!"
    assert stops_legacy == {50256}

    # Case 3: Legitimate Qwen2.5 model supporting <|im_end|> and list eos_token_id
    mock_tok_qwen = MagicMock()
    mock_tok_qwen.eos_token_id = 151643
    mock_tok_qwen.unk_token_id = None
    mock_tok_qwen.pad_token_id = None
    mock_tok_qwen.get_vocab.return_value = {"<|endoftext|>": 151643, "<|im_end|>": 151645}

    mock_model_qwen = MagicMock()
    mock_model_qwen.generation_config.eos_token_id = [151645, 151643]

    stops_qwen = get_stopping_criteria_token_ids(mock_tok_qwen, mock_model_qwen)
    assert stops_qwen == {151643, 151645}

