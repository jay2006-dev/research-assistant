"""Unit and regression tests for provider abstractions and retry handling."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from asterline.providers import (
    AuthenticationError,
    FakeLLMProvider,
    FakeLLMResponse,
    MalformedOutputError,
    ProviderError,
    RateLimitError,
    SafeGeminiProvider,
    _extract_json,
    redact_secrets,
)


class DummyModel(BaseModel):
    name: str
    count: int


def test_redact_secrets_primary_and_additional():
    """redact_secrets redacts API keys, additional secrets, and AIza patterns."""
    raw = (
        "Error with key AIzaSyD12345678901234567890123456789012 and "
        "langsmith secret ls__supersecret12345"
    )
    sanitized = redact_secrets(raw, ["ls__supersecret12345"])
    assert "AIzaSy" not in sanitized
    assert "ls__supersecret12345" not in sanitized
    assert "[REDACTED]" in sanitized


def test_extract_json_handles_various_code_fences():
    """_extract_json handles clean fences, unclosed fences, and plain JSON."""
    clean = "```json\n{\"name\": \"test\", \"count\": 1}\n```"
    assert _extract_json(clean) == "{\"name\": \"test\", \"count\": 1}"

    unclosed = "```json\n{\"name\": \"truncated\", \"count\": 2}"
    assert _extract_json(unclosed) == "{\"name\": \"truncated\", \"count\": 2}"

    raw = "{\"name\": \"plain\", \"count\": 3}"
    assert _extract_json(raw) == "{\"name\": \"plain\", \"count\": 3}"


def test_provider_retry_zero_retries_does_not_crash():
    """max_retries=0 executes 1 attempt and raises ProviderError without TypeError."""
    provider = SafeGeminiProvider(api_key="test-key-fake-12345", max_retries=0)
    assert provider._max_retries == 0

    call_count = 0

    def failing_call(prompt: str):
        nonlocal call_count
        call_count += 1
        raise Exception("429 Resource exhausted: rate limit exceeded")

    with pytest.raises(RateLimitError):
        provider._invoke_with_retries(failing_call, "test prompt")

    assert call_count == 1, "Expected exactly 1 attempt when max_retries=0"


def test_provider_auth_error_never_retries():
    """Authentication errors raise immediately on the first attempt without sleeping."""
    provider = SafeGeminiProvider(api_key="test-key-fake-12345", max_retries=2)
    call_count = 0

    def auth_failing_call(prompt: str):
        nonlocal call_count
        call_count += 1
        raise Exception("401 Unauthorized: invalid api key")

    with pytest.raises(AuthenticationError):
        provider._invoke_with_retries(auth_failing_call, "test prompt")

    assert call_count == 1, "Authentication error must not be retried"


def test_invoke_structured_retries_malformed_json_and_succeeds(monkeypatch):
    """invoke_structured retries transient malformed output and parses successfully."""
    # Prevent real sleep during testing
    monkeypatch.setattr("time.sleep", lambda s: None)

    provider = SafeGeminiProvider(api_key="test-key-fake-12345", max_retries=2)

    responses = [
        "```json\n{\"name\": \"incomplete\", ",  # Broken JSON on attempt 1
        "```json\n{\"name\": \"valid\", \"count\": 42}\n```",  # Valid JSON on attempt 2
    ]

    def mock_call_llm(p: str) -> str:
        return responses.pop(0)

    monkeypatch.setattr(provider, "_call_llm", mock_call_llm)

    result = provider.invoke_structured("dummy prompt", DummyModel)
    assert isinstance(result, DummyModel)
    assert result.name == "valid"
    assert result.count == 42


def test_invoke_structured_exhausts_retries_and_raises_malformed_output(monkeypatch):
    """When all structured attempts return malformed output, exactly 1 + max_retries attempts run."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    provider = SafeGeminiProvider(api_key="test-key-fake-12345", max_retries=2)
    call_count = 0

    def mock_call_llm(p: str) -> str:
        nonlocal call_count
        call_count += 1
        return "this is not json at all"

    monkeypatch.setattr(provider, "_call_llm", mock_call_llm)

    with pytest.raises(MalformedOutputError):
        provider.invoke_structured("dummy prompt", DummyModel)

    # 1 initial + 2 retries = 3 attempts total
    assert call_count == 3, f"Expected exactly 3 attempts, got {call_count}"


def test_safe_gemini_from_settings_redacts_api_key_and_langsmith_key(caplog):
    """SafeGeminiProvider.from_settings registers both api_key and langsmith_api_key for redaction."""
    import logging
    from asterline.config import Settings

    settings = Settings(
        _env_file=None,
        api_key="secret-gemini-key-9999",
        langsmith_api_key="secret-ls-key-8888",
    )

    provider = SafeGeminiProvider.from_settings(settings, max_retries=0)
    assert "secret-gemini-key-9999" in provider._secrets
    assert "secret-ls-key-8888" in provider._secrets

    def failing_call(p: str):
        raise Exception("401 Unauthorized for secret-gemini-key-9999 with tracing key secret-ls-key-8888")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(AuthenticationError) as exc_info:
            provider._invoke_with_retries(failing_call, "test prompt")

    err_msg = str(exc_info.value)
    assert "secret-gemini-key-9999" not in err_msg
    assert "secret-ls-key-8888" not in err_msg
    assert "[REDACTED]" in err_msg
    assert exc_info.value.__cause__ is None, "Raw exception cause must not be chained"

    # Also verify captured log output
    log_text = caplog.text
    assert "secret-gemini-key-9999" not in log_text
    assert "secret-ls-key-8888" not in log_text


def test_provider_timeout_classification_and_redaction():
    """Timeout exceptions are mapped to TimeoutError and sanitize API keys."""
    from asterline.providers import TimeoutError as AsterlineTimeoutError

    provider = SafeGeminiProvider(api_key="secret-api-key-9999", max_retries=0)

    def timeout_call(p: str, timeout=None):
        raise Exception("504 Gateway Timeout: deadline exceeded for key secret-api-key-9999")

    with pytest.raises(AsterlineTimeoutError) as exc_info:
        provider._invoke_with_retries(timeout_call, "prompt")

    err_str = str(exc_info.value)
    assert "secret-api-key-9999" not in err_str
    assert "[REDACTED]" in err_str


def test_provider_retry_aborts_when_exceeding_deadline(monkeypatch):
    """Retry attempts abort immediately when backoff wait exceeds the remaining deadline."""
    import time
    from asterline.providers import TimeoutError as AsterlineTimeoutError

    provider = SafeGeminiProvider(api_key="secret-api-key-9999", max_retries=2)

    call_count = 0

    def failing_call(p: str, timeout=None):
        nonlocal call_count
        call_count += 1
        raise Exception("429 Resource exhausted: rate limit exceeded")

    # Deadline accommodates attempt 1 (needs >= 10.0s), but cannot accommodate attempt 1 + 2s wait + 10s retry 1 (12.0s)
    tight_deadline = time.time() + 11.0

    with pytest.raises(AsterlineTimeoutError) as exc_info:
        provider._invoke_with_retries(failing_call, "prompt", deadline=tight_deadline)

    assert "deadline exceeded; aborting retry" in str(exc_info.value).lower()
    # Exactly 1 attempt ran; retry 1 was aborted before backoff
    assert call_count == 1


def test_provider_deadline_handling_below_at_and_above_minimum():
    """Verify provider deadline enforcement below minimum, at minimum, and above minimum."""
    import time
    from asterline.providers import SafeGeminiProvider, TimeoutError as AsterlineTimeoutError

    provider = SafeGeminiProvider(api_key="secret-api-key-9999", timeout=30.0, max_retries=0)

    call_timeouts = []

    def recording_call(p: str, timeout=None):
        call_timeouts.append(timeout)
        return "success"

    # 1. Below provider minimum (8.0s remaining): immediately raises TimeoutError without calling model
    call_timeouts.clear()
    sub_min_deadline = time.time() + 8.0
    with pytest.raises(AsterlineTimeoutError) as exc_sub:
        provider._invoke_with_retries(recording_call, "prompt", deadline=sub_min_deadline)
    assert "below provider minimum deadline (10.0s)" in str(exc_sub.value)
    assert len(call_timeouts) == 0, "Underlying call must never be made when budget is below 10.0s minimum"

    # 2. Exactly at or marginally above minimum (10.05s remaining): call succeeds with minimum timeout
    call_timeouts.clear()
    at_min_deadline = time.time() + 10.05
    res_min = provider._invoke_with_retries(recording_call, "prompt", deadline=at_min_deadline)
    assert res_min == "success"
    assert len(call_timeouts) == 1
    assert 9.9 <= call_timeouts[0] <= 10.1

    # 3. Comfortably above minimum (20.0s remaining): call succeeds with reduced dynamic timeout
    call_timeouts.clear()
    above_min_deadline = time.time() + 20.0
    res_above = provider._invoke_with_retries(recording_call, "prompt", deadline=above_min_deadline)
    assert res_above == "success"
    assert len(call_timeouts) == 1
    assert 19.0 <= call_timeouts[0] <= 20.0


def test_provider_http_400_deadline_error_never_retried():
    """HTTP 400 INVALID_ARGUMENT (such as deadline too short) is classified non-retryable and never retried."""
    from asterline.providers import ProviderError, SafeGeminiProvider

    provider = SafeGeminiProvider(api_key="secret-api-key-9999", max_retries=2)

    call_count = 0

    def deadline_400_call(p: str, timeout=None):
        nonlocal call_count
        call_count += 1
        raise Exception(
            "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}"
        )

    with pytest.raises(ProviderError) as exc_info:
        provider._invoke_with_retries(deadline_400_call, "prompt", deadline=None)

    assert "Provider invalid argument" in str(exc_info.value)
    assert not exc_info.value.retryable
    # Must fail on attempt 1 and never retry
    assert call_count == 1


def test_provider_per_invocation_deadline_isolation():
    """Deadline is isolated per invocation; an expired deadline on call 1 does not poison call 2."""
    import time
    from asterline.providers import TimeoutError as AsterlineTimeoutError

    provider = SafeGeminiProvider(api_key="secret-api-key-9999", max_retries=0)

    # 1. Call with expired deadline raises immediately
    expired_deadline = time.time() - 1.0
    with pytest.raises(AsterlineTimeoutError):
        provider.invoke("prompt", deadline=expired_deadline)

    # 2. Subsequent call on the SAME provider with fresh deadline succeeds
    def mock_call(p: str, timeout=None):
        return "success response"

    provider._call_llm = mock_call
    fresh_deadline = time.time() + 60.0
    resp = provider.invoke("prompt", deadline=fresh_deadline)
    assert resp == "success response"


def test_safe_gemini_dynamic_http_timeout_enforcement():
    """Verify default HTTP timeout is 30s, reduced when deadline < 30s, and expired deadline prevents request."""
    import time
    from unittest.mock import MagicMock
    from google.genai import types
    from asterline.providers import TimeoutError as AsterlineTimeoutError

    provider = SafeGeminiProvider(api_key="secret-api-key-9999", timeout=30.0, max_retries=0)
    llm = provider.llm

    resp = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part.from_text(text="verified response")],
                    role="model",
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        model_version="gemini-2.5-flash",
    )
    mock_gen = MagicMock(return_value=resp)
    llm.client.models.generate_content = mock_gen

    # 1. Default invocation (no deadline): timeout must be 30000 ms (30.0s)
    mock_gen.reset_mock()
    out = provider.invoke("test prompt default")
    assert out == "verified response"
    assert mock_gen.called
    call_config = mock_gen.call_args.kwargs.get("config")
    assert call_config.http_options.timeout == 30000

    # 2. Tight deadline invocation (< 30s remaining): timeout passed to request is reduced
    mock_gen.reset_mock()
    tight_deadline = time.time() + 15.0
    out_tight = provider.invoke("test prompt tight", deadline=tight_deadline)
    assert out_tight == "verified response"
    assert mock_gen.called
    tight_config = mock_gen.call_args.kwargs.get("config")
    assert 10000 <= tight_config.http_options.timeout <= 15000

    # 3. Expired deadline: prevents request from ever being sent
    mock_gen.reset_mock()
    expired_deadline = time.time() - 1.0
    with pytest.raises(AsterlineTimeoutError):
        provider.invoke("test prompt expired", deadline=expired_deadline)
    assert not mock_gen.called, "Underlying model request must never be sent when deadline is already expired"


def test_safe_gemini_configures_sdk_max_retries_zero():
    """Constructed Google model client receives max_retries=0 so SafeGeminiProvider owns retries."""
    provider = SafeGeminiProvider(api_key="secret-key-12345", max_retries=2)
    assert provider._max_retries == 2
    assert provider.llm.max_retries == 0


def test_classify_provider_error_deadline_exceeded_and_typed_api_error():
    """DEADLINE_EXCEEDED and typed Google APIError exceptions are classified as TimeoutError."""
    import google.genai.errors as errors
    from asterline.providers import TimeoutError as AsterlineTimeoutError, _classify_provider_error

    # 1. String-based DEADLINE_EXCEEDED matching live error format
    raw_str_err = Exception(
        "504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', 'status': 'DEADLINE_EXCEEDED'}}"
    )
    classified_str = _classify_provider_error(raw_str_err)
    assert isinstance(classified_str, AsterlineTimeoutError)
    assert classified_str.retryable is True

    # 2. Typed google.genai.errors.APIError with status DEADLINE_EXCEEDED
    resp_json = {
        "error": {
            "code": 504,
            "message": "Deadline expired before operation could complete for secret-key-xyz.",
            "status": "DEADLINE_EXCEEDED",
        }
    }
    typed_err = errors.APIError(code=504, response_json=resp_json)
    classified_typed = _classify_provider_error(typed_err, secrets=["secret-key-xyz"])
    assert isinstance(classified_typed, AsterlineTimeoutError)
    assert classified_typed.retryable is True
    assert "secret-key-xyz" not in str(classified_typed)
    assert "[REDACTED]" in str(classified_typed)

    # 3. Standard Gateway Timeout
    gw_err = Exception("504 Gateway Timeout")
    classified_gw = _classify_provider_error(gw_err)
    assert isinstance(classified_gw, AsterlineTimeoutError)

    # 4. Non-timeout 504 does not blindly classify as TimeoutError
    other_err = Exception("504 Payload Corrupted Bad Format")
    classified_other = _classify_provider_error(other_err)
    assert not isinstance(classified_other, AsterlineTimeoutError)
    assert isinstance(classified_other, ProviderError)


def test_provider_outer_retries_remain_bounded_without_sdk_multiplication(monkeypatch):
    """Outer retries remain strictly bounded by SafeGeminiProvider without SDK retry loops."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    from asterline.providers import TimeoutError as AsterlineTimeoutError

    provider = SafeGeminiProvider(api_key="secret-api-key-9999", max_retries=2)
    assert provider.llm.max_retries == 0

    attempts = 0

    def mock_failing_llm(p: str, timeout=None):
        nonlocal attempts
        attempts += 1
        raise Exception("504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'status': 'DEADLINE_EXCEEDED'}}")

    provider._call_llm = mock_failing_llm

    with pytest.raises(AsterlineTimeoutError):
        provider.invoke("test prompt")

    # Exactly 1 initial + 2 retries = 3 total attempts
    assert attempts == 3


def test_timeout_failure_cannot_bypass_deterministic_validation(monkeypatch):
    """A provider timeout failure cannot bypass deterministic validation or produce false success."""
    from asterline.graph import create_research_graph
    from asterline.providers import TimeoutError as AsterlineTimeoutError, LLMProvider

    class TimingOutProvider(LLMProvider):
        def invoke(self, prompt: str, *, deadline=None) -> str:
            raise AsterlineTimeoutError("Upstream 504 DEADLINE_EXCEEDED")

        def invoke_structured(self, prompt: str, response_model: type, *, deadline=None):
            raise AsterlineTimeoutError("Upstream 504 DEADLINE_EXCEEDED")

    graph = create_research_graph(provider=TimingOutProvider())
    result = graph.invoke({"query": "Pricing info", "max_research_cycles": 2})

    outcome = result.get("outcome", "")
    assert outcome != "success"
    response = result.get("response", {})
    # No claims should survive as validated/supported
    assert len(response.get("claims", [])) == 0


def test_extract_text_from_content_plain_string():
    """1. Plain string content remains unchanged."""
    from asterline.providers import _extract_text_from_content

    raw = "Plain text response from LLM"
    assert _extract_text_from_content(raw) == raw


def test_extract_text_from_content_single_text_block():
    """2. A list containing one text block yields its text."""
    from asterline.providers import _extract_text_from_content

    blocks = [{"type": "text", "text": "Single text block"}]
    assert _extract_text_from_content(blocks) == "Single text block"


def test_extract_text_from_content_multiple_text_blocks_in_order():
    """3. Multiple text blocks are concatenated in order."""
    from asterline.providers import _extract_text_from_content

    blocks = [
        {"type": "text", "text": "Part 1: "},
        {"type": "text", "text": "Part 2: "},
        {"type": "text", "text": "Part 3"},
    ]
    assert _extract_text_from_content(blocks) == "Part 1: Part 2: Part 3"


def test_extract_text_from_content_mixed_blocks_ignores_non_text():
    """4. Mixed text and non-text blocks include only text."""
    from asterline.providers import _extract_text_from_content

    blocks = [
        {"type": "thinking", "thinking": "Internal reasoning process..."},
        {"type": "text", "text": "The answer is 42."},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        {"type": "executable_code", "code": "print(42)"},
    ]
    assert _extract_text_from_content(blocks) == "The answer is 42."


def test_extract_text_from_content_thinking_only_or_empty_fails_safely():
    """5. Thinking-only or empty content fails safely with MalformedOutputError."""
    import pytest
    from asterline.providers import MalformedOutputError, _extract_text_from_content

    # Thinking-only
    with pytest.raises(MalformedOutputError, match="no usable text content"):
        _extract_text_from_content([{"type": "thinking", "thinking": "Thinking only"}])

    # Empty list
    with pytest.raises(MalformedOutputError, match="no usable text content"):
        _extract_text_from_content([])

    # Empty string or whitespace-only string
    with pytest.raises(MalformedOutputError, match="empty text content"):
        _extract_text_from_content("")

    with pytest.raises(MalformedOutputError, match="empty text content"):
        _extract_text_from_content("   \n\t  ")

    # Unexpected type
    with pytest.raises(MalformedOutputError, match="Unexpected provider content format"):
        _extract_text_from_content(None)


def test_safe_gemini_call_llm_with_structured_blocks(monkeypatch):
    """6 & 7: Structured content blocks containing valid JSON reach parser correctly, and malformed JSON fails closed."""
    from langchain_core.messages import AIMessage
    from asterline.providers import SafeGeminiProvider, MalformedOutputError
    from pydantic import BaseModel

    class SimpleModel(BaseModel):
        answer: str
        count: int

    provider = SafeGeminiProvider(api_key="secret-key-12345", max_retries=0)

    # 7. Valid JSON inside structured content block
    valid_blocks = [
        {"type": "thinking", "thinking": "Evaluating query..."},
        {"type": "text", "text": "```json\n{\"answer\": \"verified\", \"count\": 10}\n```"},
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content=valid_blocks)
    provider._llm = mock_llm
    parsed = provider.invoke_structured("prompt", SimpleModel)
    assert parsed.answer == "verified"
    assert parsed.count == 10

    # 6. Malformed JSON inside structured content block fails closed
    malformed_blocks = [
        {"type": "thinking", "thinking": "Evaluating query..."},
        {"type": "text", "text": "This is not JSON at all"},
    ]
    mock_llm.invoke.return_value = AIMessage(content=malformed_blocks)
    with pytest.raises(MalformedOutputError):
        provider.invoke_structured("prompt", SimpleModel)


def test_provider_factory_gemini_default():
    """1. Gemini remains default provider when no provider is explicitly set or set to gemini."""
    from asterline.config import Settings
    from asterline.providers import SafeGeminiProvider, create_provider_from_settings

    # Default settings (no provider specified)
    s_default = Settings(_env_file=None, api_key="fake-gemini-key")
    p_default = create_provider_from_settings(s_default)
    assert isinstance(p_default, SafeGeminiProvider)

    # Explicit gemini
    s_gemini = Settings(_env_file=None, model_provider="gemini", api_key="fake-gemini-key")
    p_gemini = create_provider_from_settings(s_gemini)
    assert isinstance(p_gemini, SafeGeminiProvider)


def test_provider_factory_groq_selection():
    """2. Groq is selected only when explicitly configured via model_provider/llm_provider."""
    from asterline.config import Settings
    from asterline.providers import SafeGroqProvider, create_provider_from_settings

    s_groq = Settings(_env_file=None, model_provider="groq", groq_api_key="gsk-test-key-12345")
    p_groq = create_provider_from_settings(s_groq)
    assert isinstance(p_groq, SafeGroqProvider)
    assert p_groq._model_name == "openai/gpt-oss-120b"


def test_safe_groq_missing_credentials_fails_safely():
    """3. Missing Groq credentials fail safely with AuthenticationError."""
    from asterline.config import Settings
    from asterline.providers import AuthenticationError, SafeGroqProvider

    s_empty = Settings(_env_file=None, model_provider="groq", groq_api_key="")
    with pytest.raises(AuthenticationError, match="Groq API key is required"):
        SafeGroqProvider.from_settings(s_empty)


def test_safe_groq_text_and_structured_responses():
    """4. Text and structured JSON responses from Groq are extracted and parsed correctly."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from asterline.providers import SafeGroqProvider

    provider = SafeGroqProvider(api_key="gsk-test-key-12345", max_retries=0)
    mock_llm = MagicMock()
    provider._llm = mock_llm

    # Plain text extraction
    mock_llm.invoke.return_value = AIMessage(content="Direct response from Groq")
    assert provider.invoke("test prompt") == "Direct response from Groq"

    # Structured output extraction
    mock_llm.invoke.return_value = AIMessage(
        content='```json\n{"name": "groq_plan", "count": 42}\n```'
    )
    result = provider.invoke_structured("prompt", DummyModel)
    assert isinstance(result, DummyModel)
    assert result.name == "groq_plan"
    assert result.count == 42


def test_safe_groq_malformed_response_fails_closed():
    """5. Malformed responses fail closed with MalformedOutputError."""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from asterline.providers import MalformedOutputError, SafeGroqProvider

    provider = SafeGroqProvider(api_key="gsk-test-key-12345", max_retries=0)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Total gibberish not json")
    provider._llm = mock_llm

    with pytest.raises(MalformedOutputError):
        provider.invoke_structured("prompt", DummyModel)


def test_safe_groq_timeout_retries_and_secret_redaction(caplog):
    """6. Timeouts, retries, and secret redaction work correctly for Groq."""
    import logging
    from asterline.providers import SafeGroqProvider, TimeoutError as AsterlineTimeoutError

    provider = SafeGroqProvider(
        api_key="gsk-secret-groq-key-9999",
        additional_secrets=["secret-gemini-extra"],
        max_retries=0,
    )
    assert "gsk-secret-groq-key-9999" in provider._secrets
    assert "secret-gemini-extra" in provider._secrets

    def failing_call(p: str, timeout=None):
        raise Exception("504 Gateway Timeout for key gsk-secret-groq-key-9999")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(AsterlineTimeoutError) as exc_info:
            provider._invoke_with_retries(failing_call, "test prompt")

    err_text = str(exc_info.value)
    assert "gsk-secret-groq-key-9999" not in err_text
    assert "[REDACTED]" in err_text


def test_safe_groq_configures_sdk_max_retries_zero():
    """Groq client receives max_retries=0 so SafeGroqProvider owns retry decisions."""
    from asterline.providers import SafeGroqProvider

    provider = SafeGroqProvider(api_key="gsk-test-key-12345", max_retries=2)
    assert provider._max_retries == 2
    assert provider.llm.max_retries == 0


def test_provider_isolation_groq_failure_never_falls_back_to_gemini(monkeypatch):
    """Proves that selecting Groq instantiates only SafeGroqProvider and failures never fall back to Gemini."""
    from asterline.config import Settings
    from asterline.providers import (
        AuthenticationError,
        ProviderError,
        SafeGeminiProvider,
        SafeGroqProvider,
        create_provider_from_settings,
    )

    gemini_init_called = []
    original_gemini_init = SafeGeminiProvider.__init__

    def tracked_gemini_init(self, *args, **kwargs):
        gemini_init_called.append(True)
        return original_gemini_init(self, *args, **kwargs)

    monkeypatch.setattr(SafeGeminiProvider, "__init__", tracked_gemini_init)

    # 1. Groq selected: verify factory constructs only SafeGroqProvider and never touches SafeGeminiProvider
    cfg = Settings(
        _env_file=None,
        model_provider="groq",
        groq_api_key="gsk-isolated-test-key",
        api_key="gemini-backup-key-should-never-be-used",
    )
    provider = create_provider_from_settings(cfg)
    assert isinstance(provider, SafeGroqProvider)
    assert not isinstance(provider, SafeGeminiProvider)
    assert len(gemini_init_called) == 0

    # 2. Simulate Groq provider failure (auth or API failure): ensure error is raised directly, never falling back to Gemini
    def failing_invoke(*args, **kwargs):
        raise ProviderError("Groq 401 Unauthorized: Invalid API key")

    provider._invoke_with_retries = failing_invoke

    with pytest.raises(ProviderError) as exc_info:
        provider.invoke("test prompt")

    assert "Groq 401 Unauthorized" in str(exc_info.value)
    # Gemini must still never have been initialized or invoked
    assert len(gemini_init_called) == 0


def test_suppress_tracing_boundary_prevents_chat_model_child_runs():
    """suppress_tracing_boundary isolates LangChain ChatModel calls from trace collectors."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tracers.context import collect_runs

    from asterline.providers import suppress_tracing_boundary

    class MockChat(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        @property
        def _llm_type(self):
            return "mock-chat"

    chat = MockChat()
    secret_marker = "SYNTHETIC_PROMPT_SECRET_TOKEN_777"

    # 1. Without suppression: child run is captured with secret marker in inputs
    with collect_runs() as cb_unsuppressed:
        chat.invoke(f"Coordinator prompt with {secret_marker}")

    assert len(cb_unsuppressed.traced_runs) == 1
    assert secret_marker in str(cb_unsuppressed.traced_runs[0].model_dump())

    # 2. With suppression boundary: zero child runs are emitted
    with collect_runs() as cb_suppressed:
        with suppress_tracing_boundary():
            chat.invoke(f"Coordinator prompt with {secret_marker}")

    assert len(cb_suppressed.traced_runs) == 0
