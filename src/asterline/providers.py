"""Gemini provider integration with safe exception handling and secret redaction.

Provides:
- Domain-specific exception hierarchy for provider errors.
- ``SafeGeminiProvider`` wrapping ``langchain-google-genai`` with bounded retries,
  error classification, and secret redaction.
- ``FakeLLMProvider`` for deterministic offline testing.

Secret handling
---------------
- API keys are read from configuration or environment variables, never hard-coded.
- Configured secrets and recognizable Gemini API-key patterns are redacted from
  error messages, traces, and repair prompts.
- Raw provider exceptions are never surfaced to callers; they are mapped to
  domain exceptions with sanitized messages.

Retry policy
------------
- Bounded retries (default 2, maximum 3) for transient failures only.
- No retries for authentication failures, invalid credentials, or other
  non-retryable errors.
- Provider retries are distinct from workflow iteration limits.
"""

from __future__ import annotations

import builtins
import logging
import re
import time
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Redaction ────────────────────────────────────────────────────────────────

# Matches Gemini/Google API key patterns (AIza...)
_GEMINI_KEY_PATTERN = re.compile(r"AIza[A-Za-z0-9_-]{35,}")


def redact_secrets(message: str, secrets: Sequence[str] = ()) -> str:
    """Redact configured secrets and recognizable API-key patterns from a message.

    Parameters
    ----------
    message : str
        The message to sanitize.
    secrets : Sequence[str]
        Additional secret strings to redact (e.g., from configuration).

    Returns
    -------
    str
        Sanitized message with secrets replaced by ``[REDACTED]``.
    """
    result = str(message)
    for secret in secrets:
        if secret and isinstance(secret, str):
            s = secret.strip()
            if len(s) > 3:
                result = result.replace(s, "[REDACTED]")
    result = _GEMINI_KEY_PATTERN.sub("[REDACTED]", result)
    return result


# ── Tracing Privacy Boundary ────────────────────────────────────────────────

from contextlib import contextmanager


@contextmanager
def suppress_tracing_boundary():
    """Context manager establishing a strict local boundary that suppresses tracing.

    Prevents sensitive child runs (such as coordinator LLM calls involving private
    user memories) from emitting trace spans to LangSmith, LangChain tracers, or
    run collectors.
    """
    try:
        from langchain_core.runnables.config import var_child_runnable_config
        from langchain_core.tracers.context import run_collector_var, tracing_v2_callback_var
        from langsmith.run_helpers import tracing_context

        token_cfg = var_child_runnable_config.set({})
        token_v2 = tracing_v2_callback_var.set(None)
        token_collector = run_collector_var.set(None)
        with tracing_context(enabled=False, parent=False):
            try:
                yield
            finally:
                var_child_runnable_config.reset(token_cfg)
                tracing_v2_callback_var.reset(token_v2)
                run_collector_var.reset(token_collector)
    except ImportError:
        # Fallback if tracing dependencies are not installed
        yield


# ── Provider Exceptions ─────────────────────────────────────────────────────

class ProviderError(Exception):
    """Base exception for all provider-related errors."""

    def __init__(self, message: str, *, retryable: bool = False):
        self.retryable = retryable
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Raised when API credentials are invalid or missing."""

    def __init__(self, message: str = "Authentication failed. Check your API key."):
        super().__init__(message, retryable=False)


class RateLimitError(ProviderError):
    """Raised when the provider rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded. Please wait and retry."):
        super().__init__(message, retryable=True)


class TimeoutError(ProviderError):
    """Raised when a provider request times out."""

    def __init__(self, message: str = "Provider request timed out."):
        super().__init__(message, retryable=True)


class MalformedOutputError(ProviderError):
    """Raised when the provider returns unparseable or structurally invalid output."""

    def __init__(self, message: str = "Provider returned malformed output."):
        super().__init__(message, retryable=True)


class ProviderUnavailableError(ProviderError):
    """Raised when the provider is temporarily unavailable."""

    def __init__(self, message: str = "Provider temporarily unavailable."):
        super().__init__(message, retryable=True)


# ── Error Classification ────────────────────────────────────────────────────

def _classify_provider_error(
    error: Exception, secrets: Sequence[str] = ()
) -> ProviderError:
    """Map a raw provider exception to a domain-specific ProviderError.

    Ensures raw exception messages are sanitized before propagation.
    """
    raw_message = str(error)
    safe_message = redact_secrets(raw_message, secrets)

    if isinstance(error, (TimeoutError, builtins.TimeoutError)):
        return TimeoutError(f"Timeout: {safe_message}")

    if isinstance(error, TypeError):
        return ProviderError(f"Provider TypeError: {safe_message}", retryable=False)

    # Typed Google GenAI API exception checks (e.g. google.genai.errors.APIError)
    err_status = str(getattr(error, "status", "")).upper()
    err_code = getattr(error, "code", None)

    if err_status in ("DEADLINE_EXCEEDED", "TIMEOUT"):
        return TimeoutError(f"Timeout: {safe_message}")

    if err_code == 504:
        err_msg = str(getattr(error, "message", "")).lower()
        if not err_msg or any(kw in err_msg for kw in ("deadline", "timeout", "timed out", "expired")):
            return TimeoutError(f"Timeout: {safe_message}")

    error_lower = raw_message.lower()

    # Explicit non-retryable 400 / INVALID_ARGUMENT classification
    if err_code == 400 or err_status == "INVALID_ARGUMENT" or any(
        kw in error_lower for kw in ("400", "invalid_argument", "invalid argument")
    ):
        return ProviderError(f"Provider invalid argument: {safe_message}", retryable=False)

    if any(kw in error_lower for kw in ("401", "403", "unauthenticated", "invalid api key",
                                         "api key not valid", "permission denied")):
        return AuthenticationError(f"Authentication error: {safe_message}")

    if any(kw in error_lower for kw in ("429", "rate limit", "resource exhausted", "quota")):
        return RateLimitError(f"Rate limit: {safe_message}")

    if any(kw in error_lower for kw in ("timeout", "deadline exceeded", "deadline_exceeded", "timed out")):
        return TimeoutError(f"Timeout: {safe_message}")

    if "504" in error_lower and any(kw in error_lower for kw in ("deadline", "timeout", "timed out", "expired", "gateway timeout")):
        return TimeoutError(f"Timeout: {safe_message}")

    if any(kw in error_lower for kw in ("500", "503", "unavailable", "internal")):
        return ProviderUnavailableError(f"Provider unavailable: {safe_message}")

    # Default: non-retryable provider error
    return ProviderError(f"Provider error: {safe_message}", retryable=False)


# ── Provider Protocol ────────────────────────────────────────────────────────

class LLMProvider:
    """Base interface for LLM providers."""

    def invoke(self, prompt: str, *, deadline: Optional[float] = None) -> str:
        """Send a prompt and return the text response."""
        raise NotImplementedError

    def invoke_structured(
        self, prompt: str, response_model: type, *, deadline: Optional[float] = None
    ) -> Any:
        """Send a prompt and parse the response into a Pydantic model.

        Falls back to JSON extraction from text if structured output is unavailable.
        """
        raise NotImplementedError


# ── Fake Provider for Testing ────────────────────────────────────────────────

class FakeLLMResponse(BaseModel):
    """Configurable fake response for deterministic testing."""

    text: str = ""
    structured: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class FakeLLMProvider(LLMProvider):
    """Deterministic offline provider for routine testing.

    Accepts a queue of responses. Each ``invoke`` call dequeues the next response.
    If the queue is exhausted, returns a default response.
    """

    def __init__(
        self,
        responses: Optional[list[FakeLLMResponse]] = None,
        default_text: str = "Fake LLM response",
    ):
        self._responses: list[FakeLLMResponse] = list(responses or [])
        self._default_text = default_text
        self._call_count = 0
        self._call_history: list[str] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_history(self) -> list[str]:
        return list(self._call_history)

    def _next_response(self) -> FakeLLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return FakeLLMResponse(text=self._default_text)

    def invoke(self, prompt: str, *, deadline: Optional[float] = None) -> str:
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("FakeLLMProvider: workflow execution deadline exceeded.")
        self._call_count += 1
        self._call_history.append(prompt)
        resp = self._next_response()
        if resp.error:
            raise ProviderError(resp.error)
        return resp.text

    def invoke_structured(
        self, prompt: str, response_model: type, *, deadline: Optional[float] = None
    ) -> Any:
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("FakeLLMProvider: workflow execution deadline exceeded.")
        import json as json_mod

        self._call_count += 1
        self._call_history.append(prompt)
        resp = self._next_response()
        if resp.error:
            raise ProviderError(resp.error)
        if resp.structured is not None:
            return response_model(**resp.structured)
        # Try to parse text as JSON
        try:
            json_str = _extract_json(resp.text)
            data = json_mod.loads(json_str)
            return response_model(**data)
        except Exception:
            raise MalformedOutputError(
                f"FakeLLMProvider: could not parse response as {response_model.__name__}"
            )


# ── JSON Extraction Helper ───────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Extract JSON content from fenced code blocks or raw text safely.

    Handles complete code fences, missing closing fences, and raw JSON strings.
    """
    stripped = text.strip()
    if "```json" in stripped:
        start = stripped.find("```json") + len("```json")
        end = stripped.rfind("```")
        if end > start:
            return stripped[start:end].strip()
        return stripped[start:].strip()
    if "```" in stripped:
        start = stripped.find("```") + 3
        end = stripped.rfind("```")
        if end > start:
            return stripped[start:end].strip()
        return stripped[start:].strip()
    return stripped


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from LangChain AIMessage.content.

    Handles string content and Gemini 3+ structured content block lists.
    Extracts and concatenates only text blocks in original order, ignoring
    thinking/reasoning, image, audio, or tool call blocks.
    Raises MalformedOutputError if no usable text content exists.
    """
    if isinstance(content, str):
        if not content.strip():
            raise MalformedOutputError("Provider returned empty text content.")
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    t = block.get("text", "")
                    if isinstance(t, str) and t:
                        text_parts.append(t)
        combined = "".join(text_parts)
        if not combined.strip():
            raise MalformedOutputError("Provider returned no usable text content.")
        return combined

    raise MalformedOutputError(
        f"Unexpected provider content format: {type(content).__name__}"
    )


# ── Safe Gemini Provider ─────────────────────────────────────────────────────

class SafeGeminiProvider(LLMProvider):
    """Production Gemini provider with bounded retries and secret redaction.

    Wraps ``langchain-google-genai.ChatGoogleGenerativeAI`` behind a safe
    boundary that maps all provider exceptions to domain types.

    Parameters
    ----------
    api_key : str
        Google Gemini API key. Read from configuration, never hard-coded.
    model_name : str
        Gemini model identifier (e.g., ``gemini-2.5-flash``).
    max_retries : int
        Maximum retries for transient failures (default 2, hard cap 3).
    temperature : float
        Sampling temperature for generation.
    additional_secrets : Sequence[str]
        Additional sensitive strings (e.g. LangSmith keys) to redact from logs/errors.
    """

    MIN_REQUEST_TIMEOUT: float = 10.0

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 2,
        temperature: float = 0.1,
        timeout: float = 30.0,
        additional_secrets: Sequence[str] = (),
    ):
        if not api_key:
            raise AuthenticationError("API key is required but was empty or not provided.")
        self._api_key = api_key
        self._model_name = model_name
        self._max_retries = max(0, min(max_retries, 3))  # Hard cap at 3, non-negative
        self._temperature = temperature
        self._timeout = max(self.MIN_REQUEST_TIMEOUT, timeout)
        all_secrets = [api_key]
        for s in additional_secrets:
            if s and isinstance(s, str):
                cleaned = s.strip()
                if len(cleaned) > 3:
                    all_secrets.append(cleaned)
        self._secrets = all_secrets
        self._llm = None

    @classmethod
    def from_settings(cls, settings: Any, **kwargs: Any) -> SafeGeminiProvider:
        """Create a SafeGeminiProvider configured from application Settings.

        Automatically registers configured secrets (including langsmith_api_key)
        for redaction across all provider logs and exception messages.
        """
        additional = []
        langsmith_key = getattr(settings, "langsmith_api_key", None)
        if langsmith_key:
            additional.append(langsmith_key)
        req_timeout = getattr(settings, "request_timeout", 30.0)
        return cls(
            api_key=getattr(settings, "api_key", ""),
            model_name=getattr(settings, "model_name", "gemini-2.5-flash"),
            timeout=req_timeout,
            additional_secrets=additional,
            **kwargs,
        )

    @property
    def llm(self):
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=self._api_key,
                temperature=self._temperature,
                timeout=self._timeout,
                max_retries=0,
            )
        return self._llm

    def _invoke_with_retries(self, call_fn, prompt: str, *, deadline: Optional[float] = None) -> Any:
        """Execute a provider call with bounded outer retries and cooperative deadline enforcement.

        Deadline enforcement is cooperative: checked before calls, after calls, and before/after
        retry backoff. Individual HTTP requests are bounded by the effective remaining timeout,
        with SDK-level internal retries disabled (max_retries=0) so that SafeGeminiProvider is the
        sole owner of retry decisions.
        """
        total_attempts = 1 + self._max_retries
        last_error: ProviderError = ProviderError("No provider attempts configured", retryable=False)

        for attempt in range(1, total_attempts + 1):
            now = time.time()
            # 1. Deadline check before provider call
            if deadline is not None:
                time_left = deadline - now
                if time_left <= 0:
                    raise TimeoutError("Overall workflow execution deadline exceeded before provider call.")
                if time_left < self.MIN_REQUEST_TIMEOUT:
                    raise TimeoutError(
                        f"Remaining workflow budget ({time_left:.2f}s) is below provider minimum "
                        f"deadline ({self.MIN_REQUEST_TIMEOUT:.1f}s)."
                    )
                effective_timeout = min(self._timeout, time_left)
            else:
                effective_timeout = self._timeout

            try:
                try:
                    res = call_fn(prompt, timeout=effective_timeout)
                except TypeError:
                    res = call_fn(prompt)
                # 2. Deadline check after provider call
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("Overall workflow execution deadline exceeded after provider call.")
                return res
            except Exception as raw_err:
                if isinstance(raw_err, ProviderError):
                    classified = raw_err
                else:
                    classified = _classify_provider_error(raw_err, self._secrets)
                last_error = classified

                if not classified.retryable:
                    logger.error(
                        "Non-retryable provider error (attempt %d/%d): %s",
                        attempt,
                        total_attempts,
                        redact_secrets(str(classified), self._secrets),
                    )
                    raise classified from None

                if attempt < total_attempts:
                    wait = min(2 ** attempt, 8)
                    # 3. Deadline check before retry attempt and backoff
                    if deadline is not None:
                        time_left = deadline - time.time()
                        needed_budget = wait + self.MIN_REQUEST_TIMEOUT
                        # Never start a retry that cannot reasonably fit within remaining deadline
                        if time_left < needed_budget:
                            logger.warning(
                                "Aborting provider retry: remaining deadline (%.2fs) cannot accommodate "
                                "backoff (%ds) and next attempt minimum deadline (%.1fs).",
                                time_left,
                                wait,
                                self.MIN_REQUEST_TIMEOUT,
                            )
                            raise TimeoutError(
                                f"Overall workflow execution deadline exceeded; aborting retry: {classified}"
                            ) from None

                    logger.warning(
                        "Retryable provider error (attempt %d/%d), waiting %ds: %s",
                        attempt,
                        total_attempts,
                        wait,
                        redact_secrets(str(classified), self._secrets),
                    )
                    time.sleep(wait)
                    # 4. Deadline check after retry backoff
                    if deadline is not None and time.time() >= deadline:
                        raise TimeoutError(
                            "Overall workflow execution deadline exceeded during retry backoff."
                        )
                else:
                    logger.error(
                        "Provider error after %d attempts: %s",
                        total_attempts,
                        redact_secrets(str(classified), self._secrets),
                    )

        raise last_error from None

    def _call_llm(self, p: str, timeout: Optional[float] = None) -> str:
        from langchain_core.messages import HumanMessage

        eff_timeout = timeout if (timeout is not None and timeout < self._timeout) else self._timeout
        eff_timeout = max(self.MIN_REQUEST_TIMEOUT, eff_timeout)
        result = self.llm.invoke([HumanMessage(content=p)], timeout=eff_timeout)
        return _extract_text_from_content(result.content)

    def invoke(self, prompt: str, *, deadline: Optional[float] = None) -> str:
        return self._invoke_with_retries(self._call_llm, prompt, deadline=deadline)

    def invoke_structured(
        self, prompt: str, response_model: type, *, deadline: Optional[float] = None
    ) -> Any:
        import json as json_mod

        def _call_and_parse(p: str, timeout: Optional[float] = None) -> Any:
            try:
                text = self._call_llm(p, timeout=timeout)
            except TypeError:
                text = self._call_llm(p)
            json_str = _extract_json(text)
            try:
                data = json_mod.loads(json_str)
                return response_model(**data)
            except Exception as parse_err:
                safe_msg = redact_secrets(str(parse_err), self._secrets)
                raise MalformedOutputError(
                    f"Failed to parse provider output as {response_model.__name__}: {safe_msg}"
                ) from parse_err

        return self._invoke_with_retries(_call_and_parse, prompt, deadline=deadline)


# ── Safe Groq Provider ───────────────────────────────────────────────────────

class SafeGroqProvider(LLMProvider):
    """Production Groq provider with bounded retries and secret redaction.

    Wraps ``langchain-groq.ChatGroq`` behind a safe boundary that maps
    all provider exceptions to domain types.

    Parameters
    ----------
    api_key : str
        Groq API key. Read from configuration, never hard-coded.
    model_name : str
        Groq model identifier (default ``openai/gpt-oss-120b``).
    max_retries : int
        Maximum retries for transient failures (default 2, hard cap 3).
    temperature : float
        Sampling temperature for generation.
    timeout : float
        Per-request HTTP timeout in seconds (default 30.0).
    additional_secrets : Sequence[str]
        Additional sensitive strings (e.g. Gemini key, LangSmith key) to redact.
    """

    MIN_REQUEST_TIMEOUT: float = 10.0

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-oss-120b",
        max_retries: int = 2,
        temperature: float = 0.1,
        timeout: float = 30.0,
        additional_secrets: Sequence[str] = (),
    ):
        if not api_key:
            raise AuthenticationError("Groq API key is required but was empty or not provided.")

        self._api_key = api_key
        self._model_name = model_name
        self._max_retries = min(max_retries, 3)
        self._temperature = temperature
        self._timeout = max(self.MIN_REQUEST_TIMEOUT, timeout)

        self._secrets: list[str] = [api_key]
        for s in additional_secrets:
            if s and s not in self._secrets:
                self._secrets.append(s)

        self._llm = None

    @classmethod
    def from_settings(cls, settings: Any, **kwargs: Any) -> SafeGroqProvider:
        """Create a SafeGroqProvider configured from application Settings."""
        additional = []
        for attr in ("api_key", "langsmith_api_key"):
            val = getattr(settings, attr, None)
            if val:
                additional.append(val)
        req_timeout = getattr(settings, "request_timeout", 30.0)
        groq_model = getattr(settings, "groq_model_name", "openai/gpt-oss-120b")
        return cls(
            api_key=getattr(settings, "groq_api_key", ""),
            model_name=groq_model,
            timeout=req_timeout,
            additional_secrets=additional,
            **kwargs,
        )

    @property
    def llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                model=self._model_name,
                groq_api_key=self._api_key,
                temperature=self._temperature,
                timeout=self._timeout,
                max_retries=0,
            )
        return self._llm

    def _invoke_with_retries(self, call_fn, prompt: str, *, deadline: Optional[float] = None) -> Any:
        """Execute a provider call with bounded outer retries and cooperative deadline enforcement."""
        total_attempts = 1 + self._max_retries
        last_error: ProviderError = ProviderError("No provider attempts configured", retryable=False)

        for attempt in range(1, total_attempts + 1):
            now = time.time()
            if deadline is not None:
                time_left = deadline - now
                if time_left <= 0:
                    raise TimeoutError("Overall workflow execution deadline exceeded before provider call.")
                if time_left < self.MIN_REQUEST_TIMEOUT:
                    raise TimeoutError(
                        f"Remaining workflow budget ({time_left:.2f}s) is below provider minimum "
                        f"deadline ({self.MIN_REQUEST_TIMEOUT:.1f}s)."
                    )
                effective_timeout = min(self._timeout, time_left)
            else:
                effective_timeout = self._timeout

            try:
                try:
                    res = call_fn(prompt, timeout=effective_timeout)
                except TypeError:
                    res = call_fn(prompt)
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError("Overall workflow execution deadline exceeded after provider call.")
                return res
            except Exception as raw_err:
                if isinstance(raw_err, ProviderError):
                    classified = raw_err
                else:
                    classified = _classify_provider_error(raw_err, self._secrets)
                last_error = classified

                if not classified.retryable:
                    logger.error(
                        "Non-retryable provider error (attempt %d/%d): %s",
                        attempt,
                        total_attempts,
                        redact_secrets(str(classified), self._secrets),
                    )
                    raise classified from None

                if attempt < total_attempts:
                    wait = min(2 ** attempt, 8)
                    if deadline is not None:
                        time_left = deadline - time.time()
                        needed_budget = wait + self.MIN_REQUEST_TIMEOUT
                        if time_left < needed_budget:
                            logger.warning(
                                "Aborting provider retry: remaining deadline (%.2fs) cannot accommodate "
                                "backoff (%ds) and next attempt minimum deadline (%.1fs).",
                                time_left,
                                wait,
                                self.MIN_REQUEST_TIMEOUT,
                            )
                            raise TimeoutError(
                                f"Overall workflow execution deadline exceeded; aborting retry: {classified}"
                            ) from None

                    logger.warning(
                        "Retryable provider error (attempt %d/%d), waiting %ds: %s",
                        attempt,
                        total_attempts,
                        wait,
                        redact_secrets(str(classified), self._secrets),
                    )
                    time.sleep(wait)
                    if deadline is not None and time.time() >= deadline:
                        raise TimeoutError(
                            "Overall workflow execution deadline exceeded during retry backoff."
                        )
                else:
                    logger.error(
                        "Provider error after %d attempts: %s",
                        total_attempts,
                        redact_secrets(str(classified), self._secrets),
                    )

        raise last_error from None

    def _call_llm(self, p: str, timeout: Optional[float] = None) -> str:
        from langchain_core.messages import HumanMessage

        eff_timeout = timeout if (timeout is not None and timeout < self._timeout) else self._timeout
        result = self.llm.invoke([HumanMessage(content=p)], timeout=eff_timeout)
        return _extract_text_from_content(result.content)

    def invoke(self, prompt: str, *, deadline: Optional[float] = None) -> str:
        return self._invoke_with_retries(self._call_llm, prompt, deadline=deadline)

    def invoke_structured(
        self, prompt: str, response_model: type, *, deadline: Optional[float] = None
    ) -> Any:
        import json as json_mod

        def _call_and_parse(p: str, timeout: Optional[float] = None) -> Any:
            try:
                text = self._call_llm(p, timeout=timeout)
            except TypeError:
                text = self._call_llm(p)
            json_str = _extract_json(text)
            try:
                data = json_mod.loads(json_str)
                return response_model(**data)
            except Exception as parse_err:
                safe_msg = redact_secrets(str(parse_err), self._secrets)
                raise MalformedOutputError(
                    f"Failed to parse provider output as {response_model.__name__}: {safe_msg}"
                ) from parse_err

        return self._invoke_with_retries(_call_and_parse, prompt, deadline=deadline)


# ── Failover Multi-Provider ──────────────────────────────────────────────────

class ResilientFailoverProvider(LLMProvider):
    """Wraps a primary provider with automatic secondary failover on quota / rate limit errors.

    If the primary provider encounters RateLimitError (HTTP 429 quota exhaustion)
    or ProviderUnavailableError (HTTP 503 high demand), requests automatically fail over
    to the secondary provider to ensure continuous application availability.
    """

    def __init__(
        self,
        primary: LLMProvider,
        secondary: LLMProvider,
        *,
        primary_name: str = "Gemini",
        secondary_name: str = "Groq",
    ):
        self.primary = primary
        self.secondary = secondary
        self.primary_name = primary_name
        self.secondary_name = secondary_name

    def _is_failover_error(self, err: Exception) -> bool:
        if isinstance(err, (RateLimitError, ProviderUnavailableError)):
            return True
        msg = str(err).lower()
        return any(
            kw in msg
            for kw in (
                "rate limit",
                "ratelimit",
                "429",
                "quota",
                "resource_exhausted",
                "503",
                "unavailable",
                "temporarily unavailable",
            )
        )

    def invoke(self, prompt: str, *, deadline: Optional[float] = None) -> str:
        try:
            return self.primary.invoke(prompt, deadline=deadline)
        except Exception as err:
            if self._is_failover_error(err):
                logger.warning(
                    "Primary provider %s encountered %s; failing over to secondary provider %s: %s",
                    self.primary_name,
                    type(err).__name__,
                    self.secondary_name,
                    err,
                )
                return self.secondary.invoke(prompt, deadline=deadline)
            raise

    def invoke_structured(
        self, prompt: str, response_model: type, *, deadline: Optional[float] = None
    ) -> Any:
        try:
            return self.primary.invoke_structured(
                prompt, response_model=response_model, deadline=deadline
            )
        except Exception as err:
            if self._is_failover_error(err):
                logger.warning(
                    "Primary provider %s encountered %s; failing over to secondary provider %s: %s",
                    self.primary_name,
                    type(err).__name__,
                    self.secondary_name,
                    err,
                )
                return self.secondary.invoke_structured(
                    prompt, response_model=response_model, deadline=deadline
                )
            raise


# ── Provider Factory ─────────────────────────────────────────────────────────

def create_provider_from_settings(settings: Any, *, enable_failover: bool = True, **kwargs: Any) -> LLMProvider:
    """Instantiate the configured provider from application Settings.

    Defaults to SafeGeminiProvider unless LLM_PROVIDER or MODEL_PROVIDER is explicitly 'groq'.
    When Gemini is primary and a Groq API key is present, automatically wraps in a
    ResilientFailoverProvider so quota exhaustion (HTTP 429) fails over to Groq seamlessly.
    """
    provider_name = str(getattr(settings, "model_provider", "gemini")).lower()
    if provider_name == "groq":
        return SafeGroqProvider.from_settings(settings, **kwargs)

    primary = SafeGeminiProvider.from_settings(settings, **kwargs)
    groq_key = getattr(settings, "groq_api_key", None)
    if enable_failover and groq_key and len(str(groq_key).strip()) > 0:
        secondary = SafeGroqProvider.from_settings(settings, **kwargs)
        return ResilientFailoverProvider(
            primary=primary,
            secondary=secondary,
            primary_name="Gemini",
            secondary_name="Groq",
        )
    return primary
