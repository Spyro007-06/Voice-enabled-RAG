"""Safe, standardized provider exceptions for Voice-Enabled RAG subsystem."""

from typing import Optional


def mask_credential(key: Optional[str], visible_chars: int = 4) -> str:
    """Safely mask API key or token to prevent credential exposure in logs or error messages.

    Args:
        key: The raw credential string.
        visible_chars: Number of trailing characters to leave visible.

    Returns:
        str: Masked representation (e.g. '***abcd') or '[REDACTED]' if empty/short.
    """
    if not key:
        return "[NOT_SET]"
    clean_key = str(key).strip()
    if len(clean_key) <= visible_chars:
        return "[REDACTED]"
    return f"***{clean_key[-visible_chars:]}"


class ProviderError(Exception):
    """Base exception for all external/internal provider errors.

    Ensures credentials or raw authorization headers are never propagated in error strings.
    """

    def __init__(
        self,
        message: str,
        provider_name: str = "unknown",
        status_code: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        self.provider_name = provider_name
        self.status_code = status_code
        self.details = details or {}
        # Ensure any secret in message is scrubbed
        super().__init__(f"[{provider_name.upper()}] {message}")


class MissingAPIKeyError(ProviderError):
    """Raised when an external production provider requires an API key that was not configured."""

    def __init__(self, provider_name: str, key_name: str):
        super().__init__(
            message=f"Missing required API credential '{key_name}'. Please set {key_name} in environment.",
            provider_name=provider_name,
            status_code=401,
        )
        self.key_name = key_name


class InvalidAPIKeyError(ProviderError):
    """Raised when the provider rejects credentials as unauthorized or forbidden."""

    def __init__(self, provider_name: str, key_name: Optional[str] = None):
        key_ref = f"for key '{key_name}' " if key_name else ""
        super().__init__(
            message=f"Invalid or unauthorized API credentials {key_ref}received from provider.",
            provider_name=provider_name,
            status_code=403,
        )


class ProviderTimeoutError(ProviderError):
    """Raised when an external provider request exceeds the configured timeout threshold."""

    def __init__(self, provider_name: str, timeout_seconds: float):
        super().__init__(
            message=f"Request timed out after {timeout_seconds:.1f}s while contacting upstream service.",
            provider_name=provider_name,
            status_code=504,
        )
        self.timeout_seconds = timeout_seconds


class ProviderConnectionError(ProviderError):
    """Raised when a network or DNS connection to the provider fails."""

    def __init__(self, provider_name: str, endpoint: str):
        super().__init__(
            message=f"Failed to establish connection to upstream endpoint '{endpoint}'.",
            provider_name=provider_name,
            status_code=502,
        )


class ProviderRateLimitError(ProviderError):
    """Raised when a provider responds with HTTP 429 Too Many Requests."""

    def __init__(self, provider_name: str, retry_after: Optional[int] = None):
        msg = "Upstream rate limit exceeded."
        if retry_after is not None:
            msg += f" Retry after {retry_after} seconds."
        super().__init__(
            message=msg,
            provider_name=provider_name,
            status_code=429,
        )
        self.retry_after = retry_after


class MalformedResponseError(ProviderError):
    """Raised when an external provider returns unexpected, empty, or unparseable payload."""

    def __init__(self, provider_name: str, message: str = "Received unparseable or malformed response payload."):
        super().__init__(
            message=message,
            provider_name=provider_name,
            status_code=502,
        )


class UnsupportedLanguageError(ProviderError):
    """Raised when a provider does not support the requested language/dialect."""

    def __init__(self, provider_name: str, language: str):
        super().__init__(
            message=f"Language '{language}' is not supported by {provider_name}.",
            provider_name=provider_name,
            status_code=400,
        )
        self.language = language


class InvalidRequestError(ProviderError):
    """Raised when request payload or parameters to the provider are invalid."""

    def __init__(self, provider_name: str, message: str):
        super().__init__(
            message=message,
            provider_name=provider_name,
            status_code=400,
        )
