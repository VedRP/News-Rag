class RetryableProviderError(Exception):
    """
    Raised by a provider wrapper when a call failed in a way that trying a *different*
    provider might fix: missing/expired/invalid API key, rate limit, or server-side
    overload/unavailability. backend.llm.client's router catches only this exception
    type to move on to the next configured provider.

    A provider wrapper should let any other exception (a genuine bad request, a bug in
    our prompt/params) propagate unchanged -- retrying it against a different provider
    would just fail the same way there too, and silently masking it makes debugging
    much harder.
    """
