from typing import FrozenSet


DISABLED_OBSERVABILITY_INTEGRATIONS: FrozenSet[str] = frozenset(
    {"langfuse", "langfuse_otel", "langsmith"}
)

DISABLED_OBSERVABILITY_INTEGRATIONS_MESSAGE = (
    "Langfuse and Langsmith integrations are disabled by local security policy."
)


def ensure_observability_integration_enabled(integration_name: str) -> None:
    if integration_name.lower() in DISABLED_OBSERVABILITY_INTEGRATIONS:
        raise ValueError(DISABLED_OBSERVABILITY_INTEGRATIONS_MESSAGE)
