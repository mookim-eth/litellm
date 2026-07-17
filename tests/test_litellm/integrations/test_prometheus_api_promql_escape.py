from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.prometheus_helpers import prometheus_api


def test_should_quote_untrusted_promql_label_values():
    quote = prometheus_api._quote_promql_string_literal
    assert quote("safe-hash") == '"safe-hash"'
    assert quote('value"} or sum(secret_metric)') == (
        '"value\\"} or sum(secret_metric)"'
    )
    assert quote("line\nbreak") == '"line\\nbreak"'


@pytest.mark.asyncio
async def test_should_not_allow_api_key_to_escape_promql_matcher():
    captured = {}

    class Response:
        def json(self):
            return {"data": {"result": []}}

    async def capture(url, params):
        captured["query"] = params["query"]
        return Response()

    client = MagicMock()
    client.get = AsyncMock(side_effect=capture)
    attacker_value = 'victim"} or sum(secret_metric{label="x'

    with (
        patch.object(prometheus_api, "PROMETHEUS_URL", "http://prometheus"),
        patch.object(prometheus_api, "async_http_handler", client),
    ):
        await prometheus_api.get_daily_spend_from_prometheus(attacker_value)

    query = captured["query"]
    assert query.startswith(
        'sum(delta(litellm_spend_metric_total{hashed_api_key="victim\\"}'
    )
    assert 'hashed_api_key="victim"}' not in query
