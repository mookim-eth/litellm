from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace

from scripts.backfill_cache_read_tokens import (
    TABLE_SPECS,
    add_record_to_aggregates,
)


def test_backfill_aggregates_prompt_tokens_details_cached_tokens():
    aggregates_by_table = {table_name: defaultdict() for table_name in TABLE_SPECS}
    record = SimpleNamespace(
        request_id="req-123",
        call_type="acompletion",
        startTime=datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc),
        api_key="sk-hash",
        model="chatgpt/gpt-5.4",
        model_group="gpt-5.4",
        custom_llm_provider="chatgpt",
        mcp_namespaced_tool_name=None,
        prompt_tokens=100,
        completion_tokens=20,
        spend=0.01,
        metadata={
            "usage_object": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cache_read_input_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 64},
            }
        },
        user="user-1",
        team_id="team-1",
        organization_id="org-1",
        end_user="customer-1",
        agent_id="agent-1",
        request_tags=["tag-1"],
    )

    assert (
        add_record_to_aggregates(
            record, aggregates_by_table=aggregates_by_table, model_prefix="chatgpt/"
        )
        is True
    )

    for table_name, aggregates in aggregates_by_table.items():
        assert len(aggregates) == 1, table_name
        aggregate = next(iter(aggregates.values()))
        assert aggregate.cache_read_input_tokens == 64
        assert aggregate.date == "2026-05-10"
        assert aggregate.endpoint == "/chat/completions"

    assert next(iter(aggregates_by_table["user"].values())).entity_id == "user-1"
    assert next(iter(aggregates_by_table["team"].values())).entity_id == "team-1"
    assert (
        next(iter(aggregates_by_table["organization"].values())).entity_id == "org-1"
    )
    assert (
        next(iter(aggregates_by_table["end_user"].values())).entity_id == "customer-1"
    )
    assert next(iter(aggregates_by_table["agent"].values())).entity_id == "agent-1"
    assert next(iter(aggregates_by_table["tag"].values())).entity_id == "tag-1"


def test_backfill_skips_non_matching_model_prefix():
    aggregates_by_table = {table_name: defaultdict() for table_name in TABLE_SPECS}
    record = {
        "request_id": "req-456",
        "call_type": "acompletion",
        "startTime": "2026-05-10T12:00:00+00:00",
        "api_key": "sk-hash",
        "model": "openai/gpt-4o",
        "custom_llm_provider": "openai",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "spend": 0.01,
        "metadata": {
            "usage_object": {
                "prompt_tokens_details": {"cached_tokens": 64},
            }
        },
        "user": "user-1",
        "request_tags": [],
    }

    assert (
        add_record_to_aggregates(
            record, aggregates_by_table=aggregates_by_table, model_prefix="chatgpt/"
        )
        is False
    )
    assert all(len(aggregates) == 0 for aggregates in aggregates_by_table.values())
