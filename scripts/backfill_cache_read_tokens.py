#!/usr/bin/env python3
"""
One-time backfill for daily cache-read token metrics.

Historical ChatGPT/OpenAI-compatible spend logs may only contain cache reads in:

    metadata.usage_object.prompt_tokens_details.cached_tokens

The new usage UI reads daily table fields named `cache_read_input_tokens`, so this
script scans LiteLLM_SpendLogs, derives cache-read tokens from each usage object,
aggregates by the same dimensions as the daily spend writer, and updates the
daily spend tables using Prisma model methods.

Default mode is dry-run. Pass `--execute` to write changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from litellm.litellm_core_utils.usage_token_utils import (  # noqa: E402
    get_cache_read_input_tokens_from_usage,
)
from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING  # noqa: E402


DailyKey = Tuple[Optional[str], str, str, Optional[str], str, str, str]


def _log(message: str) -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


@dataclass(frozen=True)
class DailyTableSpec:
    display_name: str
    delegate_name: str
    entity_field: str
    unique_constraint_name: str


@dataclass
class DailyAggregate:
    entity_id: Optional[str]
    date: str
    api_key: str
    model: Optional[str]
    model_group: Optional[str]
    custom_llm_provider: str
    mcp_namespaced_tool_name: str
    endpoint: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    spend: float = 0.0
    api_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    request_id: Optional[str] = None


TABLE_SPECS: Dict[str, DailyTableSpec] = {
    "user": DailyTableSpec(
        display_name="daily user spend",
        delegate_name="litellm_dailyuserspend",
        entity_field="user_id",
        unique_constraint_name=(
            "user_id_date_api_key_model_custom_llm_provider_"
            "mcp_namespaced_tool_name_endpoint"
        ),
    ),
    "team": DailyTableSpec(
        display_name="daily team spend",
        delegate_name="litellm_dailyteamspend",
        entity_field="team_id",
        unique_constraint_name=(
            "team_id_date_api_key_model_custom_llm_provider_"
            "mcp_namespaced_tool_name_endpoint"
        ),
    ),
    "organization": DailyTableSpec(
        display_name="daily organization spend",
        delegate_name="litellm_dailyorganizationspend",
        entity_field="organization_id",
        unique_constraint_name=(
            "organization_id_date_api_key_model_custom_llm_provider_"
            "mcp_namespaced_tool_name_endpoint"
        ),
    ),
    "end_user": DailyTableSpec(
        display_name="daily end-user spend",
        delegate_name="litellm_dailyenduserspend",
        entity_field="end_user_id",
        unique_constraint_name=(
            "end_user_id_date_api_key_model_custom_llm_provider_"
            "mcp_namespaced_tool_name_endpoint"
        ),
    ),
    "agent": DailyTableSpec(
        display_name="daily agent spend",
        delegate_name="litellm_dailyagentspend",
        entity_field="agent_id",
        unique_constraint_name=(
            "agent_id_date_api_key_model_custom_llm_provider_"
            "mcp_namespaced_tool_name_endpoint"
        ),
    ),
    "tag": DailyTableSpec(
        display_name="daily tag spend",
        delegate_name="litellm_dailytagspend",
        entity_field="tag",
        unique_constraint_name=(
            "tag_date_api_key_model_custom_llm_provider_"
            "mcp_namespaced_tool_name_endpoint"
        ),
    ),
}


def _get_value(record: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _as_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _date_from_start_time(start_time: Any) -> Optional[str]:
    if isinstance(start_time, datetime):
        return start_time.date().isoformat()
    if isinstance(start_time, date):
        return start_time.isoformat()
    if isinstance(start_time, str) and start_time:
        return start_time.split("T", 1)[0].split(" ", 1)[0]
    return None


def _status_from_metadata(metadata: Any) -> str:
    metadata_dict = _parse_jsonish(metadata, {})
    if isinstance(metadata_dict, dict) and metadata_dict.get("status") == "failure":
        return "failure"
    return "success"


def _request_tags_from_record(record: Any) -> List[str]:
    request_tags = _parse_jsonish(_get_value(record, "request_tags", []), [])
    if not isinstance(request_tags, list):
        return []
    return [tag for tag in request_tags if isinstance(tag, str) and tag]


def _usage_object_from_record(record: Any) -> dict:
    metadata = _parse_jsonish(_get_value(record, "metadata", {}), {})
    if not isinstance(metadata, dict):
        return {}
    usage_object = metadata.get("usage_object") or {}
    return usage_object if isinstance(usage_object, dict) else {}


def _model_matches_prefix(model: Optional[str], model_prefix: Optional[str]) -> bool:
    if not model_prefix:
        return True
    return bool(model and model.startswith(model_prefix))


def _base_aggregate_from_record(record: Any) -> Optional[DailyAggregate]:
    start_date = _date_from_start_time(_get_value(record, "startTime"))
    if start_date is None:
        return None

    usage_object = _usage_object_from_record(record)
    cache_read_input_tokens = get_cache_read_input_tokens_from_usage(usage_object)
    if cache_read_input_tokens <= 0:
        return None

    status = _status_from_metadata(_get_value(record, "metadata", {}))
    call_type = _get_value(record, "call_type", None)
    endpoint = ROUTE_ENDPOINT_MAPPING.get(call_type, None) if call_type else None

    cache_creation_input_tokens = _as_int(
        usage_object.get("cache_creation_input_tokens", 0)
    )

    return DailyAggregate(
        entity_id=None,
        date=start_date,
        api_key=str(_get_value(record, "api_key", "") or ""),
        model=_get_value(record, "model", None),
        model_group=_get_value(record, "model_group", None),
        custom_llm_provider=str(_get_value(record, "custom_llm_provider", "") or ""),
        mcp_namespaced_tool_name=str(
            _get_value(record, "mcp_namespaced_tool_name", "") or ""
        ),
        endpoint=endpoint or "",
        prompt_tokens=_as_int(_get_value(record, "prompt_tokens", 0)),
        completion_tokens=_as_int(_get_value(record, "completion_tokens", 0)),
        spend=_as_float(_get_value(record, "spend", 0.0)),
        api_requests=1,
        successful_requests=1 if status == "success" else 0,
        failed_requests=1 if status != "success" else 0,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        request_id=_get_value(record, "request_id", None),
    )


def _aggregate_key(aggregate: DailyAggregate) -> DailyKey:
    return (
        aggregate.entity_id,
        aggregate.date,
        aggregate.api_key,
        aggregate.model,
        aggregate.custom_llm_provider,
        aggregate.mcp_namespaced_tool_name,
        aggregate.endpoint,
    )


def _add_aggregate(
    aggregates: DefaultDict[DailyKey, DailyAggregate],
    aggregate: DailyAggregate,
) -> None:
    key = _aggregate_key(aggregate)
    current = aggregates.get(key)
    if current is None:
        aggregates[key] = aggregate
        return

    current.prompt_tokens += aggregate.prompt_tokens
    current.completion_tokens += aggregate.completion_tokens
    current.spend += aggregate.spend
    current.api_requests += aggregate.api_requests
    current.successful_requests += aggregate.successful_requests
    current.failed_requests += aggregate.failed_requests
    current.cache_read_input_tokens += aggregate.cache_read_input_tokens
    current.cache_creation_input_tokens += aggregate.cache_creation_input_tokens
    if aggregate.request_id is not None:
        current.request_id = aggregate.request_id


def add_record_to_aggregates(
    record: Any,
    aggregates_by_table: Dict[str, DefaultDict[DailyKey, DailyAggregate]],
    model_prefix: Optional[str] = None,
) -> bool:
    base_aggregate = _base_aggregate_from_record(record)
    if base_aggregate is None:
        return False
    if not _model_matches_prefix(base_aggregate.model, model_prefix):
        return False

    user_aggregate = DailyAggregate(**{**base_aggregate.__dict__})
    user_aggregate.entity_id = _get_value(record, "user", None)
    _add_aggregate(aggregates_by_table["user"], user_aggregate)

    team_id = _get_value(record, "team_id", None)
    if team_id is not None:
        team_aggregate = DailyAggregate(**{**base_aggregate.__dict__})
        team_aggregate.entity_id = team_id
        _add_aggregate(aggregates_by_table["team"], team_aggregate)

    organization_id = _get_value(record, "organization_id", None)
    if organization_id is not None:
        organization_aggregate = DailyAggregate(**{**base_aggregate.__dict__})
        organization_aggregate.entity_id = organization_id
        _add_aggregate(aggregates_by_table["organization"], organization_aggregate)

    end_user = _get_value(record, "end_user", None)
    if end_user:
        end_user_aggregate = DailyAggregate(**{**base_aggregate.__dict__})
        end_user_aggregate.entity_id = end_user
        _add_aggregate(aggregates_by_table["end_user"], end_user_aggregate)

    agent_id = _get_value(record, "agent_id", None)
    if agent_id is not None:
        agent_aggregate = DailyAggregate(**{**base_aggregate.__dict__})
        agent_aggregate.entity_id = agent_id
        _add_aggregate(aggregates_by_table["agent"], agent_aggregate)

    for tag in _request_tags_from_record(record):
        tag_aggregate = DailyAggregate(**{**base_aggregate.__dict__})
        tag_aggregate.entity_id = tag
        _add_aggregate(aggregates_by_table["tag"], tag_aggregate)

    return True


def _build_where_clause(
    start_date: Optional[date],
    end_date: Optional[date],
) -> Dict[str, Any]:
    where: Dict[str, Any] = {}
    if start_date is None and end_date is None:
        return where

    start_time_filter: Dict[str, str] = {}
    if start_date is not None:
        start_dt = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
        start_time_filter["gte"] = start_dt.isoformat()
    if end_date is not None:
        end_exclusive = datetime.combine(
            end_date + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        start_time_filter["lt"] = end_exclusive.isoformat()
    where["startTime"] = start_time_filter
    return where


async def collect_aggregates(
    db: Any,
    *,
    batch_size: int,
    model_prefix: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
    max_records: Optional[int],
) -> Tuple[Dict[str, DefaultDict[DailyKey, DailyAggregate]], int, int]:
    aggregates_by_table: Dict[str, DefaultDict[DailyKey, DailyAggregate]] = {
        table_name: defaultdict()
        for table_name in TABLE_SPECS
    }
    where = _build_where_clause(start_date=start_date, end_date=end_date)

    offset = 0
    scanned_records = 0
    matched_records = 0
    while True:
        take = batch_size
        if max_records is not None:
            remaining = max_records - scanned_records
            if remaining <= 0:
                break
            take = min(take, remaining)

        records = await db.litellm_spendlogs.find_many(
            where=where or None,
            order=[{"startTime": "asc"}, {"request_id": "asc"}],
            skip=offset,
            take=take,
        )
        if not records:
            break

        for record in records:
            scanned_records += 1
            if add_record_to_aggregates(
                record,
                aggregates_by_table=aggregates_by_table,
                model_prefix=model_prefix,
            ):
                matched_records += 1

        offset += len(records)
        _log(
            f"Scanned {scanned_records} spend logs; matched {matched_records} "
            "with cache-read tokens"
        )

        if len(records) < take:
            break

    return aggregates_by_table, scanned_records, matched_records


def _where_for_aggregate(
    spec: DailyTableSpec,
    aggregate: DailyAggregate,
) -> Dict[str, Any]:
    return {
        spec.unique_constraint_name: {
            spec.entity_field: aggregate.entity_id,
            "date": aggregate.date,
            "api_key": aggregate.api_key,
            "model": aggregate.model,
            "custom_llm_provider": aggregate.custom_llm_provider,
            "mcp_namespaced_tool_name": aggregate.mcp_namespaced_tool_name,
            "endpoint": aggregate.endpoint,
        }
    }


def _create_data_for_aggregate(
    spec: DailyTableSpec,
    aggregate: DailyAggregate,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        spec.entity_field: aggregate.entity_id,
        "date": aggregate.date,
        "api_key": aggregate.api_key,
        "model": aggregate.model,
        "model_group": aggregate.model_group,
        "custom_llm_provider": aggregate.custom_llm_provider,
        "mcp_namespaced_tool_name": aggregate.mcp_namespaced_tool_name,
        "endpoint": aggregate.endpoint,
        "prompt_tokens": aggregate.prompt_tokens,
        "completion_tokens": aggregate.completion_tokens,
        "spend": aggregate.spend,
        "api_requests": aggregate.api_requests,
        "successful_requests": aggregate.successful_requests,
        "failed_requests": aggregate.failed_requests,
        "cache_read_input_tokens": aggregate.cache_read_input_tokens,
        "cache_creation_input_tokens": aggregate.cache_creation_input_tokens,
    }
    if spec.entity_field == "tag":
        data["request_id"] = aggregate.request_id
    return data


async def apply_backfill(
    db: Any,
    aggregates_by_table: Dict[str, DefaultDict[DailyKey, DailyAggregate]],
    *,
    execute: bool,
) -> Dict[str, int]:
    updated_counts: Dict[str, int] = {}

    for table_name, aggregates in aggregates_by_table.items():
        spec = TABLE_SPECS[table_name]
        updated_counts[table_name] = len(aggregates)
        total_cache_read_tokens = sum(
            aggregate.cache_read_input_tokens for aggregate in aggregates.values()
        )
        _log(
            f"{spec.display_name}: {len(aggregates)} rows, "
            f"{total_cache_read_tokens} cache-read tokens"
        )
        if not execute or len(aggregates) == 0:
            continue

        delegate = getattr(db, spec.delegate_name)
        for aggregate in aggregates.values():
            await delegate.upsert(
                where=_where_for_aggregate(spec, aggregate),
                data={
                    "create": _create_data_for_aggregate(spec, aggregate),
                    "update": {
                        "cache_read_input_tokens": aggregate.cache_read_input_tokens
                    },
                },
            )

    return updated_counts


def _parse_date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    return date.fromisoformat(value)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill daily cache_read_input_tokens from spend logs."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="Postgres DATABASE_URL. Defaults to env DATABASE_URL.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write updates. Without this flag, the script only prints a dry-run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Spend log page size for Prisma find_many.",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=None,
        help="Inclusive UTC date filter, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help="Inclusive UTC date filter, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--model-prefix",
        default=None,
        help="Optional model prefix filter, e.g. 'chatgpt/'.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional safety limit for spend logs scanned.",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    if not args.database_url:
        raise ValueError("DATABASE_URL is required. Pass --database-url or set env.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if (
        args.start_date is not None
        and args.end_date is not None
        and args.start_date > args.end_date
    ):
        raise ValueError("--start-date must be <= --end-date")

    os.environ["DATABASE_URL"] = args.database_url

    from prisma import Prisma  # type: ignore

    db = Prisma()
    await db.connect()
    try:
        aggregates_by_table, scanned_records, matched_records = await collect_aggregates(
            db,
            batch_size=args.batch_size,
            model_prefix=args.model_prefix,
            start_date=args.start_date,
            end_date=args.end_date,
            max_records=args.max_records,
        )
        _log(
            f"Finished scan: scanned={scanned_records}, matched={matched_records}, "
            f"mode={'execute' if args.execute else 'dry-run'}"
        )
        await apply_backfill(
            db,
            aggregates_by_table=aggregates_by_table,
            execute=args.execute,
        )
    finally:
        await db.disconnect()

    if not args.execute:
        _log("Dry-run only. Re-run with --execute to write updates.")
    return 0


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
