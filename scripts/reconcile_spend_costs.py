#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, getcontext
from typing import Any, Callable, DefaultDict, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row

getcontext().prec = 28

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_DOTENV_PATH = REPO_ROOT / ".env"
if REPO_DOTENV_PATH.exists():
    load_dotenv(REPO_DOTENV_PATH, override=False)


DECIMAL_ZERO = Decimal("0")
DEFAULT_TOLERANCE = Decimal("0.000001")
DEFAULT_SAMPLE_LIMIT = 5


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    cache_read_input_token_cost: Decimal


# Known pricing that we want to reconcile/fix.
# Extend this map when additional models need hardcoded repair rules.
PRICE_MAP: Dict[str, ModelPricing] = {
    "chatgpt/gpt-5.4": ModelPricing(
        input_cost_per_token=Decimal("0.0000025"),
        output_cost_per_token=Decimal("0.000015"),
        cache_read_input_token_cost=Decimal("0.00000025"),
    ),
    "chatgpt/gpt-5.4-mini": ModelPricing(
        input_cost_per_token=Decimal("0.00000075"),
        output_cost_per_token=Decimal("0.0000045"),
        cache_read_input_token_cost=Decimal("0.000000075"),
    ),
    "chatgpt/gpt-5.3-codex": ModelPricing(
        input_cost_per_token=Decimal("0.00000175"),
        output_cost_per_token=Decimal("0.000014"),
        cache_read_input_token_cost=Decimal("0.000000175"),
    ),
}

# Keep this aligned with litellm.proxy.route_llm_request.ROUTE_ENDPOINT_MAPPING.
ENDPOINT_BY_CALL_TYPE: Dict[str, str] = {
    "acompletion": "/chat/completions",
    "atext_completion": "/completions",
    "aembedding": "/embeddings",
    "aimage_generation": "/image/generations",
    "aspeech": "/audio/speech",
    "atranscription": "/audio/transcriptions",
    "amoderation": "/moderations",
    "arerank": "/rerank",
    "aresponses": "/responses",
    "_aresponses_websocket": "/responses",
    "alist_input_items": "/responses/{response_id}/input_items",
    "aimage_edit": "/images/edits",
    "acancel_responses": "/responses/{response_id}/cancel",
    "acompact_responses": "/responses/compact",
    "aocr": "/ocr",
    "asearch": "/search",
    "avideo_generation": "/videos",
    "avideo_list": "/videos",
    "avideo_status": "/videos/{video_id}",
    "avideo_content": "/videos/{video_id}/content",
    "avideo_remix": "/videos/{video_id}/remix",
    "acreate_realtime_client_secret": "/realtime/client_secrets",
    "arealtime_calls": "/realtime/calls",
    "acreate_container": "/containers",
    "alist_containers": "/containers",
    "aretrieve_container": "/containers/{container_id}",
    "adelete_container": "/containers/{container_id}",
    "aupload_container_file": "/containers/{container_id}/files",
    "alist_container_files": "/containers/{container_id}/files",
    "aretrieve_container_file": "/containers/{container_id}/files/{file_id}",
    "adelete_container_file": "/containers/{container_id}/files/{file_id}",
    "aretrieve_container_file_content": "/containers/{container_id}/files/{file_id}/content",
    "acreate_skill": "/skills",
    "alist_skills": "/skills",
    "aretrieve_skill": "/skills/{skill_id}",
    "aupdate_skill": "/skills/{skill_id}",
    "adelete_skill": "/skills/{skill_id}",
    "arun_skill": "/skills/{skill_id}/run",
    "alist_skill_runs": "/skills/runs",
    "aretrieve_skill_run": "/skills/runs/{run_id}",
    "acancel_skill_run": "/skills/runs/{run_id}/cancel",
    "acreate_batch": "/batches",
    "aretrieve_batch": "/batches/{batch_id}",
    "acancel_batch": "/batches/{batch_id}/cancel",
    "alist_batches": "/batches",
    "acreate_fine_tuning_job": "/fine_tuning/jobs",
    "alist_fine_tuning_jobs": "/fine_tuning/jobs",
    "aretrieve_fine_tuning_job": "/fine_tuning/jobs/{fine_tuning_job_id}",
    "acancel_fine_tuning_job": "/fine_tuning/jobs/{fine_tuning_job_id}/cancel",
    "alist_fine_tuning_job_events": "/fine_tuning/jobs/{fine_tuning_job_id}/events",
    "adelete_fine_tuned_model": "/models/{model}",
}


@dataclass
class CurrentRow:
    record_id: Any
    spend: Decimal


@dataclass
class DiffEntry:
    key: Hashable
    record_id: Any
    current_spend: Decimal
    expected_spend: Decimal

    @property
    def delta(self) -> Decimal:
        return self.expected_spend - self.current_spend


@dataclass
class TableComparison:
    table_name: str
    expected_rows: int
    db_rows: int
    changed_rows: int
    missing_rows: int
    extra_rows: int
    expected_total: Decimal
    db_total: Decimal
    changed_entries: List[DiffEntry] = field(default_factory=list)
    missing_keys: List[Hashable] = field(default_factory=list)
    extra_keys: List[Hashable] = field(default_factory=list)

    @property
    def total_delta(self) -> Decimal:
        return self.expected_total - self.db_total


@dataclass
class Aggregates:
    spend_logs: Dict[str, Decimal] = field(default_factory=dict)
    verification_tokens: DefaultDict[str, Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    users: DefaultDict[str, Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    teams: DefaultDict[str, Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    organizations: DefaultDict[str, Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    end_users: DefaultDict[str, Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    agents: DefaultDict[str, Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    tags: DefaultDict[str, Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    team_memberships: DefaultDict[Tuple[str, str], Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    daily_users: DefaultDict[Tuple[str, str, str, str, str, str, str], Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    daily_teams: DefaultDict[Tuple[str, str, str, str, str, str, str], Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    daily_organizations: DefaultDict[
        Tuple[str, str, str, str, str, str, str], Decimal
    ] = field(default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO))
    daily_end_users: DefaultDict[
        Tuple[str, str, str, str, str, str, str], Decimal
    ] = field(default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO))
    daily_agents: DefaultDict[Tuple[str, str, str, str, str, str, str], Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )
    daily_tags: DefaultDict[Tuple[str, str, str, str, str, str, str], Decimal] = field(
        default_factory=lambda: defaultdict(lambda: DECIMAL_ZERO)
    )


@dataclass(frozen=True)
class TableSpec:
    table_name: str
    id_columns: Tuple[str, ...]
    key_columns: Tuple[str, ...]
    update_columns: Tuple[str, ...]
    touch_updated_at: bool = False


TABLE_SPECS: Dict[str, TableSpec] = {
    "LiteLLM_SpendLogs": TableSpec(
        table_name="LiteLLM_SpendLogs",
        id_columns=("request_id",),
        key_columns=("request_id",),
        update_columns=("request_id",),
    ),
    "LiteLLM_VerificationToken": TableSpec(
        table_name="LiteLLM_VerificationToken",
        id_columns=("token",),
        key_columns=("token",),
        update_columns=("token",),
        touch_updated_at=True,
    ),
    "LiteLLM_UserTable": TableSpec(
        table_name="LiteLLM_UserTable",
        id_columns=("user_id",),
        key_columns=("user_id",),
        update_columns=("user_id",),
        touch_updated_at=True,
    ),
    "LiteLLM_TeamTable": TableSpec(
        table_name="LiteLLM_TeamTable",
        id_columns=("team_id",),
        key_columns=("team_id",),
        update_columns=("team_id",),
        touch_updated_at=True,
    ),
    "LiteLLM_OrganizationTable": TableSpec(
        table_name="LiteLLM_OrganizationTable",
        id_columns=("organization_id",),
        key_columns=("organization_id",),
        update_columns=("organization_id",),
        touch_updated_at=True,
    ),
    "LiteLLM_EndUserTable": TableSpec(
        table_name="LiteLLM_EndUserTable",
        id_columns=("user_id",),
        key_columns=("user_id",),
        update_columns=("user_id",),
    ),
    "LiteLLM_AgentsTable": TableSpec(
        table_name="LiteLLM_AgentsTable",
        id_columns=("agent_id",),
        key_columns=("agent_id",),
        update_columns=("agent_id",),
        touch_updated_at=True,
    ),
    "LiteLLM_TagTable": TableSpec(
        table_name="LiteLLM_TagTable",
        id_columns=("tag_name",),
        key_columns=("tag_name",),
        update_columns=("tag_name",),
        touch_updated_at=True,
    ),
    "LiteLLM_TeamMembership": TableSpec(
        table_name="LiteLLM_TeamMembership",
        id_columns=("team_id", "user_id"),
        key_columns=("team_id", "user_id"),
        update_columns=("team_id", "user_id"),
    ),
    "LiteLLM_DailyUserSpend": TableSpec(
        table_name="LiteLLM_DailyUserSpend",
        id_columns=("id",),
        key_columns=("user_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"),
        update_columns=("id",),
        touch_updated_at=True,
    ),
    "LiteLLM_DailyTeamSpend": TableSpec(
        table_name="LiteLLM_DailyTeamSpend",
        id_columns=("id",),
        key_columns=("team_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"),
        update_columns=("id",),
        touch_updated_at=True,
    ),
    "LiteLLM_DailyOrganizationSpend": TableSpec(
        table_name="LiteLLM_DailyOrganizationSpend",
        id_columns=("id",),
        key_columns=("organization_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"),
        update_columns=("id",),
        touch_updated_at=True,
    ),
    "LiteLLM_DailyEndUserSpend": TableSpec(
        table_name="LiteLLM_DailyEndUserSpend",
        id_columns=("id",),
        key_columns=("end_user_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"),
        update_columns=("id",),
        touch_updated_at=True,
    ),
    "LiteLLM_DailyAgentSpend": TableSpec(
        table_name="LiteLLM_DailyAgentSpend",
        id_columns=("id",),
        key_columns=("agent_id", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"),
        update_columns=("id",),
        touch_updated_at=True,
    ),
    "LiteLLM_DailyTagSpend": TableSpec(
        table_name="LiteLLM_DailyTagSpend",
        id_columns=("id",),
        key_columns=("tag", "date", "api_key", "model", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint"),
        update_columns=("id",),
        touch_updated_at=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute correct spend from LiteLLM spend logs using hardcoded model pricing, "
            "compare the corrected spend/aggregates against database values, and optionally fix them.\n\n"
            "Important: aggregate reconciliation treats the currently stored LiteLLM_SpendLogs rows "
            "as the source of truth."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="Postgres DATABASE_URL. Defaults to the DATABASE_URL environment variable.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=sorted(PRICE_MAP.keys()),
        choices=sorted(PRICE_MAP.keys()),
        help=(
            "Subset of hardcoded-price models to recalculate. Default: all known repair models."
        ),
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Actually write corrected spend values back to the database.",
    )
    parser.add_argument(
        "--tolerance",
        default=str(DEFAULT_TOLERANCE),
        help=f"Numeric tolerance for considering two spend values equal. Default: {DEFAULT_TOLERANCE}",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help=f"How many sample diffs to print per table. Default: {DEFAULT_SAMPLE_LIMIT}",
    )
    return parser.parse_args()


def decimalify(value: Any) -> Decimal:
    if value is None:
        return DECIMAL_ZERO
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return DECIMAL_ZERO


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, memoryview):
        value = value.tobytes().decode("utf-8")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def extract_cached_tokens(metadata: Any) -> int:
    parsed = parse_json_field(metadata)
    if not isinstance(parsed, dict):
        return 0
    usage = parsed.get("usage_object")
    if not isinstance(usage, dict):
        return 0

    cache_read_tokens = usage.get("cache_read_input_tokens")
    if cache_read_tokens is not None:
        try:
            return max(int(cache_read_tokens), 0)
        except (TypeError, ValueError):
            pass

    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens")
        try:
            return max(int(cached_tokens), 0)
        except (TypeError, ValueError):
            return 0

    return 0


def extract_request_tags(request_tags: Any) -> List[str]:
    parsed = parse_json_field(request_tags)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if isinstance(item, str) and item]
    return []


def endpoint_for_call_type(call_type: Any) -> str:
    return ENDPOINT_BY_CALL_TYPE.get(normalize_text(call_type), "")


def iso_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = normalize_text(value)
    return text.split("T", 1)[0] if text else ""


def compute_corrected_spend(row: Dict[str, Any], target_models: set[str]) -> Decimal:
    current_spend = decimalify(row.get("spend"))
    model = normalize_text(row.get("model"))
    pricing = PRICE_MAP.get(model)
    if pricing is None or model not in target_models:
        return current_spend

    prompt_tokens = max(int(row.get("prompt_tokens") or 0), 0)
    completion_tokens = max(int(row.get("completion_tokens") or 0), 0)
    cached_tokens = min(extract_cached_tokens(row.get("metadata")), prompt_tokens)
    uncached_prompt_tokens = max(prompt_tokens - cached_tokens, 0)

    return (
        decimalify(uncached_prompt_tokens) * pricing.input_cost_per_token
        + decimalify(cached_tokens) * pricing.cache_read_input_token_cost
        + decimalify(completion_tokens) * pricing.output_cost_per_token
    )


def load_spend_logs(conn: psycopg.Connection[Any]) -> List[Dict[str, Any]]:
    query = sql.SQL(
        """
        SELECT
            request_id,
            api_key,
            spend,
            prompt_tokens,
            completion_tokens,
            "startTime",
            model,
            custom_llm_provider,
            "user",
            team_id,
            organization_id,
            end_user,
            agent_id,
            mcp_namespaced_tool_name,
            call_type,
            request_tags,
            metadata
        FROM "LiteLLM_SpendLogs"
        ORDER BY "startTime", request_id
        """
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        return list(cur.fetchall())


def build_expected_aggregates(
    spend_logs: Iterable[Dict[str, Any]],
    target_models: set[str],
    tolerance: Decimal,
) -> Tuple[Aggregates, int, Decimal]:
    aggregates = Aggregates()
    changed_spend_log_rows = 0
    total_spend_log_delta = DECIMAL_ZERO

    for row in spend_logs:
        request_id = normalize_text(row.get("request_id"))
        api_key = normalize_text(row.get("api_key"))
        user_id = normalize_text(row.get("user"))
        team_id = normalize_text(row.get("team_id"))
        organization_id = normalize_text(row.get("organization_id"))
        end_user_id = normalize_text(row.get("end_user"))
        agent_id = normalize_text(row.get("agent_id"))
        model = normalize_text(row.get("model"))
        provider = normalize_text(row.get("custom_llm_provider"))
        mcp_namespaced_tool_name = normalize_text(row.get("mcp_namespaced_tool_name"))
        endpoint = endpoint_for_call_type(row.get("call_type"))
        log_date = iso_date(row.get("startTime"))
        tags = extract_request_tags(row.get("request_tags"))

        current_spend = decimalify(row.get("spend"))
        corrected_spend = compute_corrected_spend(row=row, target_models=target_models)

        if abs(corrected_spend - current_spend) > tolerance:
            changed_spend_log_rows += 1
            total_spend_log_delta += corrected_spend - current_spend

        aggregates.spend_logs[request_id] = corrected_spend

        if api_key:
            aggregates.verification_tokens[api_key] += corrected_spend
        if user_id:
            aggregates.users[user_id] += corrected_spend
        if team_id:
            aggregates.teams[team_id] += corrected_spend
        if organization_id:
            aggregates.organizations[organization_id] += corrected_spend
        if end_user_id:
            aggregates.end_users[end_user_id] += corrected_spend
        if agent_id:
            aggregates.agents[agent_id] += corrected_spend
        if team_id and user_id:
            aggregates.team_memberships[(team_id, user_id)] += corrected_spend
        for tag in tags:
            aggregates.tags[tag] += corrected_spend

        daily_common = (log_date, api_key, model, provider, mcp_namespaced_tool_name, endpoint)
        aggregates.daily_users[(user_id, *daily_common)] += corrected_spend
        if row.get("team_id") is not None:
            aggregates.daily_teams[(team_id, *daily_common)] += corrected_spend
        if row.get("organization_id") is not None:
            aggregates.daily_organizations[(organization_id, *daily_common)] += corrected_spend
        if end_user_id:
            aggregates.daily_end_users[(end_user_id, *daily_common)] += corrected_spend
        if row.get("agent_id") is not None:
            aggregates.daily_agents[(agent_id, *daily_common)] += corrected_spend
        for tag in tags:
            aggregates.daily_tags[(tag, *daily_common)] += corrected_spend

    return aggregates, changed_spend_log_rows, total_spend_log_delta


def fetch_current_rows(
    conn: psycopg.Connection[Any],
    table_name: str,
    key_builder: Callable[[Dict[str, Any]], Hashable],
) -> Dict[Hashable, CurrentRow]:
    spec = TABLE_SPECS[table_name]
    selected_columns: List[str] = []
    for column in [*spec.id_columns, *spec.key_columns]:
        if column not in selected_columns:
            selected_columns.append(column)
    selected_identifiers = [sql.Identifier(column) for column in selected_columns]
    query = sql.SQL("SELECT {columns}, spend FROM {table}").format(
        columns=sql.SQL(", ").join([*selected_identifiers]),
        table=sql.Identifier(table_name),
    )

    rows: Dict[Hashable, CurrentRow] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur.fetchall():
            key = key_builder(row)
            if len(spec.id_columns) == 1:
                record_id: Any = row[spec.id_columns[0]]
            else:
                record_id = tuple(row[column] for column in spec.id_columns)
            rows[key] = CurrentRow(record_id=record_id, spend=decimalify(row.get("spend")))
    return rows


def compare_expected_vs_current(
    table_name: str,
    expected_rows: Dict[Hashable, Decimal],
    current_rows: Dict[Hashable, CurrentRow],
    tolerance: Decimal,
    sample_limit: int,
) -> TableComparison:
    changed_entries: List[DiffEntry] = []
    missing_keys: List[Hashable] = []
    extra_keys: List[Hashable] = []

    expected_total = sum(expected_rows.values(), DECIMAL_ZERO)
    current_total = sum((row.spend for row in current_rows.values()), DECIMAL_ZERO)

    all_keys = set(expected_rows.keys()) | set(current_rows.keys())
    for key in sorted(all_keys, key=lambda item: repr(item)):
        expected = expected_rows.get(key)
        current = current_rows.get(key)
        if expected is None and current is not None:
            if len(extra_keys) < sample_limit:
                extra_keys.append(key)
            continue
        if expected is not None and current is None:
            if len(missing_keys) < sample_limit:
                missing_keys.append(key)
            continue
        assert expected is not None and current is not None
        if abs(expected - current.spend) > tolerance and len(changed_entries) < sample_limit:
            changed_entries.append(
                DiffEntry(
                    key=key,
                    record_id=current.record_id,
                    current_spend=current.spend,
                    expected_spend=expected,
                )
            )

    changed_rows = sum(
        1
        for key, current in current_rows.items()
        if key in expected_rows and abs(expected_rows[key] - current.spend) > tolerance
    )
    missing_rows = sum(1 for key in expected_rows if key not in current_rows)
    extra_rows = sum(1 for key in current_rows if key not in expected_rows)

    return TableComparison(
        table_name=table_name,
        expected_rows=len(expected_rows),
        db_rows=len(current_rows),
        changed_rows=changed_rows,
        missing_rows=missing_rows,
        extra_rows=extra_rows,
        expected_total=expected_total,
        db_total=current_total,
        changed_entries=changed_entries,
        missing_keys=missing_keys,
        extra_keys=extra_keys,
    )


def update_rows(
    conn: psycopg.Connection[Any],
    table_name: str,
    current_rows: Dict[Hashable, CurrentRow],
    expected_rows: Dict[Hashable, Decimal],
    tolerance: Decimal,
) -> int:
    spec = TABLE_SPECS[table_name]
    changed_keys = [
        key
        for key, current in current_rows.items()
        if key in expected_rows and abs(expected_rows[key] - current.spend) > tolerance
    ]
    if not changed_keys:
        return 0

    if spec.touch_updated_at:
        update_stmt = sql.SQL("UPDATE {table} SET spend = %s, updated_at = NOW() WHERE {where}").format(
            table=sql.Identifier(table_name),
            where=sql.SQL(" AND ").join(
                sql.SQL("{} = %s").format(sql.Identifier(column))
                for column in spec.update_columns
            ),
        )
    else:
        update_stmt = sql.SQL("UPDATE {table} SET spend = %s WHERE {where}").format(
            table=sql.Identifier(table_name),
            where=sql.SQL(" AND ").join(
                sql.SQL("{} = %s").format(sql.Identifier(column))
                for column in spec.update_columns
            ),
        )

    params: List[Tuple[Any, ...]] = []
    for key in changed_keys:
        row = current_rows[key]
        new_spend = expected_rows[key]
        record_id = row.record_id
        if isinstance(record_id, tuple):
            params.append((new_spend, *record_id))
        else:
            params.append((new_spend, record_id))

    with conn.cursor() as cur:
        cur.executemany(update_stmt, params)
    return len(params)


def print_summary(
    changed_spend_log_rows: int,
    total_spend_log_delta: Decimal,
    comparisons: Sequence[TableComparison],
) -> None:
    print("=== spend log recalculation ===")
    print(f"affected spend log rows: {changed_spend_log_rows}")
    print(f"total spend log delta : {total_spend_log_delta:.12f}")
    print()

    print("=== table summary ===")
    for comparison in comparisons:
        print(
            f"[{comparison.table_name}] expected_rows={comparison.expected_rows} db_rows={comparison.db_rows} "
            f"changed={comparison.changed_rows} missing={comparison.missing_rows} extra={comparison.extra_rows} "
            f"db_total={comparison.db_total:.12f} expected_total={comparison.expected_total:.12f} "
            f"delta={comparison.total_delta:.12f}"
        )
        for entry in comparison.changed_entries:
            print(
                "  diff key={} record_id={} current={} expected={} delta={}".format(
                    repr(entry.key),
                    repr(entry.record_id),
                    f"{entry.current_spend:.12f}",
                    f"{entry.expected_spend:.12f}",
                    f"{entry.delta:.12f}",
                )
            )
        for key in comparison.missing_keys:
            print(f"  missing key={repr(key)}")
        for key in comparison.extra_keys:
            print(f"  extra key={repr(key)}")
        print()


def ensure_database_url(database_url: Optional[str]) -> str:
    if database_url:
        return database_url
    raise SystemExit(
        "DATABASE_URL is required. Pass --database-url or export DATABASE_URL first."
    )


def main() -> int:
    args = parse_args()
    database_url = ensure_database_url(args.database_url)
    tolerance = decimalify(args.tolerance)
    target_models = set(args.models)

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        spend_logs = load_spend_logs(conn)
        aggregates, changed_spend_log_rows, total_spend_log_delta = build_expected_aggregates(
            spend_logs=spend_logs,
            target_models=target_models,
            tolerance=tolerance,
        )

        current_spend_logs = {
            normalize_text(row["request_id"]): CurrentRow(
                record_id=normalize_text(row["request_id"]),
                spend=decimalify(row.get("spend")),
            )
            for row in spend_logs
        }

        comparisons: List[TableComparison] = []

        table_expectations: List[Tuple[str, Dict[Hashable, Decimal], Callable[[Dict[str, Any]], Hashable]]] = [
            (
                "LiteLLM_SpendLogs",
                aggregates.spend_logs,
                lambda row: normalize_text(row.get("request_id")),
            ),
            (
                "LiteLLM_VerificationToken",
                dict(aggregates.verification_tokens),
                lambda row: normalize_text(row.get("token")),
            ),
            (
                "LiteLLM_UserTable",
                dict(aggregates.users),
                lambda row: normalize_text(row.get("user_id")),
            ),
            (
                "LiteLLM_TeamTable",
                dict(aggregates.teams),
                lambda row: normalize_text(row.get("team_id")),
            ),
            (
                "LiteLLM_OrganizationTable",
                dict(aggregates.organizations),
                lambda row: normalize_text(row.get("organization_id")),
            ),
            (
                "LiteLLM_EndUserTable",
                dict(aggregates.end_users),
                lambda row: normalize_text(row.get("user_id")),
            ),
            (
                "LiteLLM_AgentsTable",
                dict(aggregates.agents),
                lambda row: normalize_text(row.get("agent_id")),
            ),
            (
                "LiteLLM_TagTable",
                dict(aggregates.tags),
                lambda row: normalize_text(row.get("tag_name")),
            ),
            (
                "LiteLLM_TeamMembership",
                dict(aggregates.team_memberships),
                lambda row: (
                    normalize_text(row.get("team_id")),
                    normalize_text(row.get("user_id")),
                ),
            ),
            (
                "LiteLLM_DailyUserSpend",
                dict(aggregates.daily_users),
                lambda row: (
                    normalize_text(row.get("user_id")),
                    normalize_text(row.get("date")),
                    normalize_text(row.get("api_key")),
                    normalize_text(row.get("model")),
                    normalize_text(row.get("custom_llm_provider")),
                    normalize_text(row.get("mcp_namespaced_tool_name")),
                    normalize_text(row.get("endpoint")),
                ),
            ),
            (
                "LiteLLM_DailyTeamSpend",
                dict(aggregates.daily_teams),
                lambda row: (
                    normalize_text(row.get("team_id")),
                    normalize_text(row.get("date")),
                    normalize_text(row.get("api_key")),
                    normalize_text(row.get("model")),
                    normalize_text(row.get("custom_llm_provider")),
                    normalize_text(row.get("mcp_namespaced_tool_name")),
                    normalize_text(row.get("endpoint")),
                ),
            ),
            (
                "LiteLLM_DailyOrganizationSpend",
                dict(aggregates.daily_organizations),
                lambda row: (
                    normalize_text(row.get("organization_id")),
                    normalize_text(row.get("date")),
                    normalize_text(row.get("api_key")),
                    normalize_text(row.get("model")),
                    normalize_text(row.get("custom_llm_provider")),
                    normalize_text(row.get("mcp_namespaced_tool_name")),
                    normalize_text(row.get("endpoint")),
                ),
            ),
            (
                "LiteLLM_DailyEndUserSpend",
                dict(aggregates.daily_end_users),
                lambda row: (
                    normalize_text(row.get("end_user_id")),
                    normalize_text(row.get("date")),
                    normalize_text(row.get("api_key")),
                    normalize_text(row.get("model")),
                    normalize_text(row.get("custom_llm_provider")),
                    normalize_text(row.get("mcp_namespaced_tool_name")),
                    normalize_text(row.get("endpoint")),
                ),
            ),
            (
                "LiteLLM_DailyAgentSpend",
                dict(aggregates.daily_agents),
                lambda row: (
                    normalize_text(row.get("agent_id")),
                    normalize_text(row.get("date")),
                    normalize_text(row.get("api_key")),
                    normalize_text(row.get("model")),
                    normalize_text(row.get("custom_llm_provider")),
                    normalize_text(row.get("mcp_namespaced_tool_name")),
                    normalize_text(row.get("endpoint")),
                ),
            ),
            (
                "LiteLLM_DailyTagSpend",
                dict(aggregates.daily_tags),
                lambda row: (
                    normalize_text(row.get("tag")),
                    normalize_text(row.get("date")),
                    normalize_text(row.get("api_key")),
                    normalize_text(row.get("model")),
                    normalize_text(row.get("custom_llm_provider")),
                    normalize_text(row.get("mcp_namespaced_tool_name")),
                    normalize_text(row.get("endpoint")),
                ),
            ),
        ]

        current_row_maps: Dict[str, Dict[Hashable, CurrentRow]] = {
            "LiteLLM_SpendLogs": current_spend_logs
        }

        for table_name, expected_rows, key_builder in table_expectations:
            if table_name not in current_row_maps:
                current_row_maps[table_name] = fetch_current_rows(
                    conn=conn,
                    table_name=table_name,
                    key_builder=key_builder,
                )
            comparisons.append(
                compare_expected_vs_current(
                    table_name=table_name,
                    expected_rows=expected_rows,
                    current_rows=current_row_maps[table_name],
                    tolerance=tolerance,
                    sample_limit=args.max_samples,
                )
            )

        print_summary(
            changed_spend_log_rows=changed_spend_log_rows,
            total_spend_log_delta=total_spend_log_delta,
            comparisons=comparisons,
        )

        if not args.fix:
            print("dry run only; no database changes were written.")
            missing_total = sum(item.missing_rows for item in comparisons)
            extra_total = sum(item.extra_rows for item in comparisons)
            if missing_total or extra_total:
                print(
                    "note: this script only updates existing rows. Missing/extra rows are reported but not inserted/deleted."
                )
            return 0

        updated_rows_by_table: Dict[str, int] = {}
        try:
            for table_name, expected_rows, _ in table_expectations:
                updated_rows_by_table[table_name] = update_rows(
                    conn=conn,
                    table_name=table_name,
                    current_rows=current_row_maps[table_name],
                    expected_rows=expected_rows,
                    tolerance=tolerance,
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        print("database changes committed.")
        for table_name, updated_rows in updated_rows_by_table.items():
            if updated_rows:
                print(f"  updated {table_name}: {updated_rows} rows")

        missing_total = sum(item.missing_rows for item in comparisons)
        extra_total = sum(item.extra_rows for item in comparisons)
        if missing_total or extra_total:
            print(
                "warning: missing/extra rows were only reported. This script does not insert or delete rows automatically."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
