import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes


_ACQUIRE_LEASE_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local expires_at = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local lease_id = ARGV[4]
local ttl = tonumber(ARGV[5])

redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZCARD', key) >= limit then
    return 0
end
redis.call('ZADD', key, expires_at, lease_id)
redis.call('EXPIRE', key, ttl)
return 1
"""

_RENEW_LEASE_SCRIPT = """
local key = KEYS[1]
local lease_id = ARGV[1]
local expires_at = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

if redis.call('ZSCORE', key, lease_id) == false then
    return 0
end
redis.call('ZADD', key, 'XX', expires_at, lease_id)
redis.call('EXPIRE', key, ttl)
return 1
"""

_RELEASE_LEASE_SCRIPT = """
return redis.call('ZREM', KEYS[1], ARGV[1])
"""

_COUNT_ACTIVE_LEASES_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
return redis.call('ZCARD', key)
"""


@dataclass
class _ChatGPTAccountLease:
    limiter: "ChatGPTAccountConcurrencyLimiter"
    account_key: str
    lease_id: str
    renewal_task: Optional[asyncio.Task] = None
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        if self.renewal_task is not None:
            self.renewal_task.cancel()
        await self.limiter._release(self.account_key, self.lease_id)


class ChatGPTAccountConcurrencyLimiter(CustomLogger):
    """Limit concurrent ChatGPT requests per provider account across models."""

    def __init__(self, internal_usage_cache: Any):
        self.internal_usage_cache = internal_usage_cache
        self._local_lock = asyncio.Lock()
        self._local_leases: Dict[str, set[str]] = {}
        self._auth_metadata_cache: Dict[str, Tuple[int, int, str, str]] = {}
        self._account_plan_types: Dict[str, str] = {}
        self._warned_missing_plans: set[str] = set()
        self.lease_ttl_seconds = int(
            os.getenv("LITELLM_CHATGPT_ACCOUNT_LEASE_TTL_SECONDS", "300")
        )
        redis_cache = self.internal_usage_cache.dual_cache.redis_cache
        if redis_cache is None:
            self._acquire_script = None
            self._renew_script = None
            self._release_script = None
            self._count_active_script = None
        else:
            self._acquire_script = redis_cache.async_register_script(
                _ACQUIRE_LEASE_SCRIPT
            )
            self._renew_script = redis_cache.async_register_script(_RENEW_LEASE_SCRIPT)
            self._release_script = redis_cache.async_register_script(
                _RELEASE_LEASE_SCRIPT
            )
            self._count_active_script = redis_cache.async_register_script(
                _COUNT_ACTIVE_LEASES_SCRIPT
            )

    @staticmethod
    def _get_limits() -> Dict[str, int]:
        from litellm.proxy.proxy_server import general_settings

        configured_limits = general_settings.get(
            "chatgpt_plan_max_parallel_requests"
        )
        if not isinstance(configured_limits, dict):
            return {}
        limits: Dict[str, int] = {}
        for plan_type, value in configured_limits.items():
            if isinstance(plan_type, str) and isinstance(value, int) and value > 0:
                limits[plan_type.lower()] = value
        return limits

    def _read_auth_metadata(self, auth_file_path: str) -> Tuple[str, str]:
        stat_result = os.stat(auth_file_path)
        cached = self._auth_metadata_cache.get(auth_file_path)
        if cached is not None and cached[:2] == (
            stat_result.st_mtime_ns,
            stat_result.st_size,
        ):
            return cached[2], cached[3]

        with open(auth_file_path, encoding="utf-8") as auth_file:
            auth_data = json.load(auth_file)
        account_id = auth_data.get("account_id")
        plan_type = auth_data.get("plan_type")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("ChatGPT auth file is missing account_id")
        if not isinstance(plan_type, str) or not plan_type:
            raise ValueError("ChatGPT auth file is missing plan_type")
        normalized_plan_type = plan_type.lower()
        self._auth_metadata_cache[auth_file_path] = (
            stat_result.st_mtime_ns,
            stat_result.st_size,
            account_id,
            normalized_plan_type,
        )
        return account_id, normalized_plan_type

    @staticmethod
    def _account_key(account_id: str) -> str:
        account_hash = hashlib.sha256(account_id.encode()).hexdigest()
        return f"chatgpt-account-concurrency:{account_hash}"

    async def _acquire(
        self, account_key: str, limit: int
    ) -> Optional[_ChatGPTAccountLease]:
        lease_id = str(uuid.uuid4())
        if self._acquire_script is not None:
            now = time.time()
            acquired = await self._acquire_script(
                keys=[account_key],
                args=[
                    now,
                    now + self.lease_ttl_seconds,
                    limit,
                    lease_id,
                    self.lease_ttl_seconds * 2,
                ],
            )
            if int(acquired) != 1:
                return None
            lease = _ChatGPTAccountLease(self, account_key, lease_id)
            lease.renewal_task = asyncio.create_task(self._renew_lease(lease))
            return lease

        async with self._local_lock:
            leases = self._local_leases.setdefault(account_key, set())
            if len(leases) >= limit:
                return None
            leases.add(lease_id)
        return _ChatGPTAccountLease(self, account_key, lease_id)

    async def _renew_lease(self, lease: _ChatGPTAccountLease) -> None:
        try:
            while not lease.released:
                await asyncio.sleep(max(1, self.lease_ttl_seconds // 3))
                if lease.released or self._renew_script is None:
                    return
                now = time.time()
                renewed = await self._renew_script(
                    keys=[lease.account_key],
                    args=[
                        lease.lease_id,
                        now + self.lease_ttl_seconds,
                        self.lease_ttl_seconds * 2,
                    ],
                )
                if int(renewed) != 1:
                    verbose_proxy_logger.warning(
                        "ChatGPT account concurrency lease disappeared before release"
                    )
                    return
        except asyncio.CancelledError:
            return
        except Exception:
            verbose_proxy_logger.exception(
                "Failed to renew ChatGPT account concurrency lease"
            )

    async def _release(self, account_key: str, lease_id: str) -> None:
        if self._release_script is not None:
            try:
                await self._release_script(keys=[account_key], args=[lease_id])
            except Exception:
                verbose_proxy_logger.exception(
                    "Failed to release ChatGPT account concurrency lease"
                )
            return

        async with self._local_lock:
            leases = self._local_leases.get(account_key)
            if leases is None:
                return
            leases.discard(lease_id)
            if not leases:
                self._local_leases.pop(account_key, None)

    async def get_concurrency_snapshot(self) -> Dict[str, Any]:
        """Return active provider-account leases without exposing account IDs."""
        limits = self._get_limits()
        account_plan_types = dict(self._account_plan_types)
        active_counts: Dict[str, int] = {}

        if self._count_active_script is not None:
            now = time.time()
            for account_key in account_plan_types:
                count = await self._count_active_script(keys=[account_key], args=[now])
                active_counts[account_key] = max(0, int(count))
        else:
            async with self._local_lock:
                active_counts = {
                    account_key: len(leases)
                    for account_key, leases in self._local_leases.items()
                }

        accounts: List[Dict[str, Any]] = []
        for account_key, plan_type in account_plan_types.items():
            active = active_counts.get(account_key, 0)
            if active == 0:
                continue
            limit = limits.get(plan_type)
            account_hash = account_key.rsplit(":", 1)[-1]
            accounts.append(
                {
                    "account_hash_prefix": account_hash[:12],
                    "plan_type": plan_type,
                    "active": active,
                    "limit": limit,
                    "remaining": max(0, limit - active) if limit is not None else None,
                }
            )

        accounts.sort(key=lambda item: (-item["active"], item["account_hash_prefix"]))
        return {
            "storage": "redis" if self._count_active_script is not None else "local",
            "lease_ttl_seconds": self.lease_ttl_seconds,
            "configured_limits": limits,
            "observed_account_count": len(account_plan_types),
            "active_account_count": len(accounts),
            "total_active": sum(item["active"] for item in accounts),
            "accounts": accounts,
        }

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Any] = None,
    ) -> List[dict]:
        # Atomic admission still happens after selection. Avoiding a non-atomic
        # pre-filter here ensures a concurrent release cannot hide a usable account.
        return healthy_deployments

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        limits = self._get_limits()
        if not limits:
            return None

        auth_file_path = kwargs.get("chatgpt_auth_file_path")
        if not isinstance(auth_file_path, str) or not auth_file_path:
            return None

        try:
            account_id, plan_type = self._read_auth_metadata(auth_file_path)
        except Exception as exc:
            verbose_proxy_logger.warning(
                "Skipping ChatGPT account concurrency limit for auth file %s: %s",
                os.path.basename(auth_file_path),
                exc,
            )
            return None

        limit = limits.get(plan_type)
        if limit is None:
            if plan_type not in self._warned_missing_plans:
                self._warned_missing_plans.add(plan_type)
                verbose_proxy_logger.warning(
                    "No ChatGPT account concurrency limit configured for plan_type=%s",
                    plan_type,
                )
            return None

        account_key = self._account_key(account_id)
        self._account_plan_types[account_key] = plan_type
        logging_obj = kwargs.get("litellm_logging_obj")
        if logging_obj is None or not hasattr(
            logging_obj, "add_async_deployment_cleanup_callback"
        ):
            raise RuntimeError(
                "ChatGPT account concurrency limiter could not register lease cleanup"
            )
        leased_account_keys = getattr(
            logging_obj, "_chatgpt_concurrency_leased_account_keys", None
        )
        if leased_account_keys is None:
            leased_account_keys = set()
            setattr(
                logging_obj,
                "_chatgpt_concurrency_leased_account_keys",
                leased_account_keys,
            )
        if account_key in leased_account_keys:
            return None

        lease = await self._acquire(account_key=account_key, limit=limit)
        if lease is None:
            error = litellm.RateLimitError(
                message=(
                    "ChatGPT provider account concurrency limit reached for "
                    f"plan_type={plan_type}; limit={limit}. Please retry in 10 seconds."
                ),
                llm_provider="chatgpt",
                model=str(kwargs.get("model") or ""),
                response=httpx.Response(
                    status_code=429,
                    headers={"retry-after": "10"},
                    request=httpx.Request("POST", "https://chatgpt.com/backend-api/"),
                ),
                num_retries=0,
            )
            setattr(error, "skip_deployment_cooldown", True)
            setattr(error, "is_provider_account_concurrency_limit", True)
            raise error

        leased_account_keys.add(account_key)

        async def release_lease() -> None:
            try:
                await lease.release()
            finally:
                leased_account_keys.discard(account_key)

        logging_obj.add_async_deployment_cleanup_callback(release_lease)
        return None
