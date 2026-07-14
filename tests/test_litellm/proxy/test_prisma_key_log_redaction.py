import hashlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import PrismaClient


@pytest.mark.asyncio
async def test_insert_data_debug_log_hashes_virtual_key(caplog):
    token = "sk-short-secret"
    expected_hash = hashlib.sha256(token.encode()).hexdigest()
    client = object.__new__(PrismaClient)
    client.db = MagicMock()
    client.db.litellm_verificationtoken.upsert = AsyncMock(
        return_value=SimpleNamespace(token=expected_hash)
    )

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        await client.insert_data(
            data={"token": token, "key_alias": "redaction-repro"},
            table_name="key",
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert token not in log_text
    assert expected_hash in log_text
