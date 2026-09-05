from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, UserAPIKeyAuth
from litellm.proxy.hooks.astra_team_access import ASTRA_TEAM_ID, AstraTeamAccess


@pytest.fixture
def team_lookup(monkeypatch):
    from litellm.proxy import proxy_server

    lookup = AsyncMock(
        return_value=LiteLLM_TeamTableCachedObj(
            team_id=ASTRA_TEAM_ID,
            members_with_roles=[{"user_id": "member", "role": "user"}],
        )
    )
    monkeypatch.setattr(
        proxy_server,
        "prisma_client",
        SimpleNamespace(
            db=SimpleNamespace(litellm_teamtable=SimpleNamespace(find_unique=lookup))
        ),
    )
    return lookup


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["gpt-6-astra", "gpt-6-astra-1", "gpt-6-astra-2", "chatgpt/gpt-6-astra"]
)
@pytest.mark.parametrize(
    "models", [[], ["all-proxy-models"], ["all-team-models"], ["*"]]
)
async def test_should_deny_nonmembers_regardless_of_model_allowlist(
    team_lookup, model, models
):
    with pytest.raises(HTTPException) as exc:
        await AstraTeamAccess._check_access(
            model, UserAPIKeyAuth(user_id="outsider", models=models)
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_id,team_id",
    [
        ("member", None),
        ("member", "other"),
        ("outsider", ASTRA_TEAM_ID),
        (None, ASTRA_TEAM_ID),
    ],
)
async def test_should_allow_user_membership_or_key_team(team_lookup, user_id, team_id):
    await AstraTeamAccess._check_access(
        "gpt-6-astra", UserAPIKeyAuth(user_id=user_id, team_id=team_id)
    )


@pytest.mark.asyncio
async def test_should_not_exempt_admins_or_trust_forged_identity(team_lookup):
    for auth in (
        None,
        {"user_id": "member", "team_id": ASTRA_TEAM_ID},
        UserAPIKeyAuth(user_id="outsider", user_role="proxy_admin"),
    ):
        with pytest.raises(HTTPException):
            await AstraTeamAccess._check_access("gpt-6-astra", auth)


@pytest.mark.asyncio
async def test_should_fail_closed_when_team_blocked_deleted_or_unavailable(team_lookup):
    auth = UserAPIKeyAuth(user_id="member", team_id=ASTRA_TEAM_ID)
    team_lookup.return_value.blocked = True
    with pytest.raises(HTTPException):
        await AstraTeamAccess._check_access("gpt-6-astra", auth)
    team_lookup.side_effect = RuntimeError("database unavailable")
    with pytest.raises(HTTPException):
        await AstraTeamAccess._check_access("gpt-6-astra", auth)
    team_lookup.side_effect = None
    team_lookup.return_value = None
    with pytest.raises(HTTPException):
        await AstraTeamAccess._check_access("gpt-6-astra", auth)


@pytest.mark.asyncio
async def test_should_leave_other_models_unchanged_without_db_lookup(team_lookup):
    await AstraTeamAccess._check_access("chatgpt/gpt-5.6-luna", None)
    team_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_should_recheck_membership_after_removal(team_lookup):
    auth = UserAPIKeyAuth(user_id="member")
    await AstraTeamAccess._check_access("gpt-6-astra", auth)
    team_lookup.return_value.members_with_roles = []
    with pytest.raises(HTTPException):
        await AstraTeamAccess._check_access("gpt-6-astra", auth)


@pytest.mark.asyncio
async def test_should_keep_authenticated_identity_outside_fallback_metadata(
    team_lookup,
):
    hook = AstraTeamAccess()
    logging_obj = SimpleNamespace(_astra_request_auth=UserAPIKeyAuth(user_id="member"))
    await hook.async_pre_call_hook(
        UserAPIKeyAuth(user_id="outsider"),
        DualCache(),
        {
            "model": "gpt-5.6-luna",
            "litellm_logging_obj": logging_obj,
            "_astra_request_auth": {"user_id": "member"},
        },
        "acompletion",
    )
    with pytest.raises(HTTPException):
        await hook.async_pre_call_deployment_hook(
            {
                "model": "chatgpt/gpt-6-astra",
                "litellm_logging_obj": logging_obj,
                "metadata": {
                    "user_api_key_auth": {"user_id": "member", "team_id": ASTRA_TEAM_ID}
                },
            },
            None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("member", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_should_enforce_actual_router_fallback_before_upstream(
    team_lookup, monkeypatch, member, stream
):
    from litellm.litellm_core_utils.litellm_logging import Logging

    hook = AstraTeamAccess()
    monkeypatch.setattr(litellm, "callbacks", [hook])
    router = litellm.Router(
        model_list=[
            {
                "model_name": "ordinary",
                "litellm_params": {
                    "model": "openai/gpt-5.6-luna",
                    "api_key": "test",
                    "mock_response": "litellm.RateLimitError",
                },
            },
            {
                "model_name": "private-alias",
                "litellm_params": {
                    "model": "chatgpt/gpt-6-astra",
                    "mock_response": "astra response",
                },
            },
        ],
        fallbacks=[{"ordinary": ["private-alias"]}],
        num_retries=0,
    )
    logging_obj = Logging(
        model="ordinary",
        messages=[],
        stream=False,
        call_type="acompletion",
        start_time=None,
        litellm_call_id="astra-test",
        function_id="astra-test",
    )
    await hook.async_pre_call_hook(
        UserAPIKeyAuth(user_id="member" if member else "outsider"),
        DualCache(),
        {"model": "ordinary", "litellm_logging_obj": logging_obj},
        "acompletion",
    )
    call = router.acompletion(
        model="ordinary",
        messages=[{"role": "user", "content": "hello"}],
        litellm_logging_obj=logging_obj,
        stream=stream,
    )
    if member:
        response = await call
        if stream:
            assert [chunk async for chunk in response]
            await response.aclose()
        else:
            assert response.choices[0].message.content == "astra response"
    else:
        with pytest.raises(Exception, match="astra_team") as exc:
            await call
        # Router preserves the original ordinary-model 429 when all fallbacks
        # fail. The denied Astra attempt is included in its error, not invoked.
        assert exc.value.status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("member", [False, True])
async def test_should_use_authenticated_membership_on_public_inference_routes(
    team_lookup, monkeypatch, path, stream, member
):
    from litellm.proxy import proxy_server as ps
    from litellm.proxy.utils import ProxyLogging

    if path == "/v1/responses" and stream:
        # The built-in string mock returns a non-streaming Responses object.
        # Supply a streaming provider mock, preserving both authorization hooks.
        import importlib

        from litellm.types.llms.openai import ResponseCompletedEvent

        responses_main = importlib.import_module("litellm.responses.main")
        completed = responses_main.mock_responses_api_response("allowed")

        class MockResponseStream:
            _hidden_params = {}

            async def __aiter__(self):
                yield ResponseCompletedEvent(
                    type="response.completed", response=completed, sequence_number=0
                )

            async def aclose(self):
                pass

        monkeypatch.setattr(
            responses_main,
            "mock_responses_api_response",
            lambda **kw: MockResponseStream(),
        )

    hook = AstraTeamAccess()
    monkeypatch.setattr(litellm, "callbacks", [hook])
    monkeypatch.setattr(
        ps, "proxy_logging_obj", ProxyLogging(user_api_key_cache=DualCache())
    )
    monkeypatch.setattr(
        ps,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-6-astra",
                    "litellm_params": {
                        "model": "chatgpt/gpt-6-astra",
                        "mock_response": "should not run",
                    },
                }
            ]
        ),
    )
    auth = UserAPIKeyAuth(
        user_id="member" if member else "outsider", models=["all-proxy-models"]
    )
    monkeypatch.setitem(ps.app.dependency_overrides, ps.user_api_key_auth, lambda: auth)
    forged = {
        "user_api_key_user_id": "member",
        "user_api_key_team_id": ASTRA_TEAM_ID,
        "user_api_key_auth": {"user_id": "member", "team_id": ASTRA_TEAM_ID},
    }
    data = {
        "model": "gpt-6-astra",
        "stream": stream,
        "user": "member",
        "team_id": ASTRA_TEAM_ID,
        "metadata": forged,
        "litellm_metadata": forged,
        "_astra_request_auth": forged,
        "litellm_logging_obj": {"_astra_request_auth": forged},
        "max_tokens": 10,
    }
    if path == "/v1/responses":
        data["input"] = "hello"
    else:
        data["messages"] = [{"role": "user", "content": "hello"}]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=ps.app), base_url="http://test"
    ) as client:
        response = await client.post(
            path, json=data, headers={"Authorization": "Bearer test"}
        )
    if member:
        assert response.status_code == 200, response.text
        assert "astra_team" not in response.text
    else:
        assert response.status_code == 403, response.text
        assert "astra_team" in response.text
