from unittest.mock import patch

from litellm.proxy.auth.handle_jwt import JWTHandler


def test_jwt_decode_kwargs_include_configured_issuer(monkeypatch):
    monkeypatch.setenv("JWT_AUDIENCE", "litellm-api")
    monkeypatch.setenv("JWT_ISSUER", "https://trusted-idp.example/tenant")

    kwargs = JWTHandler._build_decode_kwargs()

    assert kwargs["audience"] == "litellm-api"
    assert kwargs["issuer"] == "https://trusted-idp.example/tenant"
    assert kwargs["options"] is None


def test_unscoped_jwt_warning_emitted_once(monkeypatch):
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    monkeypatch.setattr(JWTHandler, "_unscoped_jwt_warning_emitted", False)

    with patch("litellm.proxy.auth.handle_jwt.verbose_proxy_logger") as logger:
        first = JWTHandler._build_decode_kwargs()
        second = JWTHandler._build_decode_kwargs()

    assert first["options"] == {"verify_aud": False, "verify_iss": False}
    assert second == first
    logger.warning.assert_called_once()
