import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    initialize_standard_callback_dynamic_params,
)


def test_dynamic_key_extraction_from_metadata():
    """
    Test extraction of langfuse keys from metadata in kwargs.
    This simulates a Proxy request where keys are passed in metadata.
    """
    kwargs = {
        "metadata": {
            "langfuse_public_key": "pk-test",
            "langfuse_secret_key": "sk-test",
            "langfuse_host": "https://test.langfuse.com",
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-test"
    assert params.get("langfuse_secret_key") == "sk-test"
    assert params.get("langfuse_host") == "https://test.langfuse.com"


def test_dynamic_key_extraction_from_litellm_params_metadata():
    """
    Test extraction of langfuse keys from litellm_params.metadata.
    """
    kwargs = {
        "litellm_params": {
            "metadata": {
                "langfuse_public_key": "pk-litellm",
                "langfuse_secret_key": "sk-litellm",
            }
        }
    }

    params = initialize_standard_callback_dynamic_params(kwargs)

    assert params.get("langfuse_public_key") == "pk-litellm"
    assert params.get("langfuse_secret_key") == "sk-litellm"


def test_request_environment_references_are_not_resolved():
    params = initialize_standard_callback_dynamic_params(
        {
            "langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY",
            "metadata": {"langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY"},
        }
    )

    assert "langfuse_secret_key" not in params
    assert "langfuse_public_key" not in params


if __name__ == "__main__":
    test_dynamic_key_extraction_from_metadata()
    test_dynamic_key_extraction_from_litellm_params_metadata()
