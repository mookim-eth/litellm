import pytest

from litellm.secret_managers.base_secret_manager import raise_if_unsafe_secret_name


@pytest.mark.parametrize("name", ["../secret", "path/../secret", "secret\nname"])
def test_unsafe_secret_names_are_rejected(name):
    with pytest.raises(ValueError):
        raise_if_unsafe_secret_name(name)


@pytest.mark.parametrize("name", ["safe/name", "release-1.0..2", "name@example"])
def test_safe_secret_names_remain_compatible(name):
    raise_if_unsafe_secret_name(name)
