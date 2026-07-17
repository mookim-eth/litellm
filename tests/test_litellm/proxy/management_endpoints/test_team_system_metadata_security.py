from litellm.proxy.management_endpoints.team_endpoints import TeamMemberBudgetHandler


def test_team_member_budget_id_is_server_managed():
    metadata = {"team_member_budget_id": "attacker-value", "safe": "value"}

    TeamMemberBudgetHandler.strip_system_managed_metadata_keys(metadata)

    assert metadata == {"safe": "value"}
