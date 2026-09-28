from collections.abc import Hashable
from enum import StrEnum

PolicyID = Hashable
AgentID = Hashable

ALL_AGENTS = "__all_agents__"


class Completion(StrEnum):
    TERMINATED = "terminated"
    TRUNCATED = "truncated"
