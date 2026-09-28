from collections.abc import Hashable
from typing import Any

from ezrl.core.common import AgentID
from ezrl.core.episode import Episode


class EpisodeBuffer:
    def append(self, episode: Episode): ...
    def append_step(
        self,
        episode_id: Hashable,
        actions: dict[AgentID, Any],
        observations: dict[AgentID, Any],
        rewards: dict[AgentID, Any],
        terminateds: dict[AgentID, bool],
        truncateds: dict[AgentID, bool],
    ): ...
