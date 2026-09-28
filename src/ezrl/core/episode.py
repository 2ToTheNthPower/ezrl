from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ezrl.core.callbacks import (
    Callbacks,
    EpisodeCompletedCallback,
    EpisodeStepCallback,
    run_callbacks,
)
from ezrl.core.common import ALL_AGENTS, AgentID, PolicyID


@dataclass
class AgentStep:
    action: Any
    observation: Any
    reward: Any  # intentionally not just dict[AgentID, float]
    terminated: bool
    truncated: bool
    info: Any

    # Which policy made the action
    policy_id: PolicyID
    time: float


@dataclass
class AgentEpisode:
    actions: list[Any] = field(default_factory=list)
    observations: list[Any] = field(default_factory=list)
    rewards: list[Any] = field(default_factory=list)
    infos: list[Any] = field(default_factory=list)

    times: list[float] = field(default_factory=list)
    policy_ids: list[PolicyID] = field(default_factory=list)

    terminated: bool = False
    truncated: bool = False

    def append(
        self,
        step: AgentStep,
    ):
        self.assert_not_done()

        self.actions.append(step.action)
        self.observations.append(step.observation)
        self.rewards.append(step.reward)
        self.infos.append(step.info)

        self.times.append(step.time)
        self.policy_ids.append(step.policy_id)

        self.terminated = self.terminated or step.terminated
        self.truncated = self.truncated or step.truncated

    def assert_not_done(self):
        if self.terminated or self.truncated:
            raise RuntimeError("Appending to completed AgentEpisodes is not allowed.")


class Episode(BaseModel):
    actions: dict[AgentID, list[Any]] = defaultdict(list)
    observations: dict[AgentID, list[Any]] = defaultdict(list)
    rewards: dict[AgentID, list[Any]] = defaultdict(list)
    terminateds: dict[AgentID, bool] = defaultdict(bool)
    truncateds: dict[AgentID, bool] = defaultdict(bool)
    infos: dict[AgentID, list[Any]] = defaultdict(list)

    mappings: dict[AgentID, list[PolicyID]]
    times: dict[AgentID, list[float]]

    on_episode_completed_callbacks: Callbacks[EpisodeCompletedCallback]
    on_episode_step_callbacks: Callbacks[EpisodeStepCallback]

    terminated: bool = False
    truncated: bool = False

    _active_agents: set[AgentID]
    _completed: bool = False

    @property
    def completed(self):
        return self._completed

    @completed.setter
    def completed(self, new_value: bool):
        self._completed = new_value

        if new_value is True:
            run_callbacks(self.on_episode_completed_callbacks, episode=self)

    def append(
        self,
        step: Step,
    ):
        if self.completed:
            raise RuntimeError(
                "Cannot append to episode that has been marked as completed."
            )

        for agent_id in step.mappings:
            if self.terminateds[agent_id]:
                raise ValueError(
                    f"Received agent_id={agent_id} which has already terminated for this episode."
                )

            if self.truncateds[agent_id]:
                raise ValueError(
                    f"Received agent_id={agent_id} which has already truncated for this episode."
                )

            self._active_agents.add(agent_id)

            self.actions[agent_id].append(step.actions[agent_id])
            self.observations[agent_id].append(step.observations[agent_id])
            self.rewards[agent_id].append(step.rewards[agent_id])
            self.infos[agent_id].append(step.infos[agent_id])

            self.times[agent_id].append(step.time)
            self.mappings[agent_id].append(step.mappings[agent_id])

            self.terminateds[agent_id] = (
                self.terminateds[agent_id] or step.terminateds[agent_id]
            )
            self.truncateds[agent_id] = (
                self.truncateds[agent_id] or step.truncateds[agent_id]
            )

        run_callbacks(self.on_episode_step_callbacks, episode=self, step=step)

        if step.terminateds[ALL_AGENTS]:
            self.terminated = step.terminateds[ALL_AGENTS] or self.terminated

            for agent_id in self._active_agents:
                self.terminateds[agent_id] = self.terminated

        if step.truncateds[ALL_AGENTS]:
            self.truncated = step.truncateds[ALL_AGENTS] or self.truncated

            for agent_id in self._active_agents:
                self.truncateds[agent_id] = self.truncated

        self.completed = self.terminated or self.truncated
