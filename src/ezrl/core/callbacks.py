from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ezrl.core.episode import Episode, Step

type EpisodeCompletedCallback = Callable[["Episode"], None]
type EpisodeStepCallback = Callable[["Episode", "Step"], None]

type Callbacks[CallbackType] = (
    dict[Hashable, CallbackType] | list[CallbackType] | CallbackType
)


def run_callbacks(
    callbacks: Callbacks,
    **kwargs,
):
    if isinstance(callbacks, dict):
        callbacks_list = list(callbacks.values())
    elif isinstance(callbacks, list):
        callbacks_list = callbacks
    else:
        callbacks_list = [callbacks]

    for callback in callbacks_list:
        callback(**kwargs)
