from typing import Callable

from penelope import utility as pu
from penelope.notebook import topic_modelling as tm

FilenameFieldSpec = list[str] | dict[str, Callable, str]


def assign_pivot_keys_on_load(state: tm.TopicModelContainer, *_, **__) -> None:
    """Create and assign pivot keys to the state when a topic model is loaded.
    Encode textual pivot key columns to integer codes.
    Add pivot keys to the state inferred_topics.corpus_config.extra_opts.
    The pivot keys are used to filter the document index by media and source.
    """

    pivot_keys: pu.PivotKeys = pu.PivotKeys.create_by_index(state.inferred_topics.document_index, "media", "source")

    state.inferred_topics.corpus_config.extra_opts['pivot_keys'] = pivot_keys.pivot_keys_spec
