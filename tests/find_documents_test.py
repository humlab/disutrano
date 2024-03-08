import pandas as pd
from penelope.corpus import render as rt
from penelope.notebook import topic_modelling as ntm

from notebooks.source.state_on_load import assign_pivot_keys_on_load


def test_update_state_on_load():

    state: ntm.TopicModelContainer = ntm.TopicModelContainer()
    state.load(folder='tests/test_data/test-corpus.gensim_mallet-lda', slim=True)

    assert set(state.inferred_topics.document_index.columns).intersection({'media_id', 'source_id'}) == set()
    assert isinstance(state.inferred_topics.corpus_config.pivot_keys_specs, str)

    state: ntm.TopicModelContainer = ntm.TopicModelContainer()
    state.register(None, callback=assign_pivot_keys_on_load)
    state.load(folder='tests/test_data/test-corpus.gensim_mallet-lda', slim=True)

    assert set(state.inferred_topics.document_index.columns).intersection({'media_id', 'source_id'}) == {
        'media_id',
        'source_id',
    }
    assert isinstance(state.inferred_topics.corpus_config.pivot_keys_specs, dict)


def test_find_topic_documents():

    state: ntm.TopicModelContainer = ntm.TopicModelContainer()
    state.register(None, callback=assign_pivot_keys_on_load)

    state.load(folder='tests/test_data/test-corpus.gensim_mallet-lda', slim=True)
    text_repository: rt.ITextRepository = rt.TextRepository(
        source='tests/test_data/test-corpus.zip',
        document_index=state.inferred_topics.document_index_proper,
    )

    gui: ntm.WithPivotKeysText.FindTopicDocumentsGUI = ntm.WithPivotKeysText.FindTopicDocumentsGUI(state=state)
    gui.setup()

    layout = gui.layout()
    assert layout is not None

    document_name: str = 'news_øresund_världens_första_klimatneutrala_cementfabrik_kan_byggas_på_go_202106101116_webb'

    expected_text: str = text_repository.get_text(f'{document_name}.txt')

    gui.content_type = 'text'
    gui.on_row_click(item={'document_name': document_name}, g=None)

    assert gui.text_output.value == expected_text

    gui.content_type = 'html'
    gui.on_row_click(item={'document_name': document_name}, g=None)

    assert gui.text_output.value.startswith('<h3>')

    gui.render_service.template = '{{document_id}} {{document_name}}: {{text}}'
    gui.on_row_click(item={'document_name': document_name}, g=None)

    assert gui.text_output.value == f"0 {document_name}: {expected_text}"

    gui._find_text.value = 'grön'  # pylint: disable=protected-access
    data: pd.DataFrame = gui.update()

    assert len(data) == 1
    gui.update_handler()
    assert gui._alert.value == '✅'  # pylint: disable=protected-access
