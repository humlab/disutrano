import pandas as pd
from penelope import topic_modelling as tm
from penelope.corpus import render as rt
from penelope.notebook import topic_modelling as ntm

{
    'title': 'Här är dystrast utsikter i landet',
    'source': 'Affärslivgotland Premium',
    'date': '2019-09-24 07:00',
    'media': 'webb',
    'pages': None,
    'url': 'http://ret.nu/aYeTSLaP',
    'date_time': '2019-09-24 07:00:00',
    'document_name': 'affärslivgotland_premium_här_är_dystrast_utsikter_i_landet_201909240700_webb',
    'filename': 'affärslivgotland_premium_här_är_dystrast_utsikter_i_landet_201909240700_webb.txt',
    'year': 2019,
    'input_file': 'Retriever_export_XIX.txt',
    'id': 19119,
    'document_id': 5918,
    'text': 'Här är dystrast utsikter i landet\nLågkonjunkturen kommer krypande in över Sverige, spår Nordea i en ny prognos.\nSmåland, norra Sverige och norra Mellansverige drabbas allra värst. Där väntar till \noch med negativa tillväxttal under 2019-20. Globalt sett väntar allt dystrare \nekonomiska tider. Då inkluderas även Sverige och de allra flesta ekonomer spår \nsåväl lägre tillväxt som högre arbetslöshet för svensk del. På regional nivå slår \ndock den stundande lågkonjunkturen olika. Storbanken Nordea har tittat på regionala \nutsikter och identifierat tre områden där det är extra dystert: Småland och \nGotland/Öland, och regionen från Gävleborg och Dalarna upp till Norrbottens län \nspås få det allra tuffast. I norra Sverige väntas regionens ekonomi krympa med \n0,2 procent kommande år, i Småland spås utvecklingen bli minus 0,4 procent 2020. \nFlera saker gemensamt Regionerna har flera saker gemensamt: de är väldigt omvärldsberoende \noch industrin är väldigt viktig i de här regionerna. Dessutom har de \nen svag befolkningstillväxt. Antingen minskar befolkningen i arbetsför ålder eller \nså ligger den stilla, säger Susanne Spector, senioranalytiker på Nordea till \nTT. Just utvecklingen i Norrland ger också, enligt Nordeas analytiker, en föraning \nom vad väntar för övriga Sverige. Som det går för Norrland går det också för \növriga Sverige. Och även om det finns ljusglimtar i form av till exempel Northvolts \nbatterifabrik i Skellefteå, är det överlag mörkt för norrländsk del. Svagast \ni klassen Den positiva jobbtillväxten i regionen har dessutom upphört och regionens \nsysselsättning förväntas enligt rapporten att utvecklas svagast i Sverige \nunder 2019-2020. Man kan nästan bli lite förvånad själv över hur bra indikator \nNorrland är på utvecklingen för resten av riket. Sverige är fortfarande väldigt \nomvärldsberoende. Det pratas mycket om tjänstesektorn men i grunden är Sverige en \nliten exportberoende nation som påverkas mycket av globala svängningar och det \nmärks först i Norrland, säger Susanne Spector. Image-text: Susanne Spector, senioranalytiker \npå Nordea. För smålänningar och norrlänningar väntar dystrare tider, \ni alla fall enligt Nordeas konjunkturrapport. Arkivbild. Extra-info: Fakta Fakta: \nPrognos för Sveriges olika regioner Måtten anges i BRP som står för bruttoregionalprodukt \noch motsvarar tillväxttal (BNP) för riket Region 2017 2018 2019 2020 \n2021 Sverige 2,1 2,4 1,3 1,2 1,7 Västsverige 3,0 2,6 1,5 1,3 1,8 Sydsverige \n3,6 2,2 1,1 0,8 1,3 Småland med Gotland/Öland 2,7 1,8 - 0,2 - 0,4 0,2 Stockholm \n0,9 3,2 2,4 2,6 2,8 Östra Mellansverige 2,9 1,9 1,1 0,8 1,3 Norra Sverige 1,7 1,2 \n- 0,3 - 0,2 0,5 Källa: Nordea',
}


def test_find_topic_documents():

    state: ntm.TopicModelContainer = ntm.TopicModelContainer()
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

    gui._find_text.value = 'grön'
    data: pd.DataFrame = gui.update()

    assert len(data) == 1
    gui.update_handler()
    assert gui._alert.value == '✅'

    assert set(gui.pivot_keys.id_names) == {'media_id', 'source_id'}
    assert set(gui.pivot_keys.text_names) == {'media', 'source'}
