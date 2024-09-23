import pandas as pd
from penelope import topic_modelling as ntm

INPUT_FOLDER: str = 'data/tmp/20240409_140-TF2-MP0.02-5000000-lower-no-stopwords-lemma.gensim_mallet-lda-text'
OUTPUT_FOLDER: str = 'data/tmp/CLUSTER_20240409_140-TF2-MP0.02-5000000-lower-no-stopwords-lemma.gensim_mallet-lda-text'


# def test_merge_inferred_topics():

#     clusers_filename: str = 'data/tmp/clusters.tsv'
#     input_folder: str = INPUT_FOLDER
#     output_folder: str = OUTPUT_FOLDER

#     merge_topics_to_clusters(input_folder, clusers_filename, output_folder)

# def merge_topics_to_clusters(clusers_filename: str, input_folder: str, output_folder: str):
#     cluster_mapping: dict[str, list[int]] = (
#         pd.read_csv(clusers_filename, sep='\t').groupby('cluster_name')['topic_id'].agg(list).to_dict()
#     )

#     inferred_topics: ntm.InferredTopicsData = ntm.InferredTopicsData.load(folder=input_folder, slim=True)

#     assert inferred_topics is not None

#     inferred_topics.merge(cluster_mapping)

#     inferred_topics.compress()

#     shutil.rmtree(output_folder, ignore_errors=True)

#     inferred_topics.store(target_folder=output_folder)

#     shutil.copy(f'{input_folder}/model_options.json', f'{output_folder}/model_options.json')
#     shutil.copy(clusers_filename, f'{output_folder}/clusters.tsv')


def test_using_simple_fake_data():
    document_index: pd.DataFrame = pd.DataFrame(
        columns=['document_id', 'document_name', 'n_tokens'],
        data=[
            [0, 'doc1', 99],
            [1, 'doc2', 99],
            [2, 'doc3', 99],
            [3, 'doc4', 99],
            [4, 'doc5', 99],
        ],
    ).set_index('document_id')

    dictionary: pd.DataFrame = pd.DataFrame(
        columns=['token_id', 'token'],
        data=[
            [0, 'a'],
            [1, 'b'],
            [2, 'c'],
            [3, 'd'],
            [4, 'e'],
        ],
    ).set_index('token_id')

    document_topic_weights: pd.DataFrame = pd.DataFrame(
        columns=['document_id', 'topic_id', 'weight'],
        data=[
            # doc1: a a a a b
            [0, 0, 0.8],
            [0, 1, 0.2],
            # doc2: c c c c d
            [1, 2, 0.8],
            [1, 3, 0.2],
            # doc3: c c c c c c b b b a
            [2, 2, 0.6],
            [2, 1, 0.3],
            [2, 0, 0.1],
            # doc4: d d d d c
            [3, 4, 0.8],
            [3, 3, 0.2],
            # doc5: d
            [4, 4, 1.0],
        ],
    )

    topic_token_weights: pd.DataFrame = pd.DataFrame(
        columns=['topic_id', 'token_id', 'weight'],
        data=[
            # t0: a a a b c
            [0, 0, 0.6],
            [0, 1, 0.2],
            [0, 2, 0.2],
            # t1:  d b c
            [1, 3, 0.8],
            [1, 1, 0.1],
            [1, 2, 0.1],
            # t2:  b b b b b b a a c d
            [2, 1, 0.6],
            [2, 0, 0.2],
            [2, 2, 0.1],
            [2, 3, 0.1],
            # t3:  b b b b b b b b b c
            [3, 1, 0.9],
            [3, 2, 0.1],
            # t3:  a a d e
            [4, 0, 0.4],
            [4, 2, 0.2],
            [4, 4, 0.2],
        ],
    )
    token2id: dict[int, str] = dictionary['token'].to_dict()  # pylint: disable=unsubscriptable-object
    topic_token_overview: pd.DataFrame = ntm.compute_topic_token_overview(topic_token_weights, token2id, 3)
    topic_token_overview['label'] = ['T1', 'T2', 'T3', 'T4', 'T5']
    inferred_topics: ntm.InferredTopicsData = ntm.InferredTopicsData(
        document_index=document_index,
        dictionary=dictionary,
        document_topic_weights=document_topic_weights,
        topic_token_weights=topic_token_weights,
        topic_token_overview=topic_token_overview,
        topic_diagnostics=None,
        token_diagnostics=None,
    )

    assert inferred_topics is not None

    cluster_mapping: dict[str, list[int]] = {
        'C1': [0, 1],
        'C2': [2, 3],
        'C3': [4],
    }
    expected_document_topic_weights = pd.DataFrame(
        columns=['document_id', 'topic_id', 'weight'],
        data=[
            [0, 0, 1.0],
            [1, 2, 1.0],
            [2, 0, 0.4],
            [2, 2, 0.6],
            [3, 2, 0.2],
            [3, 4, 0.8],
            [4, 4, 1.0],
        ],
    )
    inferred_topics.merge(cluster_mapping)

    assert set(inferred_topics.document_topic_weights.topic_id.unique()) == {0, 2, 4}
    assert set(inferred_topics.topic_token_weights.topic_id.unique()) == {0, 2, 4}

    assert inferred_topics.topic_token_overview.label.tolist() == ['C1', 'T2', 'C2', 'T4', 'T5']
    assert inferred_topics.document_topic_weights.equals(expected_document_topic_weights)

    inferred_topics.compress()

    assert set(inferred_topics.document_topic_weights.topic_id.unique()) == {0, 1, 2}
    assert inferred_topics.topic_token_overview.label.tolist() == ['C1', 'C2', 'T5']

    expected_compressed_document_topic_weights = expected_document_topic_weights.copy()
    expected_compressed_document_topic_weights['topic_id'] = (
        expected_compressed_document_topic_weights.topic_id.replace({0: 0, 1: 0, 2: 1, 3: 1, 4: 2})
    )

    assert inferred_topics.document_topic_weights.equals(expected_compressed_document_topic_weights)
