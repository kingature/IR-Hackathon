from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
import pyterrier as pt
import pandas as pd
from utility import *

tira = Client()
ensure_pyterrier_is_loaded()


def concat(query, reps, llm_output):
    return f'{query} ' * reps + llm_output


def pyterrier_index_from_tira(dataset):
    ret = tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', dataset) + '/index'
    return pt.IndexFactory.of(ret)


def read_and_expand_queries(model, dset_name):
    json_res = read_queries(model, dset_name)
    tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

    def pt_tokenize(text):
        return ' '.join(tokeniser.getTokens(text))

    expanded_queries = {q['query-id']: pt_tokenize(concat(q['query-text'], 5, q['response'])) for q in json_res}
    return expanded_queries


def expand_queries_in_dataframe(df, expanded_queries):
    df['text'] = df['qid'].apply(lambda x: expanded_queries[x])
    df['query'] = df['qid'].apply(lambda x: expanded_queries[x])
    return df


def do_evaluation(model, dset_list):
    eval_dfs = []
    for dset_name in dset_list:
        expanded_queries = read_and_expand_queries(model, dset_name)
        pt_dataset = pt.get_dataset(f"irds:ir-benchmarks/{dset_name}")

        pt_expand_query = pt.apply.generic(lambda i: expand_queries_in_dataframe(i, expanded_queries))

        index = pyterrier_index_from_tira(dset_name)

        bm25 = pt.BatchRetrieve(index, wmodel="BM25")
        bm25_with_expansion = pt_expand_query >> bm25

        df = pt.Experiment([bm25, bm25_with_expansion], pt_dataset.get_topics('query'), pt_dataset.get_qrels(),
                           ['ndcg_cut.10', 'recall_1000'], names=['BM25', 'BM25+CoT'], verbose=True)
        df['dataset'] = dset_name
        df['model'] = model
        eval_dfs.append(df)

    eval_df = pd.concat(eval_dfs)
    eval_df.to_json(f'chain-of-thoughts/generated/eval-{model}.jsonl', lines=True, orient='records')
