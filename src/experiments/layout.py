import pyterrier as pt
import pandas as pd
import ir_datasets

from tira.third_party_integrations import ensure_pyterrier_is_loaded
from tira.rest_api_client import Client
from util.utility import *

tira = Client()
ensure_pyterrier_is_loaded()


class Layout:
    def __init__(self, name, dsets, flan, llama):
        self.flan = flan
        self.llama = llama
        self.dsets = dsets
        self.name = name

    def run_all(self):
        self.run_chain_of_thoughts()
        self.run_similar_queries_fs()
        self.run_similar_queries_zs()

    def run_chain_of_thoughts(self):
        for dset_name in self.dsets:
            dataset = ir_datasets.load(f'ir-benchmarks/{dset_name}')

            outputs = self.flan.chain_of_thoughts(list(dataset.queries_iter()))
            save_queries("chain-of-thoughts", "flan-ul2", dset_name, list(dataset.queries_iter()), outputs)

            outputs = self.llama.chain_of_thoughts(list(dataset.queries_iter()))
            save_queries("chain-of-thoughts", "llama", dset_name, list(dataset.queries_iter()), outputs)

    def run_similar_queries_fs(self):
        for dset_name in self.dsets:
            dataset = ir_datasets.load(f'ir-benchmarks/{dset_name}')

            outputs = self.flan.similar_queries_fs(list(dataset.queries_iter()))
            save_queries("similar-queries-fs", "flan-ul2", dset_name, list(dataset.queries_iter()), outputs)

            outputs = self.llama.similar_queries_fs(list(dataset.queries_iter()))
            save_queries("similar-queries-fs", "llama", dset_name, list(dataset.queries_iter()), outputs)

    def run_similar_queries_zs(self):
        for dset_name in self.dsets:
            dataset = ir_datasets.load(f'ir-benchmarks/{dset_name}')

            outputs = self.flan.similar_queries_zs(list(dataset.queries_iter()))
            save_queries("similar-queries-zs", "flan-ul2", dset_name, list(dataset.queries_iter()), outputs)

            outputs = self.llama.similar_queries_zs(list(dataset.queries_iter()))
            save_queries("similar-queries-zs", "llama", dset_name, list(dataset.queries_iter()), outputs)

    def eval_all(self):
        for exp_name in ['chain-of-thoughts', 'similar-queries-fs', 'similar-queries-zs']:
            for model_name in ['flan-ul2', 'llama']:
                self.do_evaluation(exp_name, model_name)

    def eval(self, exp_names, model_names):
        for exp_name in exp_names:
            for model_name in model_names:
                self.do_evaluation(exp_name, model_name)

    def do_evaluation(self, exp_name, model_name):
        eval_dfs = []
        for dset_name in self.dsets:
            expanded_queries = Layout.get_as_dict(exp_name, model_name, dset_name)
            pt_dataset = pt.get_dataset(f"irds:ir-benchmarks/{dset_name}")

            pt_expand_query = pt.apply.generic(lambda i: Layout.expand_queries_in_dataframe(i, expanded_queries))

            index = Layout.pyterrier_index_from_tira(dset_name)

            bm25 = pt.BatchRetrieve(index, wmodel="BM25")
            bm25_with_expansion = pt_expand_query >> bm25

            df = pt.Experiment([bm25, bm25_with_expansion], pt_dataset.get_topics('query'), pt_dataset.get_qrels(),
                               ['ndcg_cut.10', 'recall_1000'], names=['BM25', f'BM25+{self.name}'], verbose=True)
            df['dataset'] = dset_name
            df['model'] = model_name
            eval_dfs.append(df)

        eval_df = pd.concat(eval_dfs)
        eval_df.to_json(f'chain-of-thoughts/generated/eval-{model_name}.jsonl', lines=True, orient='records')

    @staticmethod
    def concat(query, reps, llm_output):
        return f'{query} ' * reps + llm_output

    @staticmethod
    def pyterrier_index_from_tira(dataset):
        ret = tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', dataset) + '/index'
        return pt.IndexFactory.of(ret)

    @staticmethod
    def get_as_dict(exp_name, model, dset_name):
        json_res = read_queries(exp_name, model, dset_name)
        tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

        def pt_tokenize(text):
            return ' '.join(tokeniser.getTokens(text))

        expanded_queries = {q['query-id']: pt_tokenize(Layout.concat(q['query-text'], 5, q['response'])) for q in json_res}
        return expanded_queries

    @staticmethod
    def expand_queries_in_dataframe(df, expanded_queries):
        df['text'] = df['qid'].apply(lambda x: expanded_queries[x])
        df['query'] = df['qid'].apply(lambda x: expanded_queries[x])
        return df
