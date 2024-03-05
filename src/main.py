#!/usr/bin/env python

from src.experiments.experiments import ChainOfThoughts, SimilarQueriesFS, SimilarQueriesZS
from src.models.flan_ul2 import FlanUL2Wrapper
from src.models.llama import Llama2Wrapper
from src.util.utility import get_all_datasets, get_similar

if __name__ == '__main__':
    dset_list = [get_similar('marco')]

    if len(dset_list) == 0:
        dset_list = get_all_datasets()

    flan_model = FlanUL2Wrapper(min_len=10, max_len=200, temperature=0.5)
    llama_model = Llama2Wrapper(min_len=10, max_len=200, temperature=1.1)

    chain_of_thoughts = ChainOfThoughts(name="CoT", flan=flan_model, llama=llama_model, dsets=dset_list)
    chain_of_thoughts.run_all()
    chain_of_thoughts.eval(["chain-of-thoughts"], ["llama", "flan-ul2"])

    similar_queries_fs = SimilarQueriesFS(name="Q2E/FS", flan=flan_model, llama=llama_model, dsets=dset_list)
    similar_queries_fs.run_all()
    similar_queries_fs.eval(["chain-of-thoughts"], ["llama", "flan-ul2"])

    similar_queries_zs = SimilarQueriesZS(name="Q2E/ZS", flan=flan_model, llama=llama_model, dsets=dset_list)
    similar_queries_zs.run_all()
    similar_queries_zs.eval(["chain-of-thoughts"], ["llama", "flan-ul2"])
